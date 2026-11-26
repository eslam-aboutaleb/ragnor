"""LlamaIndex adapters against real stores, plus adversarial tenant isolation.

Every test runs on the in-memory store. Qdrant runs when
``RAGNOR_TEST_QDRANT_URL`` is set and pgvector when
``RAGNOR_TEST_DATABASE_URL`` is set (CI provides both as service containers).
No mocks: real servers, real llama-index-core, real ``IngestionPipeline.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.schema import Document, NodeRelationship, QueryBundle, TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from ragnor.embeddings import register_dimension  # noqa: E402
from ragnor.ingestion.pipeline import IngestionPipeline, VersionRecord  # noqa: E402
from ragnor.integrations.llamaindex import (  # noqa: E402
    LlamaIndexChunker,
    LlamaIndexDocumentSource,
    RagnorRetriever,
    RagnorVectorStore,
)
from ragnor.retrieval import DenseRetrieval, HybridRetrieval  # noqa: E402
from ragnor.stores.memory import InMemoryVectorStore  # noqa: E402

DIM = 64
DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
QDRANT_URL = os.environ.get("RAGNOR_TEST_QDRANT_URL")
MODEL = "llamaindex-hash-test"
register_dimension(MODEL, DIM)


def hash_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in text.lower().split():
        vec[hashlib.sha256(token.strip(".,").encode()).digest()[0] % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return hash_embed(text)


@dataclass
class Settings:
    embedding_model: str = MODEL
    embedding_provider: str = "test"


@dataclass
class Backend:
    name: str
    store: Any
    sessions: Any  # session factory for IngestionPipeline
    tenant_rows: Any  # async fn(tenant) -> int: rows physically stored for a tenant


class _Null:
    async def commit(self) -> None:
        return None


@asynccontextmanager
async def _null_sessions() -> AsyncIterator[_Null]:
    yield _Null()


PG_DDL = """CREATE TABLE chunks (
    id text PRIMARY KEY, text text NOT NULL, embedding vector({dim}),
    tenant_id text NOT NULL, node_content text, node_type text, ref_doc_id text,
    kind text, source text, external_id text, version_id text, source_id text,
    extractor text,
    tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
)"""


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "qdrant",
            marks=pytest.mark.skipif(not QDRANT_URL, reason="RAGNOR_TEST_QDRANT_URL not set"),
        ),
        pytest.param(
            "pgvector",
            marks=pytest.mark.skipif(not DB_URL, reason="RAGNOR_TEST_DATABASE_URL not set"),
        ),
    ]
)
async def backend(request: pytest.FixtureRequest) -> AsyncIterator[Backend]:
    if request.param == "memory":
        store = InMemoryVectorStore(embedding_dim=DIM, metric="cosine")

        async def rows(tenant: str) -> int:
            return await store.count(tenant_id=tenant)

        yield Backend("memory", store, _null_sessions, rows)
    elif request.param == "qdrant":
        from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

        from ragnor.stores.qdrant import QdrantStore  # noqa: PLC0415

        client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
        name = "li_" + uuid.uuid4().hex
        store = QdrantStore(table_name=name, embedding_dim=DIM, client=client)

        async def rows(tenant: str) -> int:
            return await store.count(tenant_id=tenant)

        try:
            yield Backend("qdrant", store, _null_sessions, rows)
        finally:
            await client.delete_collection(name)
            await client.close()
    else:
        from sqlalchemy import text  # noqa: PLC0415
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

        from ragnor.stores.pgvector import PgVectorStore  # noqa: PLC0415

        schema = "li_" + uuid.uuid4().hex
        admin = create_async_engine(DB_URL)
        async with admin.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text(f"CREATE SCHEMA {schema}"))  # noqa: S608
            await conn.execute(text(f"SET LOCAL search_path TO {schema}, public"))  # noqa: S608
            await conn.execute(text(PG_DDL.format(dim=DIM)))
        engine = create_async_engine(
            DB_URL, connect_args={"server_settings": {"search_path": f"{schema},public"}}
        )
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        store = PgVectorStore(
            table_name="chunks", embedding_dim=DIM, session_factory=sessions, metric="cosine"
        )

        async def rows(tenant: str) -> int:
            async with sessions() as s:
                res = await s.execute(
                    text("SELECT count(*) FROM chunks WHERE tenant_id = :t"), {"t": tenant}
                )
                return int(res.scalar_one())

        try:
            yield Backend("pgvector", store, sessions, rows)
        finally:
            await engine.dispose()
            async with admin.begin() as conn:
                await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))  # noqa: S608
            await admin.dispose()


def node(node_id: str, text: str, **metadata: Any) -> TextNode:
    n = TextNode(id_=node_id, text=text, metadata=dict(metadata))
    n.embedding = hash_embed(text)
    return n


def q(text: str, k: int = 10, **kw: Any) -> VectorStoreQuery:
    return VectorStoreQuery(query_embedding=hash_embed(text), similarity_top_k=k, **kw)


ALICE = {"a1": "invoice payment terms net thirty", "a2": "garden watering schedule morning"}
BOB = {"b1": "invoice dispute escalated legal", "b2": "quarterly roadmap planning notes"}


async def two_tenants(b: Backend) -> tuple[RagnorVectorStore, RagnorVectorStore]:
    alice = RagnorVectorStore(b.store, isolation_scope="alice")
    bob = RagnorVectorStore(b.store, isolation_scope="bob")
    await alice.async_add([node(k, v, kind="note") for k, v in ALICE.items()])
    await bob.async_add([node(k, v, kind="memo") for k, v in BOB.items()])
    return alice, bob


def ids(result: Any) -> set[str]:
    return {n.node_id for n in result.nodes}


# --------------------------------------------------- real-backend behavior


async def test_roundtrip_and_modes(backend: Backend) -> None:
    alice, _ = await two_tenants(backend)
    dense = await alice.aquery(q("invoice payment"))
    assert dense.ids[0] == "a1"
    assert dense.nodes[0].get_content() == ALICE["a1"]
    assert dense.nodes[0].metadata == {"kind": "note"}
    hybrid = await alice.aquery(
        q("watering garden", 2, mode=VectorStoreQueryMode.HYBRID, query_str="watering garden")
    )
    assert hybrid.ids[0] == "a2"
    if (
        backend.name != "qdrant"
    ):  # qdrant keyword stage is a token-overlap scan: still covered below
        text = await alice.aquery(
            VectorStoreQuery(
                query_str="payment terms", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
            )
        )
        assert text.ids[0] == "a1"


async def test_get_and_delete_variants(backend: Backend) -> None:
    alice, _ = await two_tenants(backend)
    assert {n.node_id for n in await alice.aget_nodes(["a1", "a2"])} == {"a1", "a2"}
    assert {n.node_id for n in await alice.aget_nodes()} == {"a1", "a2"}
    only_note = MetadataFilters(filters=[MetadataFilter(key="kind", value="note")])
    assert {n.node_id for n in await alice.aget_nodes(filters=only_note)} == {"a1", "a2"}
    in_filter = MetadataFilters(
        filters=[MetadataFilter(key="kind", value=["note", "zzz"], operator=FilterOperator.IN)]
    )
    if backend.name == "memory":
        assert len(await alice.aget_nodes(filters=in_filter)) == 2
    else:  # qdrant declares no IN support; pgvector declares it but its SQL does not bind lists
        with pytest.raises(Exception):  # noqa: B017, PT011 - fail closed, backend-specific type
            await alice.aget_nodes(filters=in_filter)
    parent = Document(text="p", doc_id="doc-9")
    child = node("c1", "child chunk about turtles")
    child.relationships = {NodeRelationship.SOURCE: parent.as_related_node_info()}
    await alice.async_add([child])
    await alice.adelete("doc-9")
    assert "c1" not in {n.node_id for n in await alice.aget_nodes()}
    await alice.adelete_nodes(["a1"])
    assert {n.node_id for n in await alice.aget_nodes()} == {"a2"}


async def test_upsert_is_idempotent(backend: Backend) -> None:
    alice, _ = await two_tenants(backend)
    await alice.async_add([node("a1", ALICE["a1"], kind="note")])
    assert await backend.tenant_rows("alice") == 2
    await alice.async_add([node("a1", "replaced text entirely", kind="note")])
    assert await backend.tenant_rows("alice") == 2
    assert (await alice.aget_nodes(["a1"]))[0].get_content() == "replaced text entirely"


# ------------------------------------------------------ adversarial tenancy


async def test_cross_scope_read_every_query_path(backend: Backend) -> None:
    alice, bob = await two_tenants(backend)
    for query in (
        q("invoice dispute escalated legal roadmap"),
        q("invoice", 10, mode=VectorStoreQueryMode.HYBRID, query_str="invoice dispute"),
    ):
        assert ids(await alice.aquery(query)) <= {"a1", "a2"}
        assert ids(await bob.aquery(query)) <= {"b1", "b2"}
    assert await alice.aget_nodes(["b1", "b2"]) == []
    assert ids_of(await alice.aget_nodes()) == {"a1", "a2"}
    assert ids_of(await bob.aget_nodes()) == {"b1", "b2"}


def ids_of(nodes: list[Any]) -> set[str]:
    return {n.node_id for n in nodes}


async def test_cross_scope_write_cannot_overwrite_same_node_id(backend: Backend) -> None:
    alice, bob = await two_tenants(backend)
    await bob.async_add([node("a1", "bob hijacks alice id", kind="memo")])
    assert (await alice.aget_nodes(["a1"]))[0].get_content() == ALICE["a1"]
    assert (await bob.aget_nodes(["a1"]))[0].get_content() == "bob hijacks alice id"
    assert await backend.tenant_rows("alice") == 2
    assert await backend.tenant_rows("bob") == 3


async def test_cross_scope_delete_and_clear(backend: Backend) -> None:
    alice, bob = await two_tenants(backend)
    await alice.adelete_nodes(["b1", "b2"])
    await alice.adelete("anything")
    await alice.adelete_nodes(
        filters=MetadataFilters(filters=[MetadataFilter(key="kind", value="memo")])
    )
    assert await backend.tenant_rows("bob") == 2
    await alice.aclear()
    assert await backend.tenant_rows("alice") == 0
    assert await backend.tenant_rows("bob") == 2
    await bob.aclear()
    assert await backend.tenant_rows("bob") == 0


async def test_filter_injection_and_scope_field_spoofing(backend: Backend) -> None:
    alice, bob = await two_tenants(backend)
    for key in ("tenant_id", "TENANT_ID", "Tenant_Id"):
        with pytest.raises(ValueError, match="isolation scope"):
            await alice.aquery(
                q("x", filters=MetadataFilters(filters=[MetadataFilter(key=key, value="bob")]))
            )
        with pytest.raises(ValueError, match="isolation scope"):
            await alice.aget_nodes(
                filters=MetadataFilters(filters=[MetadataFilter(key=key, value="bob")])
            )
        with pytest.raises(ValueError, match="isolation scope"):
            await alice.adelete_nodes(
                filters=MetadataFilters(filters=[MetadataFilter(key=key, value="bob")])
            )
    for evil_key in ("kind; DROP TABLE chunks", "kind OR 1=1", "kind--", 'kind"'):
        with pytest.raises(ValueError):
            await alice.aquery(
                q("x", filters=MetadataFilters(filters=[MetadataFilter(key=evil_key, value="a")]))
            )
    for evil_value in ("note' OR '1'='1", "x') OR tenant_id = 'bob", "%", "*"):
        res = await alice.aquery(
            q(
                "invoice",
                filters=MetadataFilters(filters=[MetadataFilter(key="kind", value=evil_value)]),
            )
        )
        assert res.nodes == []
    wide_in = MetadataFilters(
        filters=[MetadataFilter(key="kind", value=["note", "memo"], operator=FilterOperator.IN)]
    )
    if backend.name == "memory":
        assert ids(await alice.aquery(q("invoice", filters=wide_in))) <= {"a1", "a2"}
    else:
        with pytest.raises(Exception):  # noqa: B017, PT011 - fail closed
            await alice.aquery(q("invoice", filters=wide_in))
    for spoof in ("tenant_id", "TENANT_ID"):
        with pytest.raises(ValueError, match="different scope"):
            await alice.async_add([node("s", "spoof", **{spoof: "bob"})])
    assert await backend.tenant_rows("alice") == 2
    assert await backend.tenant_rows("bob") == 2
    # same-scope value is accepted and not duplicated as a column
    await alice.async_add([node("ok", "same scope claim", tenant_id="alice", kind="note")])
    assert "ok" in ids_of(await alice.aget_nodes())


async def test_async_paths_interleaved_across_tenants(backend: Backend) -> None:
    alice = RagnorVectorStore(backend.store, isolation_scope="alice")
    bob = RagnorVectorStore(backend.store, isolation_scope="bob")

    async def writer(store: RagnorVectorStore, prefix: str, n: int) -> None:
        for i in range(n):
            await store.async_add([node(f"{prefix}{i}", f"{prefix} document number {i}", kind="k")])

    await asyncio.gather(writer(alice, "al", 8), writer(bob, "bo", 8))
    assert await backend.tenant_rows("alice") == 8
    assert await backend.tenant_rows("bob") == 8

    async def reader(store: RagnorVectorStore, prefix: str) -> set[str]:
        out: set[str] = set()
        for _ in range(5):
            out |= ids(await store.aquery(q(f"{prefix} document number", 20)))
        return out

    a_ids, b_ids = await asyncio.gather(reader(alice, "al"), reader(bob, "bo"))
    assert a_ids and all(i.startswith("al") for i in a_ids)
    assert b_ids and all(i.startswith("bo") for i in b_ids)
    await asyncio.gather(alice.aclear(), bob.adelete_nodes(["bo0"]))
    assert await backend.tenant_rows("alice") == 0
    assert await backend.tenant_rows("bob") == 7


async def test_constructor_guards(backend: Backend) -> None:
    for bad in ("", None):
        with pytest.raises(ValueError):
            RagnorVectorStore(backend.store, isolation_scope=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RagnorVectorStore(backend.store, isolation_scope="a", scope_field="tenant; drop")
    with pytest.raises(ValueError):
        RagnorRetriever(DenseRetrieval(backend.store, Embedder()), isolation_scope="")


async def test_retriever_scoped_and_filters_cannot_override(backend: Backend) -> None:
    await two_tenants(backend)
    for scope, corpus in (("alice", ALICE), ("bob", BOB)):
        dense = RagnorRetriever(
            DenseRetrieval(backend.store, Embedder()),
            isolation_scope=scope,
            threshold=2.0,
            similarity_top_k=10,
        )
        texts = {h.node.get_content() for h in await dense.aretrieve("invoice roadmap garden")}
        assert texts and texts <= set(corpus.values())
        hybrid = RagnorRetriever(
            HybridRetrieval(backend.store, backend.store, Embedder()),
            isolation_scope=scope,
            threshold=2.0,
            similarity_top_k=10,
        )
        texts = {h.node.get_content() for h in await hybrid.aretrieve(QueryBundle("invoice"))}
        assert texts <= set(corpus.values())
    for key in ("tenant_id",):
        with pytest.raises(ValueError, match="scope"):
            RagnorRetriever(
                DenseRetrieval(backend.store, Embedder()),
                isolation_scope="alice",
                filters={key: "bob"},
            )


# ------------------------------------------------- IngestionPipeline.run


class Versions:
    """Version store with a real delete, scoped per tenant."""

    def __init__(self, store: Any, tenant: str) -> None:
        self.store = store
        self.tenant = tenant
        self.active: VersionRecord | None = None

    async def find_active(self, session: Any) -> VersionRecord | None:
        return self.active

    async def close_active(self, session: Any) -> None:
        self.active = None

    async def create_version(self, session: Any, snapshot: Any) -> VersionRecord:
        self.active = VersionRecord(uuid.uuid4().hex[:8], snapshot, source_id=self.tenant)
        return self.active

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        await self.store.delete(
            session=session, filters={"version_id": version_id, "tenant_id": self.tenant}
        )


def pipeline_for(b: Backend, tenant: str, size: int = 24) -> tuple[IngestionPipeline, Versions]:
    chunker = LlamaIndexChunker(
        SentenceSplitter(chunk_size=size, chunk_overlap=0), chunk_metadata={"tenant_id": tenant}
    )
    pipe = IngestionPipeline(
        store=b.store,
        embedder=Embedder(),
        chunker=chunker,
        session_factory=b.sessions,
        settings=Settings(),  # type: ignore[arg-type]
    )
    return pipe, Versions(b.store, tenant)


TEXT_A = "The invoice is due in thirty days. " * 3 + "The garden needs water every morning."
TEXT_A2 = "Completely new content about turtles and ponds. " * 3


async def test_ingestion_pipeline_run_with_llamaindex_source_and_chunker(backend: Backend) -> None:
    pipe_a, ver_a = pipeline_for(backend, "alice")
    pipe_b, ver_b = pipeline_for(backend, "bob")
    src_a = LlamaIndexDocumentSource([Document(text=TEXT_A, doc_id="d1")])
    src_b = LlamaIndexDocumentSource(
        [Document(text=TEXT_A, doc_id="d1")]
    )  # same text, other tenant
    n_a = await pipe_a.run(src_a, ver_a)
    n_b = await pipe_b.run(src_b, ver_b)
    assert n_a > 1 and n_b == n_a
    assert await backend.tenant_rows("alice") == n_a
    assert await backend.tenant_rows("bob") == n_b  # no id collision between tenants
    # unchanged source: skipped, no new rows (the returned count is store-wide, ragnor behavior)
    await pipe_a.run(src_a, ver_a)
    assert await backend.tenant_rows("alice") == n_a
    assert await backend.tenant_rows("bob") == n_b
    # scoped retriever sees only its tenant's pipeline-written rows
    for scope in ("alice", "bob"):
        retriever = RagnorRetriever(
            DenseRetrieval(backend.store, Embedder()),
            isolation_scope=scope,
            threshold=2.0,
            similarity_top_k=50,
        )
        hits = await retriever.aretrieve("invoice thirty days")
        assert len(hits) == n_a
        assert "invoice" in hits[0].node.get_content().lower()
    # changed source: old version retired, new chunks replace old for that tenant only
    n_new = await pipe_a.run(LlamaIndexDocumentSource([Document(text=TEXT_A2, doc_id="d1")]), ver_a)
    assert await backend.tenant_rows("alice") == n_new
    assert await backend.tenant_rows("bob") == n_b
    # a changed splitter setting changes the snapshot and re-chunks
    pipe_small, _ = pipeline_for(backend, "bob", size=12)
    again = await pipe_small.run(src_b, ver_b)
    assert again >= n_b
    assert await backend.tenant_rows("alice") == n_new
