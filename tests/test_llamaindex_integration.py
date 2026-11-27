"""Real tests for the optional LlamaIndex adapters.

No mocks: llama-index-core is installed, the vector store is ragnor's real
in-memory store, chunking uses LlamaIndex's real ``SentenceSplitter``, and
embeddings are a deterministic hashing embedder (no network). The module is
skipped when ``llama-index-core`` is absent (the core package has no such
dependency).
"""

from __future__ import annotations

import asyncio
import hashlib
import math

import pytest

pytest.importorskip("llama_index.core")

from tests.llamaindex_env import harden_nltk_cache  # noqa: E402

harden_nltk_cache()

from llama_index.core import Settings, StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.core.base.embeddings.base import BaseEmbedding  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.readers.base import BaseReader  # noqa: E402
from llama_index.core.schema import (  # noqa: E402
    Document,
    NodeRelationship,
    QueryBundle,
    TextNode,
)
from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from ragnor.chunking import Chunker  # noqa: E402
from ragnor.ingestion.sources import DocumentSource  # noqa: E402
from ragnor.integrations.llamaindex import (  # noqa: E402
    LlamaIndexChunker,
    LlamaIndexDocumentSource,
    RagnorRetriever,
    RagnorVectorStore,
)
from ragnor.retrieval import DenseRetrieval, HybridRetrieval, ScopeViolationError  # noqa: E402
from ragnor.stores.memory import InMemoryVectorStore  # noqa: E402

DIM = 64


def hash_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for token in text.lower().split():
        digest = hashlib.sha256(token.strip(".,").encode()).digest()
        vec[digest[0] % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashEmbedding(BaseEmbedding):
    """Real (deterministic) LlamaIndex embedding model."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return hash_embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return hash_embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return hash_embed(text)


async def _embedder(texts: list[str]) -> list[list[float]]:
    return [hash_embed(t) for t in texts]


class _Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await _embedder(texts)

    async def embed_query(self, text: str) -> list[float]:
        return hash_embed(text)


def make_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding_dim=DIM, metric="cosine")


def node(node_id: str, text: str, **metadata: object) -> TextNode:
    n = TextNode(id_=node_id, text=text, metadata=dict(metadata))
    n.embedding = hash_embed(text)
    return n


CORPUS_A = {
    "a1": "invoice payment terms are net thirty days",
    "a2": "the garden needs watering every morning",
}
CORPUS_B = {
    "b1": "invoice dispute escalated to the legal team",
    "b2": "quarterly roadmap planning meeting notes",
}


async def seeded() -> tuple[InMemoryVectorStore, RagnorVectorStore, RagnorVectorStore]:
    store = make_store()
    alice = RagnorVectorStore(store, isolation_scope="alice")
    bob = RagnorVectorStore(store, isolation_scope="bob")
    await alice.async_add([node(k, v, kind="note") for k, v in CORPUS_A.items()])
    await bob.async_add([node(k, v, kind="memo") for k, v in CORPUS_B.items()])
    return store, alice, bob


# --------------------------------------------------------------- inbound


class StubReader(BaseReader):
    """A real BaseReader subclass returning fixed documents."""

    def load_data(self, **kwargs: object) -> list[Document]:
        return [
            Document(text="first body", doc_id="d1", metadata={"file_name": "one.txt"}),
            Document(text="second body", doc_id="d2", metadata={"lang": "en"}),
        ]


class BrokenReader(BaseReader):
    def load_data(self, **kwargs: object) -> list[Document]:
        raise PermissionError("denied")


async def test_source_converts_reader_documents() -> None:
    source = LlamaIndexDocumentSource(StubReader())
    assert isinstance(source, DocumentSource)
    docs = await source.load()
    assert [d.content for d in docs] == ["first body", "second body"]
    assert docs[0].source_name == "one.txt"
    assert docs[0].external_id == "d1"
    assert docs[1].source_name == "d2"
    assert docs[1].metadata == {"lang": "en"}


async def test_source_accepts_document_list_and_empty_is_authoritative() -> None:
    assert await LlamaIndexDocumentSource([]).load() == []
    docs = await LlamaIndexDocumentSource([Document(text="x", doc_id="z")]).load()
    assert docs[0].content == "x"


async def test_source_propagates_reader_failure() -> None:
    with pytest.raises(PermissionError):
        await LlamaIndexDocumentSource(BrokenReader()).load()


def test_chunker_satisfies_protocol_and_is_deterministic() -> None:
    chunker = LlamaIndexChunker(SentenceSplitter(chunk_size=64, chunk_overlap=0))
    assert isinstance(chunker, Chunker)
    text = " ".join(f"Sentence number {i} talks about topic {i % 3}." for i in range(40))
    first = chunker.chunk(text)
    again = chunker.chunk(text)
    assert len(first) > 1
    assert [c.id for c in first] == [c.id for c in again]
    assert len({c.id for c in first}) == len(first)
    assert all(c.text.strip() for c in first)
    assert "".join(c.text.replace(" ", "") for c in first).startswith("Sentencenumber0")


def test_chunker_config_tracks_splitter_settings() -> None:
    small = LlamaIndexChunker(SentenceSplitter(chunk_size=64, chunk_overlap=0))
    large = LlamaIndexChunker(SentenceSplitter(chunk_size=256, chunk_overlap=0))
    assert small.config["chunk_size"] == 64
    assert small.config != large.config
    assert small.version == "llamaindex:SentenceSplitter"


def test_chunker_rejects_non_splitter() -> None:
    with pytest.raises(TypeError):
        LlamaIndexChunker(object())  # type: ignore[arg-type]


async def test_source_chunker_into_store_end_to_end() -> None:
    docs = await LlamaIndexDocumentSource(
        [Document(text=" ".join(CORPUS_A.values()) + ". " + CORPUS_B["b2"], doc_id="d")]
    ).load()
    chunks = LlamaIndexChunker(SentenceSplitter(chunk_size=32, chunk_overlap=0)).chunk(
        docs[0].content
    )
    store = make_store()
    await store.upsert(
        [
            {"id": c.id, "text": c.text, "metadata": {}, "embedding": hash_embed(c.text)}
            for c in chunks
        ],
        extra_fields={"tenant_id": "alice"},
    )
    hits = await store.vector_search(
        hash_embed("garden watering"), 1, 2.0, metadata_fields=["tenant_id"]
    )
    assert "garden" in hits[0]["document"]


# -------------------------------------------------------------- outbound


async def test_vector_store_dense_query_roundtrips_nodes() -> None:
    _, alice, _ = await seeded()
    result = await alice.aquery(
        VectorStoreQuery(query_embedding=hash_embed("invoice payment"), similarity_top_k=2)
    )
    assert result.ids[0] == "a1"
    assert isinstance(result.nodes[0], TextNode)
    assert result.nodes[0].text == CORPUS_A["a1"]
    assert result.nodes[0].metadata == {"kind": "note"}
    assert result.similarities == sorted(result.similarities, reverse=True)


async def test_vector_store_tenant_isolation_on_every_path() -> None:
    store, alice, bob = await seeded()
    query = VectorStoreQuery(query_embedding=hash_embed("invoice"), similarity_top_k=10)
    assert {n.node_id for n in (await alice.aquery(query)).nodes} == {"a1", "a2"}
    assert {n.node_id for n in (await bob.aquery(query)).nodes} == {"b1", "b2"}
    # get by id cannot cross tenants
    assert await alice.aget_nodes(["b1"]) == []
    # delete by id cannot cross tenants
    await alice.adelete_nodes(["b1"])
    assert await store.count(tenant_id="bob") == 2
    # clear only clears own scope
    await alice.aclear()
    assert await store.count(tenant_id="alice") == 0
    assert await store.count(tenant_id="bob") == 2


async def test_vector_store_rejects_scope_spoofing() -> None:
    store = make_store()
    alice = RagnorVectorStore(store, isolation_scope="alice")
    with pytest.raises(ValueError, match="different scope"):
        await alice.async_add([node("x", "evil", tenant_id="bob")])
    with pytest.raises(ValueError, match="isolation scope"):
        await alice.aquery(
            VectorStoreQuery(
                query_embedding=hash_embed("x"),
                filters=MetadataFilters(filters=[MetadataFilter(key="tenant_id", value="bob")]),
            )
        )
    assert await store.count() == 0


async def test_vector_store_requires_scope_and_embedding() -> None:
    with pytest.raises(ValueError):
        RagnorVectorStore(make_store(), isolation_scope="")
    store = RagnorVectorStore(make_store(), isolation_scope="a")
    with pytest.raises(ValueError, match="embedding"):
        await store.async_add([TextNode(id_="n", text="t")])


async def test_vector_store_fails_closed_on_cross_scope_row() -> None:
    """If a backend ever returned another scope's row, the adapter raises."""
    inner = make_store()
    _, alice, _ = await seeded()
    alice_view = RagnorVectorStore(inner, isolation_scope="alice")
    await alice_view.async_add([node("a9", "stray note")])
    rows = await inner.get(ids=["alice:a9"], metadata_fields=alice_view._metadata_fields())
    rows[0]["metadata"]["tenant_id"] = "bob"
    with pytest.raises(ScopeViolationError):
        alice_view._node(rows[0])


async def test_vector_store_metadata_filters_and_unsupported_operators() -> None:
    _, alice, _ = await seeded()
    only_note = MetadataFilters(filters=[MetadataFilter(key="kind", value="note")])
    other = MetadataFilters(filters=[MetadataFilter(key="kind", value="memo")])
    q = VectorStoreQuery(query_embedding=hash_embed("invoice"), similarity_top_k=5)
    q.filters = only_note
    assert len((await alice.aquery(q)).nodes) == 2
    q.filters = other
    assert (await alice.aquery(q)).nodes == []
    q.filters = MetadataFilters(
        filters=[MetadataFilter(key="kind", value=["note", "x"], operator=FilterOperator.IN)]
    )
    assert len((await alice.aquery(q)).nodes) == 2
    q.filters = MetadataFilters(
        filters=[MetadataFilter(key="kind", value="n", operator=FilterOperator.GT)]
    )
    with pytest.raises(NotImplementedError):
        await alice.aquery(q)


async def test_vector_store_hybrid_and_text_modes() -> None:
    _, alice, _ = await seeded()
    hybrid = await alice.aquery(
        VectorStoreQuery(
            query_embedding=hash_embed("watering garden"),
            query_str="watering garden",
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=2,
        )
    )
    assert hybrid.ids[0] == "a2"
    text = await alice.aquery(
        VectorStoreQuery(
            query_str="payment terms", mode=VectorStoreQueryMode.TEXT_SEARCH, similarity_top_k=2
        )
    )
    assert text.ids[0] == "a1"
    with pytest.raises(NotImplementedError):
        await alice.aquery(
            VectorStoreQuery(query_embedding=hash_embed("x"), mode=VectorStoreQueryMode.MMR)
        )


async def test_vector_store_delete_by_ref_doc_and_upsert_idempotent() -> None:
    store = make_store()
    s = RagnorVectorStore(store, isolation_scope="alice")
    parent = Document(text="parent", doc_id="doc-1")
    child = node("c1", "child one text")
    child.relationships = {NodeRelationship.SOURCE: parent.as_related_node_info()}
    other = node("c2", "unrelated text")
    await s.async_add([child, other])
    await s.async_add([child])  # same id again: replaced, not duplicated
    assert await store.count() == 2
    await s.adelete("doc-1")
    assert [n.node_id for n in await s.aget_nodes()] == ["c2"]


async def test_vector_store_delete_nodes_needs_a_selector() -> None:
    _, alice, _ = await seeded()
    with pytest.raises(ValueError):
        await alice.adelete_nodes()


def test_vector_store_sync_api_works_outside_and_inside_a_loop() -> None:
    store = make_store()
    s = RagnorVectorStore(store, isolation_scope="alice")
    s.add([node("s1", "sync add path")])
    q = VectorStoreQuery(query_embedding=hash_embed("sync add"), similarity_top_k=1)
    assert s.query(q).ids == ["s1"]

    async def inside() -> list[str]:
        return s.query(q).ids  # sync API called from a running loop

    assert asyncio.run(inside()) == ["s1"]
    s.delete_nodes(["s1"])
    assert s.get_nodes() == []
    s.clear()


async def test_llamaindex_index_end_to_end_with_real_embedding_model() -> None:
    Settings.embed_model = HashEmbedding()
    Settings.llm = None
    store = make_store()
    vs = RagnorVectorStore(store, isolation_scope="alice")
    docs = [Document(text=t, doc_id=k) for k, t in CORPUS_A.items()]
    index = await asyncio.to_thread(
        VectorStoreIndex.from_documents,
        docs,
        storage_context=StorageContext.from_defaults(vector_store=vs),
    )
    retriever = index.as_retriever(similarity_top_k=1)
    hits = await retriever.aretrieve("garden watering morning")
    assert hits[0].node.get_content() == CORPUS_A["a2"]
    assert await store.count(tenant_id="alice") == 2


# ------------------------------------------------------------- retriever


async def _populated_store() -> InMemoryVectorStore:
    store = make_store()
    await RagnorVectorStore(store, isolation_scope="alice").async_add(
        [node(k, v, kind="note") for k, v in CORPUS_A.items()]
    )
    await RagnorVectorStore(store, isolation_scope="bob").async_add(
        [node(k, v, kind="memo") for k, v in CORPUS_B.items()]
    )
    return store


async def test_retriever_dense_and_hybrid_are_scoped() -> None:
    store = await _populated_store()
    embedder = _Embedder()
    dense = RagnorRetriever(
        DenseRetrieval(store, embedder),
        isolation_scope="alice",
        threshold=2.0,
        similarity_top_k=5,
        metadata_fields=["kind"],
    )
    hits = await dense.aretrieve("invoice payment")
    assert {h.node.node_id for h in hits} == {"alice:a1", "alice:a2"}
    assert hits[0].node.get_content() == CORPUS_A["a1"]
    assert hits[0].score >= hits[-1].score

    hybrid = RagnorRetriever(
        HybridRetrieval(store, store, embedder),
        isolation_scope="bob",
        threshold=2.0,
        similarity_top_k=5,
    )
    bob_hits = await hybrid.aretrieve(QueryBundle("invoice dispute"))
    assert {h.node.node_id for h in bob_hits} <= {"bob:b1", "bob:b2"}
    assert bob_hits[0].node.get_content() == CORPUS_B["b1"]


def test_retriever_sync_path_and_isolation() -> None:
    store = asyncio.run(_populated_store())
    retriever = RagnorRetriever(
        DenseRetrieval(store, _Embedder()), isolation_scope="bob", threshold=2.0
    )
    hits = retriever.retrieve("quarterly roadmap")
    assert hits[0].node.get_content() == CORPUS_B["b2"]
    assert all(h.node.node_id.startswith("bob:") for h in hits)


async def test_retriever_scope_cannot_be_overridden() -> None:
    store = await _populated_store()
    with pytest.raises(ValueError, match="scope"):
        RagnorRetriever(
            DenseRetrieval(store, _Embedder()),
            isolation_scope="alice",
            filters={"tenant_id": "bob"},
        )
    with pytest.raises(ValueError):
        RagnorRetriever(DenseRetrieval(store, _Embedder()), isolation_scope="")


def test_core_import_does_not_pull_in_llamaindex() -> None:
    import subprocess
    import sys

    code = (
        "import sys, ragnor, ragnor.retrieval, ragnor.stores, ragnor.chunking;"
        "sys.exit(1 if any(m.startswith('llama_index') for m in sys.modules) else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603
    assert result.returncode == 0
