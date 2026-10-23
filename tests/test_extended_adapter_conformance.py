"""Behavioral conformance tests for host-supplied outer adapters.

These fakes/proxies live OUTSIDE src/ragnor to exercise ports as an SDK
consumer. Memory-backed tests are not claims about durable distributed hosts.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.evidence.types import ScopeAuthorizationError
from ragnor.ingestion.records import RawDocument
from ragnor.ingestion.sources import DocumentSource
from ragnor.jobs.outbox import EmbeddingJobStore, InMemoryEmbeddingJobStore
from ragnor.retrieval.capabilities import (
    BackendFeature,
    CapabilityUnsupportedError,
    FilterOperator,
    StoreCapabilities,
)
from ragnor.retrieval.strategies import HybridRetrieval, Reranker
from ragnor.retrieval.types import RetrievalRequest
from ragnor.stores.base import DocumentStore, LexicalIndex, VectorIndex
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.sync.manifest import InMemoryManifestStore, ManifestRecord, ManifestStore
from ragnor.sync.state import InMemorySyncStateStore, SyncStateStore
from ragnor.tools.bridge import KnowledgeBridge
from ragnor.tools.security import ScopeAuthorizer
from tests.test_external_adapter_conformance import HostChunker, HostEmbedder


class HostExtractor:
    """Independent, vendor-neutral extractor implementation."""

    id = "host-text"
    version = "1"
    options = {"encoding": "utf-8"}

    def extract(self, data: bytes, *, name: str = "") -> str:
        return data.decode("utf-8").replace("\\r\\n", "\\n")


class HostSource:
    def __init__(self, extractor: HostExtractor) -> None:
        self.extractor = extractor

    async def load(self) -> list[RawDocument]:
        content = self.extractor.extract(b"refund policy", name="policy.txt")
        return [
            RawDocument(
                source_name="policy.txt",
                external_id="policy-1",
                content=content,
            )
        ]


class HostStorageAdapter:
    """Out-of-tree document store forwarding to a host-selected backend."""

    def __init__(self) -> None:
        self.backend = InMemoryVectorStore(embedding_dim=2)

    async def upsert(self, documents: list[dict[str, Any]], **kwargs: Any) -> None:
        await self.backend.upsert(documents, **kwargs)

    async def count(self, **kwargs: Any) -> int:
        return await self.backend.count(**kwargs)

    async def get(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.backend.get(**kwargs)

    async def delete(self, **kwargs: Any) -> int:
        return await self.backend.delete(**kwargs)


class HostIndex:
    """Independently injected vector and lexical index with explicit capabilities."""

    def __init__(self, *, allow_scope: bool) -> None:
        features = {BackendFeature.EQUALITY_FILTER}
        if allow_scope:
            features.add(BackendFeature.ISOLATION_SCOPE)
        self.capabilities = StoreCapabilities(
            filter_operators=frozenset({FilterOperator.EQ}),
            features=frozenset(features),
        )
        self.filters_seen: list[dict[str, Any]] = []

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.filters_seen.append(filters)
        return [
            {
                "id": "a",
                "document": "refund covered",
                "distance": 0.1,
                "keyword_score": None,
                "metadata": {"tenant_id": "tenant-a"},
            }
        ]

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.filters_seen.append(filters)
        return [
            {
                "id": "a",
                "document": "refund covered",
                "distance": None,
                "keyword_score": 1.0,
                "metadata": {"tenant_id": "tenant-a"},
            }
        ]


class HostReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return sorted(results, key=lambda result: str(result["id"]), reverse=True)


class HostManifestAdapter:
    """Host-defined port wrapper: exercises injection, not SQL atomicity."""

    def __init__(self) -> None:
        self.backend = InMemoryManifestStore()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


class HostSyncStateAdapter:
    def __init__(self) -> None:
        self.backend = InMemorySyncStateStore()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


class HostJobStoreAdapter:
    def __init__(self) -> None:
        self.backend = InMemoryEmbeddingJobStore()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


class HostRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(
        self, query: str, n_results: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls += 1
        return []


@pytest.mark.asyncio
async def test_external_source_and_extractor_can_be_injected() -> None:
    source = HostSource(HostExtractor())
    assert isinstance(source, DocumentSource)
    rag = await RagnorBuilder.local(
        HostEmbedder(), embedding_dim=2
    ).with_chunker(HostChunker()).build()
    count = await rag.pipeline.run(source, rag._version_store)
    assert count == 1
    assert await rag.count() == 1


@pytest.mark.asyncio
async def test_external_document_store_observes_filter_and_delete_contracts() -> None:
    store = HostStorageAdapter()
    assert isinstance(store, DocumentStore)
    await store.upsert([
        {"id": "a", "text": "alpha", "embedding": [1.0, 0.0], "tenant": "a"},
        {"id": "b", "text": "beta", "embedding": [0.0, 1.0], "tenant": "b"},
    ])
    assert await store.count(tenant="a") == 1
    rows = await store.get(filters={"tenant": "a"})
    assert [row["id"] for row in rows] == ["a"]
    assert await store.delete(ids=["a"], filters={"tenant": "a"}) == 1
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_external_hybrid_indexes_and_reranker_are_composable() -> None:
    index = HostIndex(allow_scope=True)
    reranker = HostReranker()
    assert isinstance(index, VectorIndex)
    assert isinstance(index, LexicalIndex)
    assert isinstance(reranker, Reranker)
    strategy = HybridRetrieval(
        vector_index=index,
        lexical_index=index,
        embedder=HostEmbedder(),
        reranker=reranker,
    )
    request = RetrievalRequest(
        query="refund",
        threshold=1.0,
        filters={"tenant_id": "tenant-a"},
        mandatory_capabilities={"isolation_scope"},
        strict=True,
    )
    results = await strategy.retrieve(request)
    assert [result["id"] for result in results] == ["a"]
    assert reranker.calls == 1
    assert index.filters_seen == [{"tenant_id": "tenant-a"}] * 2


@pytest.mark.asyncio
async def test_external_index_without_isolation_capability_fails_closed() -> None:
    index = HostIndex(allow_scope=False)
    strategy = HybridRetrieval(
        vector_index=index,
        lexical_index=index,
        embedder=HostEmbedder(),
    )
    request = RetrievalRequest(
        query="refund",
        threshold=1.0,
        mandatory_capabilities={"isolation_scope"},
        strict=True,
    )
    with pytest.raises(CapabilityUnsupportedError):
        await strategy.retrieve(request)
    assert index.filters_seen == []


@pytest.mark.asyncio
async def test_external_manifest_and_sync_ports_preserve_cas_and_fencing() -> None:
    manifest = HostManifestAdapter()
    sync = HostSyncStateAdapter()
    assert isinstance(manifest, ManifestStore)
    assert isinstance(sync, SyncStateStore)
    record = ManifestRecord(scope="tenant-a", document_id="doc")
    assert await manifest.compare_and_swap(record, expected_fence=0)
    stale = ManifestRecord(scope="tenant-a", document_id="doc")
    assert not await manifest.compare_and_swap(stale, expected_fence=0)
    active = await manifest.get("tenant-a", "doc")
    assert active is not None and active.fence_token == 1
    token = await sync.acquire_lease("tenant-a", "worker-1")
    assert token is not None and await sync.validate_token(token)
    assert await sync.acquire_lease("tenant-a", "worker-2") is None
    assert await sync.save_checkpoint("tenant-a", {"revision": 1}, token)
    assert await sync.release_lease(token)
    assert not await sync.validate_token(token)
    assert not await sync.save_checkpoint("tenant-a", {"revision": 2}, token)


def test_external_job_port_is_structurally_replaceable() -> None:
    assert isinstance(HostJobStoreAdapter(), EmbeddingJobStore)


@pytest.mark.asyncio
async def test_host_authorizer_denies_before_querying_retriever() -> None:
    retriever = HostRetriever()
    authorizer = ScopeAuthorizer(
        scope_mapping=lambda client, collection, scope: (
            client == "client-a" and scope == "tenant-a"
        ),
        default_allow=False,
    )
    bridge = KnowledgeBridge(retriever=retriever, authorizer=authorizer)
    with pytest.raises(ScopeAuthorizationError):
        await bridge.search_knowledge(
            query="refund", client_id="client-b", scope="tenant-a"
        )
    assert retriever.calls == 0
