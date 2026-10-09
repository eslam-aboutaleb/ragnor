"""Tests for ragit.retrieval.

Covers the :class:`HybridRetriever` orchestrator:
the distance-threshold fallback, filter passthrough,
the ``SearchResult`` conversion, and -- the key
plan-04 assertion -- retrieval against the
:class:`ragit.stores.InMemoryVectorStore`, which
proves the retriever has no pgvector dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any
from unittest.mock import patch

import pytest

from ragit.retrieval import (
    HybridRetriever,
    RetrieverSpec,
    as_dicts,
)
from ragit.stores.memory import InMemoryVectorStore
from ragit.types import SearchResult


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self) -> None:
        self.embedding_model = "text-embedding-3-small"
        self.rag_distance_threshold = 1.3
        self.vector_store_provider = "memory"
        self.openai_api_key = "test-key"
        self.openai_api_base = ""
        self.embedding_drain_interval_seconds = 5.0
        self.embedding_drain_batch_size = 10
        self.job_stale_after_seconds = 300.0
        self.worker_id = "test-worker"
        self.ingest_on_startup = True
        self.database_url = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"


class _HashingEmbedder:
    """Deterministic embedder: tokens hash into vector dimensions.

    Shared tokens land in shared dimensions, so texts that
    share vocabulary get close L2 distance -- bag-of-words
    similarity without any network calls.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _CapturingStore:
    """Store that records the arguments hybrid_search received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "embedding": embedding,
                "n_results": n_results,
                "threshold": threshold,
                "text_field": text_field,
                "metadata_fields": metadata_fields,
                "filters": filters,
            }
        )
        return [
            {
                "id": "1",
                "document": "captured document",
                "metadata": {"section": "Captured"},
                "distance": 0.5,
                "keyword_score": None,
                "_rrf_score": 0.8,
            }
        ]


def _spec(**kwargs: Any) -> RetrieverSpec:
    return RetrieverSpec(table_name="documents", **kwargs)


@pytest.mark.asyncio
async def test_retrieve_returns_search_results() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        results = await retriever.retrieve("water damage")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].document == "captured document"
    assert results[0].distance == 0.5
    assert results[0].rrf_score == 0.8


@pytest.mark.asyncio
async def test_distance_threshold_none_falls_back_to_settings() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query")

    assert store.calls[0]["threshold"] == 1.3


@pytest.mark.asyncio
async def test_explicit_distance_threshold_is_passed_through() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query", distance_threshold=0.7)

    assert store.calls[0]["threshold"] == 0.7


@pytest.mark.asyncio
async def test_filters_and_n_results_are_passed_through() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query", n_results=3, owner_id="owner-1", status="open")

    assert store.calls[0]["n_results"] == 3
    assert store.calls[0]["filters"] == {"owner_id": "owner-1", "status": "open"}


@pytest.mark.asyncio
async def test_retriever_retrieves_from_in_memory_store() -> None:
    """The retriever works against the in-memory store -- no pgvector.

    Seeds three documents and retrieves by keyword (token overlap)
    and by vector similarity (shared-token L2 distance), proving
    the orchestrator carries no pgvector dependency.
    """
    embedder = _HashingEmbedder(dim=8)
    store = InMemoryVectorStore(
        table_name="documents",
        id_field="id",
        embedding_dim=8,
        embedding_fn=embedder,
    )
    await store.upsert(
        [
            {
                "id": "1",
                "text": "water damage coverage",
                "metadata": {"section": "Water"},
                "embedding": embedder.embed("water damage coverage"),
            },
            {
                "id": "2",
                "text": "fire damage coverage",
                "metadata": {"section": "Fire"},
                "embedding": embedder.embed("fire damage coverage"),
            },
            {
                "id": "3",
                "text": "the deductible amount",
                "metadata": {"section": "Deductible"},
                "embedding": embedder.embed("the deductible amount"),
            },
        ]
    )

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(
            spec=_spec(metadata_fields=["section"]),
            settings=_FakeSettings(),
        )

        # Keyword + vector: "water damage" shares both tokens with
        # document 1 and only "damage" with document 2.
        results = await retriever.retrieve("water damage", n_results=5)
        assert results, "expected results for a seeded query"
        assert all(isinstance(result, SearchResult) for result in results)
        assert results[0].document == "water damage coverage"
        assert results[0].metadata == {"section": "Water"}

        # Keyword-only: "water" appears in document 1 alone.
        keyword_results = await retriever.retrieve("water", n_results=5)
        assert keyword_results[0].document == "water damage coverage"

        # Vector similarity: the query embedding of "water damage
        # coverage" is closest to document 1's embedding.
        vector_results = await retriever.retrieve("water damage coverage", n_results=5)
        assert vector_results[0].document == "water damage coverage"

        # Dict consumers keep working through as_dicts().
        as_dict = as_dicts(results)
        assert as_dict[0]["id"] == "1"
        assert as_dict[0]["document"] == "water damage coverage"
        assert as_dict[0]["metadata"] == {"section": "Water"}
        assert as_dict[0]["distance"] is not None
        assert as_dict[0]["_rrf_score"] > 0.0


@pytest.mark.asyncio
async def test_retrieve_propagates_store_errors() -> None:
    store = _CapturingStore()

    async def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("search failed")

    store.hybrid_search = _fail  # type: ignore[method-assign]
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragit.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragit.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        with pytest.raises(RuntimeError, match="search failed"):
            await retriever.retrieve("query")


def test_as_dicts_maps_every_field() -> None:
    results = [
        SearchResult(
            id="1",
            document="doc",
            metadata={"section": "S"},
            distance=0.5,
            keyword_score=0.25,
            rrf_score=0.8,
        ),
        SearchResult(id="2", document="doc2"),
    ]
    dicts = as_dicts(results)
    assert dicts[0] == {
        "id": "1",
        "document": "doc",
        "metadata": {"section": "S"},
        "distance": 0.5,
        "keyword_score": 0.25,
        "_rrf_score": 0.8,
    }
    assert dicts[1]["distance"] is None
    assert dicts[1]["keyword_score"] is None
    assert dicts[1]["_rrf_score"] == 0.0


def test_retriever_spec_validates_identifiers() -> None:
    with pytest.raises(ValueError, match="Unsafe table_name"):
        RetrieverSpec(table_name="bad-name")
    with pytest.raises(ValueError, match="Unsafe metadata_field"):
        RetrieverSpec(table_name="documents", metadata_fields=["bad-field"])
    spec = _spec(id_field="id", text_field="text", metadata_fields=["section"])
    assert spec.table_name == "documents"
