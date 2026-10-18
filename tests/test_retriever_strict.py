"""Strict propagation tests for :class:`HybridRetriever.retrieve`.

``retrieve(strict=...)`` must reach the store on the legacy
path and the strategy on the injected-strategy path, so a real
storage outage is distinguishable from a legitimate empty
result under ``strict=True`` and failure behavior is uniform
across both paths.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ragnor.retrieval import HybridRetriever, RetrieverSpec
from ragnor.types import RetrievalError


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self) -> None:
        self.embedding_model = "text-embedding-3-small"
        self.rag_distance_threshold = 1.3
        self.vector_store_provider = "memory"
        self.openai_api_key = "test-key"
        self.openai_api_base = ""
        self.database_url = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"


class _FakeEmbedder:
    """Trivial embedder: every text maps to a zero vector."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class _CapturingStore:
    """Store that records every hybrid_search invocation."""

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
                "n_results": n_results,
                "threshold": threshold,
                "strict": strict,
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


class _FailingStore:
    """Store implementing the swallow/raise contract that always fails."""

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
        if strict:
            raise RetrievalError("index down")
        return []


def _spec() -> RetrieverSpec:
    return RetrieverSpec(table_name="documents")


def _retriever(store: Any) -> HybridRetriever:
    """Build a HybridRetriever whose embedder and store are injected."""
    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=_FakeEmbedder(),
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        return HybridRetriever(spec=_spec(), settings=_FakeSettings())


@pytest.mark.asyncio
async def test_retrieve_passes_strict_true_to_store() -> None:
    """The legacy path forwards strict=True to hybrid_search."""
    store = _CapturingStore()
    retriever = _retriever(store)

    await retriever.retrieve("query", strict=True)

    assert store.calls[0]["strict"] is True


@pytest.mark.asyncio
async def test_retrieve_passes_strict_false_to_store_by_default() -> None:
    """The legacy path defaults to strict=False (swallow failures)."""
    store = _CapturingStore()
    retriever = _retriever(store)

    await retriever.retrieve("query")

    assert store.calls[0]["strict"] is False


@pytest.mark.asyncio
async def test_failing_store_raises_under_strict() -> None:
    """A storage outage surfaces as RetrievalError under strict=True."""
    retriever = _retriever(_FailingStore())

    with pytest.raises(RetrievalError, match="index down"):
        await retriever.retrieve("query", strict=True)


@pytest.mark.asyncio
async def test_failing_store_returns_empty_without_strict() -> None:
    """A storage outage is swallowed into [] under strict=False."""
    retriever = _retriever(_FailingStore())

    assert await retriever.retrieve("query", strict=False) == []


@pytest.mark.asyncio
async def test_retrieve_passes_strict_to_injected_strategy() -> None:
    """The strategy path forwards strict the same way -- uniform behavior."""
    captured: dict[str, Any] = {}

    class _CapturingStrategy:
        async def retrieve(
            self,
            query: str,
            n_results: int = 5,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return [
                {
                    "id": "1",
                    "document": "doc",
                    "metadata": {},
                    "distance": None,
                    "keyword_score": None,
                }
            ]

    retriever = HybridRetriever(
        spec=_spec(),
        settings=_FakeSettings(),
        strategy=_CapturingStrategy(),
    )
    await retriever.retrieve("query", strict=True)

    assert captured["strict"] is True
