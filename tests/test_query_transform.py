"""Tests for query transformation hooks and bounds."""

from __future__ import annotations

import asyncio

import pytest

from ragit.retrieval.query_transform import (
    BoundedQueryTransformer,
    QueryTransformer,
)
from ragit.retrieval.strategies import HybridRetrieval
from ragit.retrieval.types import RetrievalRequest


class EchoTransformer:
    async def transform(self, query: str) -> list[str]:
        return [f"{query} synonym 1", f"{query} synonym 2"]


class RunawayTransformer:
    async def transform(self, query: str) -> list[str]:
        # Generates 100 queries
        return [f"{query} variant {i}" for i in range(100)]


class HangingTransformer:
    async def transform(self, query: str) -> list[str]:
        await asyncio.sleep(10.0)
        return ["never returned"]


class FailingTransformer:
    async def transform(self, query: str) -> list[str]:
        raise RuntimeError("LLM service unavailable")


def test_query_transformer_protocol_conformance() -> None:
    t = EchoTransformer()
    assert isinstance(t, QueryTransformer)


@pytest.mark.asyncio
async def test_bounded_transformer_enforces_max_queries() -> None:
    bounded = BoundedQueryTransformer(
        RunawayTransformer(),
        max_queries=4,
        include_original=True,
    )
    results = await bounded.transform("database")
    # Original + 3 variants = 4
    assert len(results) == 4
    assert results[0] == "database"


@pytest.mark.asyncio
async def test_bounded_transformer_handles_timeout_gracefully() -> None:
    bounded = BoundedQueryTransformer(
        HangingTransformer(),
        timeout_seconds=0.05,
        include_original=True,
    )
    # Never raises; falls back to original query
    results = await bounded.transform("latency test")
    assert results == ["latency test"]


@pytest.mark.asyncio
async def test_bounded_transformer_handles_error_gracefully() -> None:
    bounded = BoundedQueryTransformer(
        FailingTransformer(),
        include_original=True,
    )
    # Never raises; falls back to original query
    results = await bounded.transform("fault test")
    assert results == ["fault test"]


@pytest.mark.asyncio
async def test_bounded_transformer_deduplication() -> None:
    class DuplicateTransformer:
        async def transform(self, query: str) -> list[str]:
            return [query, query.upper(), f"{query} new", f"{query} new"]

    bounded = BoundedQueryTransformer(DuplicateTransformer(), max_queries=10)
    results = await bounded.transform("cat")
    assert results == ["cat", "cat new"]


@pytest.mark.asyncio
async def test_query_transformer_integration_in_hybrid_retrieval() -> None:
    queries_seen: list[str] = []

    class CapturingIndex:
        async def vector_search(self, embedding, n_results, threshold, **filters):
            return [{"id": "v1", "document": "hit", "distance": 0.1}]

        async def keyword_search(self, query, n_results, **filters):
            queries_seen.append(query)
            return [{"id": "k1", "document": "hit", "keyword_score": 1.0}]

    async def fake_embedder(texts):
        return [[0.1] * 5 for _ in texts]

    store = CapturingIndex()
    strategy = HybridRetrieval(
        vector_index=store,
        lexical_index=store,
        embedder=fake_embedder,
        query_transformer=EchoTransformer(),
    )

    req = RetrievalRequest(query="distributed systems", threshold=1.0)
    await strategy.retrieve(req)

    # All generated queries were searched
    assert "distributed systems" in queries_seen
    assert "distributed systems synonym 1" in queries_seen
    assert "distributed systems synonym 2" in queries_seen
