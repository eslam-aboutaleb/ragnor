"""Tests for candidate_k vs final_k retrieval and scope validation."""

from __future__ import annotations

import pytest

from ragnor.retrieval.strategies import HybridRetrieval
from ragnor.retrieval.types import RetrievalRequest, ScopeViolationError


class SpyIndex:
    def __init__(self, doc_prefix: str, total_docs: int = 20) -> None:
        self.doc_prefix = doc_prefix
        self.total_docs = total_docs
        self.last_n_results: int | None = None
        self.last_filters: dict | None = None

    async def vector_search(self, embedding, n_results, threshold, **filters):
        self.last_n_results = n_results
        self.last_filters = filters
        return [
            {
                "id": f"{self.doc_prefix}_{i}",
                "document": f"Text {i}",
                "distance": 0.05 * i,
                "metadata": {"tenant_id": filters.get("tenant_id", "t1")},
            }
            for i in range(min(n_results, self.total_docs))
        ]

    async def keyword_search(self, query, n_results, **filters):
        self.last_n_results = n_results
        self.last_filters = filters
        return [
            {
                "id": f"{self.doc_prefix}_{i}",
                "document": f"Text {i}",
                "keyword_score": 10.0 - i,
                "metadata": {"tenant_id": filters.get("tenant_id", "t1")},
            }
            for i in range(min(n_results, self.total_docs))
        ]


class MockReranker:
    def __init__(self) -> None:
        self.called_with_count = 0

    async def rerank(self, query: str, results: list[dict]):
        self.called_with_count = len(results)
        # Reverse the candidates so last becomes first
        reversed_results = list(reversed(results))
        for idx, item in enumerate(reversed_results):
            item["_reranker_score"] = float(100 - idx)
        return reversed_results


@pytest.mark.asyncio
async def test_candidate_k_separately_exposed_and_fused() -> None:
    v_idx = SpyIndex("v", total_docs=30)
    l_idx = SpyIndex("l", total_docs=30)

    async def fake_embedder(texts):
        return [[0.1] * 10 for _ in texts]

    reranker = MockReranker()

    strategy = HybridRetrieval(
        vector_index=v_idx,
        lexical_index=l_idx,
        embedder=fake_embedder,
        reranker=reranker,
    )

    req = RetrievalRequest(
        query="what is candidate_k",
        final_k=3,
        candidate_k=15,
        threshold=1.5,
    )

    results = await strategy.retrieve(req)

    # Indexes were asked for candidate_k (15), not final_k (3)
    assert v_idx.last_n_results == 15
    assert l_idx.last_n_results == 15

    # Reranker received candidate_k candidates
    assert reranker.called_with_count == 15

    # Final result was sliced to final_k (3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_scope_validation_defense_in_depth() -> None:
    class LeakyIndex:
        async def vector_search(self, embedding, n_results, threshold, **filters):
            return [
                {"id": "doc1", "document": "allowed", "metadata": {"tenant_id": "tenant_a"}},
                {"id": "doc2", "document": "leaked", "metadata": {"tenant_id": "tenant_b"}},
            ]

        async def keyword_search(self, query, n_results, **filters):
            return []

    async def fake_embedder(texts):
        return [[0.1] * 5 for _ in texts]

    store = LeakyIndex()
    strategy = HybridRetrieval(vector_index=store, lexical_index=store, embedder=fake_embedder)

    # Non-strict mode: silently drops leaked document
    req_non_strict = RetrievalRequest(
        query="test",
        threshold=1.0,
        isolation_scope="tenant_a",
        strict=False,
    )
    res = await strategy.retrieve(req_non_strict)
    assert len(res) == 1
    assert res[0]["id"] == "doc1"

    # Strict mode: raises ScopeViolationError on leak
    req_strict = RetrievalRequest(
        query="test",
        threshold=1.0,
        isolation_scope="tenant_a",
        strict=True,
    )
    with pytest.raises(ScopeViolationError) as exc_info:
        await strategy.retrieve(req_strict)
    assert "Scope violation" in str(exc_info.value)
    assert "tenant_b" in str(exc_info.value)
