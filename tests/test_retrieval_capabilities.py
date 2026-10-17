"""Tests for backend capability declarations and fail-closed validation."""

from __future__ import annotations

import pytest

from ragit.retrieval.capabilities import (
    BackendFeature,
    CapabilityUnsupportedError,
    FilterOperator,
    StoreCapabilities,
)
from ragit.retrieval.strategies import DenseRetrieval, HybridRetrieval, KeywordRetrieval
from ragit.retrieval.types import RetrievalRequest


class MockVectorStore:
    def __init__(self, capabilities: StoreCapabilities | None = None) -> None:
        if capabilities is not None:
            self.capabilities = capabilities

    async def vector_search(self, *args, **kwargs):
        return [{"id": "v1", "document": "vector hit", "distance": 0.1}]

    async def keyword_search(self, *args, **kwargs):
        return [{"id": "k1", "document": "keyword hit", "keyword_score": 1.0}]


def test_capabilities_supports_operator_and_feature() -> None:
    caps = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ, FilterOperator.IN}),
        features=frozenset({BackendFeature.DOCUMENT_FETCH, BackendFeature.TRANSACTION_GUARANTEE}),
    )
    assert caps.supports_operator(FilterOperator.EQ)
    assert caps.supports_operator("eq")
    assert caps.supports_operator(FilterOperator.IN)
    assert not caps.supports_operator(FilterOperator.GT)

    assert caps.supports_feature(BackendFeature.DOCUMENT_FETCH)
    assert caps.supports_feature("document_fetch")
    assert caps.supports_feature(BackendFeature.TRANSACTION_GUARANTEE)
    assert not caps.supports_feature(BackendFeature.NATIVE_LEXICAL_SEARCH)


def test_capabilities_validate_constraints_fail_closed() -> None:
    caps = StoreCapabilities(
        features=frozenset({BackendFeature.DOCUMENT_FETCH}),
    )
    # Valid constraint passes
    caps.validate_constraints(["document_fetch"])

    # Unsupported constraint fails closed
    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        caps.validate_constraints(["transaction_guarantee"])
    assert "transaction_guarantee" in str(exc_info.value)
    assert "Failing closed" in str(exc_info.value)


def test_capabilities_validate_filters_fail_closed() -> None:
    # Caps with only equality filter
    caps = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
    )
    caps.validate_filters({"author": "Alice"})

    # IN filter should fail closed because IN operator is missing
    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        caps.validate_filters({"tags": ["news", "tech"]})
    assert "IN filters" in str(exc_info.value)


@pytest.mark.asyncio
async def test_hybrid_retrieval_fails_closed_on_unsupported_capability() -> None:
    unsupported_caps = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
        features=frozenset({BackendFeature.APPROXIMATE_LEXICAL_SEARCH}),
    )
    store = MockVectorStore(capabilities=unsupported_caps)

    async def fake_embedder(texts):
        return [[0.1, 0.2] for _ in texts]

    strategy = HybridRetrieval(
        vector_index=store,
        lexical_index=store,
        embedder=fake_embedder,
    )

    req = RetrievalRequest(
        query="test query",
        threshold=1.0,
        mandatory_capabilities={"transaction_guarantee"},
    )

    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        await strategy.retrieve(req)
    assert "transaction_guarantee" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dense_and_keyword_retrieval_fail_closed_on_unsupported_filter() -> None:
    caps_no_in = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
    )
    store = MockVectorStore(capabilities=caps_no_in)

    async def fake_embedder(texts):
        return [[0.1, 0.2] for _ in texts]

    dense = DenseRetrieval(vector_index=store, embedder=fake_embedder)
    keyword = KeywordRetrieval(lexical_index=store)

    req_with_in = RetrievalRequest(
        query="test",
        threshold=1.0,
        filters={"category": ["a", "b"]},
    )

    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        await dense.retrieve(req_with_in)
    assert "IN filters" in str(exc_info.value)

    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        await keyword.retrieve(req_with_in)
    assert "IN filters" in str(exc_info.value)
