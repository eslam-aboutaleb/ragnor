"""Retrieval orchestration, strategies, capabilities, and types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragit.retrieval.capabilities import (
    BackendFeature,
    CapabilityUnsupportedError,
    FilterOperator,
    StoreCapabilities,
    get_backend_capabilities,
)
from ragit.retrieval.query_transform import (
    BoundedQueryTransformer,
    QueryTransformer,
)
from ragit.retrieval.retriever import (
    HybridRetriever,
    RetrieverSpec,
    as_dicts,
)
from ragit.retrieval.scores import (
    RawScore,
    ScoreKind,
)
from ragit.retrieval.strategies import (
    DenseRetrieval,
    HybridRetrieval,
    KeywordRetrieval,
    Reranker,
    RetrievalStrategy,
)
from ragit.retrieval.types import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedDocument,
    ScopeViolationError,
)

if TYPE_CHECKING:
    from ragit.retrieval.rerankers import LiteLLMReranker

__all__ = [
    "BackendFeature",
    "BoundedQueryTransformer",
    "CapabilityUnsupportedError",
    "DenseRetrieval",
    "FilterOperator",
    "HybridRetrieval",
    "HybridRetriever",
    "KeywordRetrieval",
    "LiteLLMReranker",
    "QueryTransformer",
    "RawScore",
    "Reranker",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievalStrategy",
    "RetrievedDocument",
    "RetrieverSpec",
    "ScopeViolationError",
    "ScoreKind",
    "StoreCapabilities",
    "as_dicts",
    "get_backend_capabilities",
]


def __getattr__(name: str) -> Any:
    if name == "LiteLLMReranker":
        from ragit.retrieval.rerankers import LiteLLMReranker  # noqa: PLC0415

        return LiteLLMReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
