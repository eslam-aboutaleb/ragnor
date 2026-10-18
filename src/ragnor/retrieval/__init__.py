"""Retrieval orchestration, strategies, capabilities, and types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnor.retrieval.capabilities import (
    BackendFeature,
    CapabilityUnsupportedError,
    FilterOperator,
    StoreCapabilities,
    get_backend_capabilities,
)
from ragnor.retrieval.query_transform import (
    BoundedQueryTransformer,
    QueryTransformer,
)
from ragnor.retrieval.retriever import (
    HybridRetriever,
    RetrieverSpec,
    as_dicts,
)
from ragnor.retrieval.scores import (
    RawScore,
    ScoreKind,
)
from ragnor.retrieval.strategies import (
    DenseRetrieval,
    HybridRetrieval,
    KeywordRetrieval,
    Reranker,
    RetrievalStrategy,
)
from ragnor.retrieval.types import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedDocument,
    ScopeViolationError,
)

if TYPE_CHECKING:
    from ragnor.retrieval.rerankers import LiteLLMReranker

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
        from ragnor.retrieval.rerankers import LiteLLMReranker  # noqa: PLC0415

        return LiteLLMReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
