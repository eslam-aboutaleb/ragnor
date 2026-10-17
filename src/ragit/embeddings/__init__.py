"""Embedding seam: function protocol, LiteLLM provider, registry, dimensions.

The LiteLLM integration ships behind the ``litellm`` extra and
is imported lazily (via the module ``__getattr__`` below), so
``import ragit.embeddings`` works without litellm installed.

New in this release:

* :class:`~ragit.embeddings.profile.EmbeddingProfile` — provider/model/dimension
  identity and canonical hash.
* :class:`~ragit.embeddings.executor.EmbeddingExecutor` — resilient embedding
  executor with retry, concurrency control, and response validation.
* :class:`~ragit.embeddings.cache.EmbeddingCache` — scoped, profile-keyed
  in-process cache with LRU eviction and optional TTL.
* :class:`~ragit.embeddings.metrics.EmbeddingMetrics` — accumulated call/hit/retry
  counters.
"""

from __future__ import annotations

import importlib
from typing import Any

from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.cache import EmbeddingCache, EmbeddingCacheConfig
from ragit.embeddings.dimensions import (
    FALLBACK_DIMENSION,
    get_embedding_dimension,
    register_dimension,
)
from ragit.embeddings.executor import (
    EmbeddingError,
    EmbeddingExecutor,
    EmbeddingPermanentError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)
from ragit.embeddings.metrics import EmbeddingMetrics
from ragit.embeddings.profile import (
    DistanceMetric,
    EmbeddingProfile,
    NormalizationMode,
    ProfileValidationError,
    TaskMode,
    profiles_compatible_for_query,
    profiles_require_reindex,
    validate_profile_at_startup,
)
from ragit.embeddings.registry import (
    embedding_registry,
    get_embedding_function,
)

_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    # Exported name -> (module, attribute).
    "LitellmEmbeddingFunction": (
        "ragit.embeddings.litellm",
        "LitellmEmbeddingFunction",
    ),
    "LiteLLMEmbeddingProvider": (
        "ragit.embeddings.registry",
        "LiteLLMEmbeddingProvider",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily import the LiteLLM integration so litellm stays optional."""
    if name in _LAZY_PROVIDERS:
        module_name, attribute = _LAZY_PROVIDERS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                "The LiteLLM integration requires the 'litellm' extra: pip install 'ragit[litellm]'"
            ) from exc
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Legacy / unchanged
    "EmbeddingFunction",
    "FALLBACK_DIMENSION",
    "LitellmEmbeddingFunction",
    "LiteLLMEmbeddingProvider",
    "embedding_registry",
    "get_embedding_dimension",
    "get_embedding_function",
    "register_dimension",
    # Profile
    "DistanceMetric",
    "EmbeddingProfile",
    "NormalizationMode",
    "ProfileValidationError",
    "TaskMode",
    "profiles_compatible_for_query",
    "profiles_require_reindex",
    "validate_profile_at_startup",
    # Executor
    "EmbeddingError",
    "EmbeddingExecutor",
    "EmbeddingPermanentError",
    "EmbeddingResponseError",
    "EmbeddingTransientError",
    # Cache
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    # Metrics
    "EmbeddingMetrics",
]
