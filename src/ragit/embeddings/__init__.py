"""Embedding seam: function protocol, LiteLLM provider, registry, dimensions."""

from __future__ import annotations

from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.dimensions import (
    FALLBACK_DIMENSION,
    get_embedding_dimension,
    register_dimension,
)
from ragit.embeddings.litellm import LitellmEmbeddingFunction
from ragit.embeddings.registry import (
    LiteLLMEmbeddingProvider,
    embedding_registry,
    get_embedding_function,
)

__all__ = [
    "EmbeddingFunction",
    "FALLBACK_DIMENSION",
    "LitellmEmbeddingFunction",
    "LiteLLMEmbeddingProvider",
    "embedding_registry",
    "get_embedding_dimension",
    "get_embedding_function",
    "register_dimension",
]
