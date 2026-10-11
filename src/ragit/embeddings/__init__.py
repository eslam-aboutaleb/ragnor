"""Embedding seam: function protocol, LiteLLM provider, registry, dimensions.

The LiteLLM integration ships behind the ``litellm`` extra and
is imported lazily (via the module ``__getattr__`` below), so
``import ragit.embeddings`` works without litellm installed.
"""

from __future__ import annotations

import importlib
from typing import Any

from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.dimensions import (
    FALLBACK_DIMENSION,
    get_embedding_dimension,
    register_dimension,
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
    "EmbeddingFunction",
    "FALLBACK_DIMENSION",
    "LitellmEmbeddingFunction",
    "LiteLLMEmbeddingProvider",
    "embedding_registry",
    "get_embedding_dimension",
    "get_embedding_function",
    "register_dimension",
]
