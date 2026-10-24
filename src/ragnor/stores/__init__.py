"""Vector store abstractions and providers for ragnor.

The package is vector-database agnostic: the
:class:`VectorStore` ABC defines the legacy fused
contract, the capability Protocols
(:class:`VectorIndex`, :class:`LexicalIndex`,
:class:`DocumentStore`) describe what a backend can
do structurally, and :class:`HybridVectorStore`
composes the two search capabilities into the fused
contract. Each backend lives in its own module that
registers itself with the provider registry on
import. Backend modules are loaded lazily (by
:class:`ragnor.stores.store_factory.RegistryVectorStoreFactory`
or the ``__getattr__`` below) so every backend's driver
dependency stays optional.
"""

from __future__ import annotations

import importlib
from typing import Any

from ragnor.stores.metrics import (
    DEFAULT_DISTANCE_METRIC,
    SUPPORTED_DISTANCE_METRICS,
    DistanceMetric,
    cosine_distance,
    cosine_similarity,
    normalize,
    resolve_distance_metric,
)
from ragnor.stores.base import (
    DocumentStore,
    EmbeddingFunction,
    HybridVectorStore,
    LegacyHybridStoreAdapter,
    LexicalIndex,
    VectorIndex,
    VectorStore,
)
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.stores.registry import get_vector_store, vector_store_registry
from ragnor.stores.store_factory import RegistryVectorStoreFactory, VectorStoreFactory

_LAZY_BACKENDS: dict[str, tuple[str, str, str]] = {
    # Exported name -> (module, attribute, optional extra).
    "PgVectorStore": ("ragnor.stores.pgvector", "PgVectorStore", "pgvector"),
    "QdrantStore": ("ragnor.stores.qdrant", "QdrantStore", "qdrant"),
    "ChromaStore": ("ragnor.stores.chroma", "ChromaStore", "chroma"),
    "FaissStore": ("ragnor.stores.faiss", "FaissStore", "faiss"),
}


def __getattr__(name: str) -> Any:
    """Lazily import a database backend so its driver stays optional."""
    if name in _LAZY_BACKENDS:
        module_name, attribute, extra = _LAZY_BACKENDS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"The {extra} backend requires the '{extra}' extra: pip install 'ragnor[{extra}]'"
            ) from exc
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChromaStore",
    "DEFAULT_DISTANCE_METRIC",
    "DistanceMetric",
    "DocumentStore",
    "EmbeddingFunction",
    "FaissStore",
    "HybridVectorStore",
    "InMemoryVectorStore",
    "LegacyHybridStoreAdapter",
    "LexicalIndex",
    "PgVectorStore",
    "QdrantStore",
    "RegistryVectorStoreFactory",
    "SUPPORTED_DISTANCE_METRICS",
    "VectorIndex",
    "VectorStore",
    "VectorStoreFactory",
    "cosine_distance",
    "cosine_similarity",
    "get_vector_store",
    "normalize",
    "resolve_distance_metric",
    "vector_store_registry",
]
