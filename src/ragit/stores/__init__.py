"""Vector store abstractions and providers for ragit.

The package is vector-database agnostic: the
:class:`VectorStore` ABC defines the contract, and each
backend lives in its own module that registers itself with
the provider registry on import. Backend modules are loaded
lazily (by :class:`ragit.stores.store_factory.RegistryVectorStoreFactory`
or the ``__getattr__`` below) so every backend's driver
dependency stays optional.
"""

from __future__ import annotations

import importlib
from typing import Any

from ragit.stores.base import EmbeddingFunction, VectorStore
from ragit.stores.memory import InMemoryVectorStore
from ragit.stores.registry import get_vector_store, vector_store_registry
from ragit.stores.store_factory import RegistryVectorStoreFactory, VectorStoreFactory

_LAZY_BACKENDS: dict[str, tuple[str, str, str]] = {
    # Exported name -> (module, attribute, optional extra).
    "PgVectorStore": ("ragit.stores.pgvector", "PgVectorStore", "pgvector"),
    "QdrantStore": ("ragit.stores.qdrant", "QdrantStore", "qdrant"),
    "ChromaStore": ("ragit.stores.chroma", "ChromaStore", "chroma"),
    "FaissStore": ("ragit.stores.faiss", "FaissStore", "faiss"),
}


def __getattr__(name: str) -> Any:
    """Lazily import a database backend so its driver stays optional."""
    if name in _LAZY_BACKENDS:
        module_name, attribute, extra = _LAZY_BACKENDS[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"The {extra} backend requires the '{extra}' extra: "
                f"pip install 'ragit[{extra}]'"
            ) from exc
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChromaStore",
    "EmbeddingFunction",
    "FaissStore",
    "InMemoryVectorStore",
    "PgVectorStore",
    "QdrantStore",
    "RegistryVectorStoreFactory",
    "VectorStore",
    "VectorStoreFactory",
    "get_vector_store",
    "vector_store_registry",
]
