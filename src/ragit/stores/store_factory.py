"""Abstract factory for vector store backends.

:class:`VectorStoreFactory` is the abstraction that creates
vector stores. ragit's default implementation,
:class:`RegistryVectorStoreFactory`, selects a backend from
the provider registry (``memory``, ``pgvector``, ``qdrant``,
``chroma``, ``faiss``, or any host-registered backend), so
the vector database is chosen through a factory instead of
hardcoded imports -- including the zero-SQL in-memory store,
which the same factory produces without touching any database.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ragit.registry import Registry
from ragit.stores.base import VectorStore

if TYPE_CHECKING:
    from ragit.config import RagSettings


class VectorStoreFactory(ABC):
    """Abstract factory that creates vector stores for any backend.

    Subclass this to create stores for a vector database ragit
    does not know about, or to customize backend construction.
    The factory is the single seam between ragit and the
    vector database.
    """

    @abstractmethod
    def create(
        self,
        settings: RagSettings | None = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create a vector store.

        Args:
            settings: Host settings; ``vector_store_provider``
                selects the backend.
            **kwargs: Backend constructor arguments
                (``table_name``, ``id_field``, ``embedding_dim``,
                ``session_factory``, ``settings``, ``embedding_fn``).

        Returns:
            A configured :class:`VectorStore`.
        """


class RegistryVectorStoreFactory(VectorStoreFactory):
    """Creates vector stores from the provider registry.

    Reads ``settings.vector_store_provider`` (default
    ``"memory"``), imports the matching backend module on
    demand so its driver dependency stays optional, and
    instantiates the provider with the standard kwargs.
    """

    def __init__(
        self,
        registry: Registry[VectorStore] | None = None,
        default_provider: str = "memory",
    ) -> None:
        """Create the factory.

        Args:
            registry: Provider registry to select from; defaults
                to ragit's ``vector_store_registry``.
            default_provider: Provider used when ``settings`` is
                omitted or carries no ``vector_store_provider``.
        """
        if registry is None:
            from ragit.stores.registry import vector_store_registry  # noqa: PLC0415

            registry = vector_store_registry
        self._registry = registry
        self._default_provider = default_provider

    def create(
        self,
        settings: RagSettings | None = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create a vector store from the registry."""
        provider = self._default_provider
        if settings is not None:
            provider = getattr(
                settings, "vector_store_provider", self._default_provider
            )
        self._load_builtin_provider(provider)
        store_class = self._registry.get(provider)
        return store_class(**kwargs)

    def _load_builtin_provider(self, name: str) -> None:
        """Import a backend module so its registration runs.

        ragit ships one module per built-in backend, named
        after its provider (``ragit.stores.memory``,
        ``ragit.stores.pgvector``, ``ragit.stores.qdrant``,
        ...). Backend modules register themselves on import, and
        importing them lazily by convention keeps every
        backend's driver dependency out of ragit's import graph
        until the backend is actually selected -- so adding a
        backend is a matter of dropping in a module, not
        editing a chain of imports. A provider already in the
        registry (for example a host-registered override) is
        left untouched, and a name with no matching module is
        left to the registry's ``KeyError``.

        Raises:
            ImportError: If the backend module exists but its
                optional driver dependency is not installed.
        """
        key = name.lower()
        if key in self._registry.list():
            return
        if not key.isidentifier():
            return
        module_name = f"ragit.stores.{key}"
        if importlib.util.find_spec(module_name) is None:
            return
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"The {name!r} vector store backend could not be loaded. "
                f"Built-in backends ship as optional extras -- install the "
                f"backend's extra, e.g. pip install 'ragit[{key}]'. "
                f"Underlying error: {exc}"
            ) from exc
