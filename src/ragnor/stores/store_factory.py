"""Abstract factory for vector store backends.

:class:`VectorStoreFactory` is the abstraction that creates
vector stores. ragnor's default implementation,
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
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ragnor.core.registry import Registry
from ragnor.stores.base import VectorStore

if TYPE_CHECKING:
    from ragnor.core.config import RagSettings


class VectorStoreFactory(ABC):
    """Abstract factory that creates vector stores for any backend.

    Subclass this to create stores for a vector database ragnor
    does not know about, or to customize backend construction.
    The factory is the single seam between ragnor and the
    vector database.
    """

    @abstractmethod
    def create(
        self,
        settings: RagSettings | None = None,
        metric: str | None = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create a vector store.

        Args:
            settings: Host settings; ``vector_store_provider``
                selects the backend.
            metric: Distance metric for the vector stage
                (``"l2"`` by default, ``"cosine"`` opt-in).
                Forwarded to the backend constructor only when it
                is not ``None``, so custom backends whose
                constructor does not accept the kwarg keep
                working. ``settings`` is forwarded only to
                backends that declare an explicit ``settings``
                parameter.
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
                to ragnor's ``vector_store_registry``.
            default_provider: Provider used when ``settings`` is
                omitted or carries no ``vector_store_provider``.
        """
        if registry is None:
            from ragnor.stores.registry import vector_store_registry  # noqa: PLC0415

            registry = vector_store_registry
        self._registry = registry
        self._default_provider = default_provider

    def create(
        self,
        settings: RagSettings | None = None,
        metric: str | None = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create a vector store from the registry.

        ``metric`` is forwarded to the selected backend only when
        it is not ``None``, so a custom registered backend whose
        constructor does not accept the kwarg keeps working.
        ``settings`` is forwarded only to backends whose
        constructor declares an explicit ``settings`` parameter,
        so a backend can read its optional host fields (such as
        ``rag_distance_metric`` or the pgvector
        ``database_url``); a constructor that only accepts
        ``**kwargs`` receives the documented registry kwargs
        unchanged.
        """
        provider = self._default_provider
        if settings is not None:
            provider = getattr(settings, "vector_store_provider", self._default_provider)
        self._load_builtin_provider(provider)
        store_class = self._registry.get(provider)
        if metric is not None:
            kwargs["metric"] = metric
        if (
            settings is not None
            and "settings" not in kwargs
            and self._accepts_settings(store_class)
        ):
            kwargs["settings"] = settings
        return store_class(**kwargs)

    @staticmethod
    def _accepts_settings(store_class: type) -> bool:
        """Return whether ``store_class.__init__`` declares a ``settings`` parameter.

        Only an explicit parameter counts: a constructor accepting
        ``**kwargs`` is treated as not declaring it, so the
        registry kwargs a custom backend receives stay exactly
        the documented ones.
        """
        try:
            signature = inspect.signature(store_class)
        except (TypeError, ValueError):  # pragma: no cover - exotic builtins
            return False
        return "settings" in signature.parameters

    def _load_builtin_provider(self, name: str) -> None:
        """Import a backend module so its registration runs.

        ragnor ships one module per built-in backend, named
        after its provider (``ragnor.stores.memory``,
        ``ragnor.stores.pgvector``, ``ragnor.stores.qdrant``,
        ...). Backend modules register themselves on import, and
        importing them lazily by convention keeps every
        backend's driver dependency out of ragnor's import graph
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
        module_name = f"ragnor.stores.{key}"
        if importlib.util.find_spec(module_name) is None:
            return
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"The {name!r} vector store backend could not be loaded. "
                f"Built-in backends ship as optional extras -- install the "
                f"backend's extra, e.g. pip install 'ragnor[{key}]'. "
                f"Underlying error: {exc}"
            ) from exc
