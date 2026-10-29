"""Vector store registry and factory.

Providers register themselves under a name (``pgvector``,
``memory``) and are looked up case-insensitively through the
generic :class:`ragnor.core.registry.Registry`. The factory
:func:`get_vector_store` replaces the if/else chain the host
application used before the extraction; it delegates to the
registry-backed
:class:`ragnor.stores.store_factory.RegistryVectorStoreFactory`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnor.core.registry import Registry
from ragnor.stores.base import VectorStore
from ragnor.stores.store_factory import RegistryVectorStoreFactory

if TYPE_CHECKING:
    from ragnor.core.config import RagSettings
    from ragnor.db.session import SessionProvider
    from ragnor.stores.base import EmbeddingFunction

vector_store_registry: Registry[VectorStore] = Registry[VectorStore]()
"""Registry of vector store provider classes."""


def get_vector_store(  # noqa: PLR0913, PLR0917
    table_name: str,
    id_field: str = "id",
    embedding_dim: int | None = None,
    settings: RagSettings | None = None,
    session_factory: SessionProvider | None = None,
    embedding_fn: EmbeddingFunction | None = None,
    metric: str | None = None,
) -> VectorStore:
    """Create a vector store through the registry-backed factory.

    The provider class is read from ``settings.vector_store_provider``
    (default ``"pgvector"``) and instantiated with the remaining
    arguments, so host applications inject their session factory,
    embedding function, and embedding dimension through the factory
    instead of the store importing host modules.

    Args:
        table_name: Name of the database table.
        id_field: Name of the primary key column.
        embedding_dim: Dimensionality of the embedding vectors.
        settings: Host settings. ``vector_store_provider`` selects
            the provider; the settings object is forwarded to the
            backend constructor when it declares an explicit
            ``settings`` parameter, so backends can read their
            optional host fields such as ``rag_distance_metric``
            for the metric (when ``metric`` is omitted).
        session_factory: Session provider injected into the store.
        embedding_fn: Embedding function injected into the store for
            automatic query embedding.
        metric: Distance metric for the vector stage (``"l2"`` by
            default, ``"cosine"`` opt-in; resolved
            case-insensitively). Forwarded to the selected backend
            only when it is not ``None``, so custom registered
            backends whose constructor does not accept the kwarg
            keep working.

    Returns:
        Configured VectorStore instance based on the provider setting.

    Raises:
        KeyError: If no provider is registered under the configured
            name. The message lists the registered names.
        ValueError: If ``metric`` is not a supported distance metric
            (raised by the selected backend).
    """
    factory = RegistryVectorStoreFactory()
    kwargs: dict[str, Any] = {
        "table_name": table_name,
        "id_field": id_field,
        "embedding_dim": embedding_dim,
        "session_factory": session_factory,
        "embedding_fn": embedding_fn,
    }
    if metric is not None:
        kwargs["metric"] = metric
    return factory.create(settings=settings, **kwargs)
