"""Vector store abstraction for ragnor.

Defines the interface for hybrid retrieval (vector + keyword
search) operations, allowing different backends (pgvector,
in-memory, etc.) to be swapped via the store registry without
changing the retrieval or ingestion code.

Storage is split from retrieval strategy. Three structural
capability Protocols describe what a backend can do:

* :class:`VectorIndex` -- nearest-neighbor search over
  embeddings;
* :class:`LexicalIndex` -- keyword search over document
  text;
* :class:`DocumentStore` -- document lifecycle (upsert,
  count, delete, get).

A host adapts an existing search engine to a capability
Protocol by implementing its methods -- no ragnor
inheritance required. :class:`HybridVectorStore` composes
the two search capabilities into the legacy
:class:`VectorStore` hybrid contract (Reciprocal Rank
Fusion), and :class:`LegacyHybridStoreAdapter` wraps a
pre-refactor backend that only implements the fused
contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from ragnor.core.exceptions import BackendUnavailableError, OperationTimeoutError
from ragnor.stores.metrics import DistanceMetric, resolve_distance_metric
from ragnor.stores._scoring import fuse_hybrid
from ragnor.core.validation import (
    validate_embedding,
    validate_identifier,
    validate_search_identifiers,
)


EmbeddingFunction = Callable[[list[str]], Awaitable[list[list[float]]]]
"""Async callable that embeds a list of texts into vectors."""

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """Structural capability: similarity search over embeddings.

    Implementations search a vector index for the nearest
    neighbors of ``embedding`` and return at most
    ``n_results`` result dicts ordered by similarity (best
    first). The method requires a valid embedding -- it does
    not auto-embed the query -- and it always raises on
    operational failure (there is no ``strict`` parameter;
    the opt-in swallow contract lives in the legacy
    ``hybrid_search`` and in the retrieval strategies).

    Result dict shape: ``id``, ``document``, ``metadata``,
    ``distance`` (the similarity distance), and
    ``keyword_score`` (``None`` -- this stage found nothing
    by keyword). ``**filters`` are equality filters matched
    against document fields; keys must be valid SQL
    identifiers. A supplied ``session`` is caller-owned;
    without one, each backend follows its documented
    lifecycle.
    """

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Vector similarity search protocol method."""
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Structural capability: keyword search over document text.

    Implementations search their lexical index for documents
    matching ``query`` and return at most ``n_results``
    result dicts ordered by keyword relevance (best first).
    There is no ``strict`` parameter: operational failures
    always propagate (the opt-in swallow contract lives in
    the legacy ``hybrid_search`` and in the retrieval
    strategies).

    Result dict shape: ``id``, ``document``, ``metadata``,
    ``distance`` (``None`` -- this stage found nothing by
    vector similarity), and ``keyword_score`` (the keyword
    relevance score). ``**filters`` are equality filters
    matched against document fields; keys must be valid SQL
    identifiers. A supplied ``session`` is caller-owned;
    without one, each backend follows its documented
    lifecycle.

    Performance notice: implementing this Protocol does not
    guarantee production-scale keyword search. pgvector is
    the only built-in native lexical index (PostgreSQL
    full-text search); the other built-in backends expose
    this Protocol through an in-process token-overlap scan
    intended for local development, small corpora, and
    testing. Hosts at scale combine a pgvector
    :class:`VectorIndex` with an external lexical engine
    adapted to this Protocol.
    """

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Lexical keyword search protocol method."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Structural capability: document lifecycle operations.

    Implementations persist documents (``upsert``), count
    them (``count``), delete them (``delete``), and fetch
    them directly (``get``). ``delete`` and ``get`` have no
    ``strict`` parameter: failures always propagate, and
    ``delete`` requires at least one selector (``ids`` or
    ``filters``) and rejects an empty ``ids`` list -- an
    empty id list never means "delete everything". When both
    selectors are supplied, both restrictions apply
    (conjunction). A supplied ``session`` is caller-owned;
    without one, each backend follows its documented
    lifecycle. ``delete`` returns the number of deleted
    documents, with consistent meaning across backends.
    ``get`` returns ``id``/``document``/``metadata`` dicts
    (score keys are meaningless for direct fetches); with no
    selector it returns every document.
    """

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Upsert documents into the store."""
        ...

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the store."""
        ...

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete documents selected by ``ids`` and/or ``filters``."""
        ...

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``."""
        ...


class VectorStore(ABC):
    """Abstract interface for vector store operations.

    The legacy composite contract: hybrid search (dense +
    keyword, fused with Reciprocal Rank Fusion), counting,
    and upsert. New backends should implement the capability
    Protocols (:class:`VectorIndex`, :class:`LexicalIndex`,
    :class:`DocumentStore`) and subclass
    :class:`HybridVectorStore`; this ABC remains the
    registry-facing name the host code and the docs depend
    on.

    Session ownership rule: when the caller supplies a
    ``session`` to a method, the caller owns the transaction
    and is responsible for committing or rolling it back.
    When no session is supplied, the store opens its own
    session and commits where the operation is transactional.
    """

    async def get_embeddings(self, ids: list[str], session: Any = None) -> dict[str, list[float]]:
        """Fetch vectors for generation copies without invoking an embedding provider."""
        raise NotImplementedError("Store does not support embedding fetch for metadata updates")

    @abstractmethod
    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Perform hybrid vector + PostgreSQL full-text search using Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector.
            n_results: Maximum number of results to return.
            threshold: Maximum distance for vector search part.
            text_field: Name of the column containing the document text.
            metadata_fields: Additional column names to include in the
                result metadata dict.
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            strict: When ``True``, retrieval failures raise
                :class:`ragnor.core.types.RetrievalError` instead of being
                swallowed into an empty result list.
            **filters: Additional filters (e.g., tenant_id).

        Returns:
            List of result dicts with keys: id, document, metadata,
            distance, keyword_score, _rrf_score.
        """
        ...

    @abstractmethod
    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the store.

        Args:
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            strict: When ``True``, failures raise
                :class:`ragnor.core.types.RetrievalError` instead of
                being swallowed into ``0``.
            **filters: Optional filters (e.g., tenant_id).

        Returns:
            Number of matching documents.
        """
        ...

    @abstractmethod
    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Upsert documents into the store.

        Args:
            documents: List of document dicts with keys: id, text,
                metadata, embedding.
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            text_field: Name of the text column.
            embedding_field: Name of the embedding column.
            id_field: Name of the primary key column. Defaults to the
                store's configured id field.
            extra_fields: Additional static fields to include in every
                INSERT (e.g., ``{"tenant_id": "..."}``).
        """
        ...


class HybridVectorStore(VectorStore):
    """Composite base: RRF-fused hybrid search over capability stages.

    Built-in in-process backends subclass this base and
    implement the :class:`VectorIndex` and
    :class:`LexicalIndex` capability methods
    (:meth:`vector_search` and :meth:`keyword_search`); this
    base provides the legacy :meth:`hybrid_search` by running
    both stages -- each independently ranked and truncated to
    ``n_results`` by the capability method -- and fusing the
    two truncated result lists with Reciprocal Rank Fusion
    (:func:`ragnor.stores._scoring.fuse_hybrid`).

    The constructor accepts the uniform registry kwargs
    (``table_name``, ``id_field``, ``embedding_dim``,
    ``session_factory``, ``settings``, ``embedding_fn``,
    ``metric``) so subclasses stay drop-in compatible with
    :func:`ragnor.stores.get_vector_store`.

    ``metric`` selects the vector stage's distance metric
    (``"l2"`` by default, ``"cosine"`` opt-in); see
    :mod:`ragnor.stores.metrics`. Distance thresholds are distances
    under that metric: a threshold is never comparable across
    two different metrics, so a host that switches metrics must
    re-tune it.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str = "documents",
        id_field: str = "id",
        embedding_dim: int | None = None,
        session_factory: Any = None,
        settings: Any = None,
        embedding_fn: EmbeddingFunction | None = None,
        metric: str | None = None,
    ) -> None:
        """Initialize the composite store.

        Args:
            table_name: Logical table (or collection) name.
            id_field: Name of the primary key field.
            embedding_dim: Expected embedding dimensionality. When
                provided, embeddings are validated like the pgvector
                store does.
            session_factory: Session provider the store uses when a
                method is called without a session (backends without
                a database ignore it).
            settings: Host settings (backends without a database
                ignore them). The optional ``rag_distance_metric``
                field selects the metric when ``metric`` is omitted.
            embedding_fn: Async callable that embeds a list of texts,
                used to auto-embed queries when ``hybrid_search`` is
                called with an empty embedding.
            metric: Distance metric for the vector stage (``"l2"``
                by default, ``"cosine"`` opt-in; resolved
                case-insensitively).

        Raises:
            ValueError: If ``table_name`` or ``id_field`` contain
                unsafe characters, or ``metric`` is not a supported
                distance metric.
        """
        validate_identifier(table_name, "table_name")
        validate_identifier(id_field, "id_field")
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self._session_factory = session_factory
        self._settings = settings
        self._embedding_fn = embedding_fn
        self.metric: DistanceMetric = resolve_distance_metric(metric, settings)

    @abstractmethod
    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Search for the nearest neighbors of ``embedding``."""
        ...

    @abstractmethod
    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Search for documents matching ``query``."""
        ...

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector stage plus keyword stage, RRF-fused.

        Validates identifiers and ``n_results`` and auto-embeds an
        empty query embedding via the injected ``embedding_fn``
        (raising ``ValueError`` when none is configured) before the
        search runs; those ``ValueError`` s propagate. The two
        capability stages then run and their truncated results are
        fused with Reciprocal Rank Fusion. A stage failure is
        logged and swallowed into an empty list unless
        ``strict=True``, which raises
        :class:`ragnor.core.types.RetrievalError`.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty, the query is
                embedded automatically using the injected
                ``embedding_fn``.
            n_results: Maximum number of results to return.
            threshold: Maximum distance for the vector search part.
            text_field: Name of the field containing the document text.
            metadata_fields: Fields to include in the result metadata.
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            strict: When ``True``, a failed search raises
                :class:`ragnor.core.types.RetrievalError`.
            **filters: Equality filters matched against document
                fields (metadata and extra fields).

        Returns:
            list[dict]: Result dicts sorted by RRF score, with the
            same keys as the other backends' hybrid search.

        Raises:
            ValueError: If identifiers are unsafe, ``n_results`` is
                not positive, or the embedding is empty and no
                ``embedding_fn`` is configured.
            ragnor.core.types.RetrievalError: If a search stage fails and
                ``strict=True``.
        """
        validate_search_identifiers(text_field, filters, metadata_fields)
        metadata_fields = metadata_fields or []

        if n_results < 1:
            raise ValueError(f"n_results must be a positive integer; got {n_results}")

        if not embedding:
            if self._embedding_fn is None:
                raise ValueError(
                    "embedding_fn is required to auto-embed the query; "
                    "pass an embedding or construct the store with "
                    "embedding_fn"
                )
            embedding_list = await self._embedding_fn([query])
            embedding = embedding_list[0]

        if self.embedding_dim is not None:
            validate_embedding(embedding, self.embedding_dim, label="query embedding")

        try:
            vector_results = await self.vector_search(
                embedding=embedding,
                n_results=n_results,
                threshold=threshold,
                text_field=text_field,
                metadata_fields=metadata_fields,
                session=session,
                **filters,
            )
            keyword_results = await self.keyword_search(
                query=query,
                n_results=n_results,
                text_field=text_field,
                metadata_fields=metadata_fields,
                session=session,
                **filters,
            )
            return fuse_hybrid(vector_results, keyword_results, n_results)
        except Exception as exc:
            logger.error("Hybrid search failed on %s: %s", self.table_name, type(exc).__name__)
            if strict:
                if isinstance(exc, TimeoutError):
                    raise OperationTimeoutError(
                        f"Hybrid search timed out on {self.table_name} ({type(exc).__name__})"
                    ) from exc
                raise BackendUnavailableError(
                    f"Hybrid search failed on {self.table_name} ({type(exc).__name__})"
                ) from exc
            return []


class LegacyHybridStoreAdapter(VectorStore):
    """Wrap a pre-refactor custom backend for the default retrieval path.

    A legacy backend implements the fused :class:`VectorStore`
    contract (``hybrid_search``/``count``/``upsert``) but not
    the capability Protocols. A fused hybrid result cannot be
    decomposed into independent vector and keyword stages, so
    this adapter deliberately implements no capability methods:
    it must not pretend to satisfy :class:`VectorIndex`,
    :class:`LexicalIndex`, or :class:`DocumentStore`
    (``isinstance(adapter, VectorIndex)`` is ``False``).
    Legacy backends keep working on the default retrieval path
    with zero changes -- wrap them in this adapter and register
    the adapter under the backend's provider name.
    """

    def __init__(self, backend: VectorStore) -> None:
        """Wrap a legacy backend.

        Args:
            backend: A pre-refactor backend implementing the
                :class:`VectorStore` ABC.
        """
        self._backend = backend

    @property
    def backend(self) -> VectorStore:
        """The wrapped legacy backend."""
        return self._backend

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int = 5,
        threshold: float = 0.0,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Delegate to the wrapped backend's ``hybrid_search``."""
        return await self._backend.hybrid_search(
            query=query,
            embedding=embedding,
            n_results=n_results,
            threshold=threshold,
            text_field=text_field,
            metadata_fields=metadata_fields,
            session=session,
            strict=strict,
            **filters,
        )

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Delegate to the wrapped backend's ``count``."""
        return await self._backend.count(session=session, strict=strict, **filters)

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to the wrapped backend's ``upsert``."""
        await self._backend.upsert(
            documents=documents,
            session=session,
            text_field=text_field,
            embedding_field=embedding_field,
            id_field=id_field,
            extra_fields=extra_fields,
        )
