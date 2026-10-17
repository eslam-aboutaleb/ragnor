"""Generic hybrid retrieval orchestrator.

The orchestrator is table-agnostic: the caller
supplies a :class:`RetrieverSpec` naming the table
and the metadata columns to surface, so no table
name is hardcoded anywhere in ragit.

The retriever embeds the query, delegates the
two-stage search (vector similarity + keyword
search, merged with Reciprocal Rank Fusion) to the
injected vector store, and returns
:class:`ragit.types.SearchResult` dataclasses.
:func:`as_dicts` converts them back to the dict
shape the stores return, so existing dict consumers
keep working during the transition.

A retrieval strategy (see
:mod:`ragit.retrieval.strategies`) can be
injected instead: when a strategy is supplied, the
constructor builds nothing -- no embedder, no store
-- and retrieval delegates to the strategy, which
owns its capability dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragit.config import RagSettings
from ragit.embeddings import get_embedding_dimension, get_embedding_function
from ragit.retrieval.types import RetrievalRequest, RetrievalResponse
from ragit.stores import get_vector_store
from ragit.telemetry.events import EventKind, NullEventSink, TelemetryEvent
from ragit.types import SearchResult
from ragit.validation import validate_identifier

if TYPE_CHECKING:
    from ragit.db.session import SessionProvider

logger = logging.getLogger(__name__)


@dataclass
class RetrieverSpec:
    """Description of the table a :class:`HybridRetriever` searches.

    Attributes:
        table_name: Table (or view) holding the documents.
        id_field: Primary key column.
        text_field: Column holding the document text.
        metadata_fields: Columns surfaced in each result's
            metadata dict.
        embedding_dim: Expected embedding dimensionality. When
            ``None``, the dimension registered for
            ``settings.embedding_model`` is used.
    """

    table_name: str
    id_field: str = "id"
    text_field: str = "text"
    metadata_fields: list[str] = field(default_factory=list)
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        """Validate every SQL identifier in the spec."""
        validate_identifier(self.table_name, "table_name")
        validate_identifier(self.id_field, "id_field")
        validate_identifier(self.text_field, "text_field")
        for name in self.metadata_fields:
            validate_identifier(name, "metadata_field")


def _to_search_result(row: dict[str, Any]) -> SearchResult:
    """Convert a store result dict to a :class:`SearchResult`."""
    return SearchResult(
        id=row.get("id"),
        document=row["document"],
        metadata=row.get("metadata") or {},
        distance=row.get("distance"),
        keyword_score=row.get("keyword_score"),
        rrf_score=row.get("_rrf_score", 0.0),
    )


def as_dicts(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Convert :class:`SearchResult` dataclasses to dicts.

    The dicts carry the same keys the vector stores return
    (``id``, ``document``, ``metadata``, ``distance``,
    ``keyword_score``, ``_rrf_score``), so callers written
    against the dict-based API keep working.

    Args:
        results: Search results from :meth:`HybridRetriever.retrieve`.

    Returns:
        The equivalent list of result dicts.
    """
    return [
        {
            "id": result.id,
            "document": result.document,
            "metadata": result.metadata,
            "distance": result.distance,
            "keyword_score": result.keyword_score,
            "_rrf_score": result.rrf_score,
        }
        for result in results
    ]


class HybridRetriever:
    """Hybrid retrieval over an arbitrary document table.

    By default the retriever resolves the embedder
    and the hybrid store from ``settings`` and
    ``spec``. When a retrieval strategy is injected,
    the constructor builds nothing -- no embedder, no
    store -- and :meth:`retrieve` delegates to the
    strategy, which owns its capability dependencies
    (for example a pgvector
    :class:`ragit.stores.base.VectorIndex` plus an
    external lexical engine adapted to
    :class:`ragit.stores.base.LexicalIndex`, or a
    keyword-only strategy with no embedding
    dependency at all).
    """

    def __init__(  # noqa: PLR0913
        self,
        spec: RetrieverSpec,
        settings: RagSettings,
        session_factory: SessionProvider | None = None,
        strategy: Any = None,
        event_sink: Any | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            spec: Table and column description of the index.
            settings: Settings selecting the embedding provider
                and carrying the default distance threshold.
            session_factory: Session provider the vector store
                uses when a method is called without a session.
                When omitted, the store opens its own sessions
                from ``settings.database_url``.
            strategy: Optional retrieval strategy (see
                :mod:`ragit.retrieval.strategies`). When
                supplied, the constructor builds nothing --
                no embedder and no store -- and retrieval
                delegates to the strategy. When omitted (the
                default), the embedder and the hybrid store
                are built from ``settings`` and ``spec``.
            event_sink: Optional sink for telemetry events.
        """
        self._spec = spec
        self._settings = settings
        self._session_factory = session_factory
        self._strategy = strategy
        self._event_sink = event_sink if event_sink is not None else NullEventSink()
        if strategy is None:
            self._embed_fn = get_embedding_function(settings)
            embedding_dim = spec.embedding_dim
            if embedding_dim is None:
                embedding_dim = get_embedding_dimension(settings.embedding_model)
            self._store = get_vector_store(
                table_name=spec.table_name,
                id_field=spec.id_field,
                embedding_dim=embedding_dim,
                settings=settings,
                session_factory=session_factory,
                embedding_fn=self._embed_fn,
            )
        else:
            self._embed_fn = None
            self._store = None

    async def retrieve(  # noqa: PLR0913
        self,
        query: str | RetrievalRequest,
        n_results: int = 5,
        distance_threshold: float | None = None,
        strict: bool = False,
        candidate_k: int | None = None,
        **filters: Any,
    ) -> list[SearchResult]:
        """Retrieve documents matching ``query``.

        Args:
            query: The natural-language search query or a typed
                :class:`~ragit.retrieval.types.RetrievalRequest`.
            n_results: Maximum number of candidates to fetch.
            distance_threshold: Maximum L2 distance for the vector
                search part. When ``None``, falls back to
                ``settings.rag_distance_threshold``.
            strict: When ``True``, store failures raise
                :class:`ragit.types.RetrievalError`.
            candidate_k: Optional candidate count for initial stages.
            **filters: Additional equality filters.

        Returns:
            Search results sorted by RRF score.
        """
        t0 = time.perf_counter()
        if isinstance(query, RetrievalRequest):
            req = query
            query_str = req.query
            n_results = req.final_k
            candidate_k = req.candidate_k
            if req.threshold is not None:
                distance_threshold = req.threshold
            strict = req.strict
            merged_filters = dict(req.filters)
            if req.isolation_scope is not None:
                merged_filters[req.scope_field] = req.isolation_scope
            merged_filters.update(filters)
            filters = merged_filters
        else:
            query_str = query

        if distance_threshold is None:
            distance_threshold = self._settings.rag_distance_threshold

        if self._strategy is not None:
            kwargs = dict(filters)
            if candidate_k is not None:
                kwargs["candidate_k"] = candidate_k
            rows = await self._strategy.retrieve(
                query=query_str,
                n_results=n_results,
                threshold=distance_threshold,
                text_field=self._spec.text_field,
                metadata_fields=self._spec.metadata_fields,
                strict=strict,
                **kwargs,
            )
        else:
            effective_n = (
                candidate_k if candidate_k is not None and candidate_k > n_results else n_results
            )
            embeddings = await self._embed_fn([query_str])
            query_embedding = embeddings[0]

            rows = await self._store.hybrid_search(
                query=query_str,
                embedding=query_embedding,
                n_results=effective_n,
                threshold=distance_threshold,
                text_field=self._spec.text_field,
                metadata_fields=self._spec.metadata_fields,
                strict=strict,
                **filters,
            )
            rows = rows[:n_results]

        duration_ms = (time.perf_counter() - t0) * 1000.0
        results = [_to_search_result(row) for row in rows]
        self._event_sink.emit(
            TelemetryEvent(
                event_kind=EventKind.SEARCH_COMPLETED,
                stage="retrieval",
                duration_ms=duration_ms,
                counts={
                    "candidate_k": candidate_k if candidate_k is not None else n_results,
                    "final_k": n_results,
                    "result_count": len(results),
                },
            )
        )
        return results

    async def retrieve_typed(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Execute retrieval and return a typed :class:`RetrievalResponse`."""
        if self._strategy is not None:
            if hasattr(self._strategy, "retrieve_typed"):
                return await self._strategy.retrieve_typed(request)
            rows = await self._strategy.retrieve(request)
            return RetrievalResponse.from_dicts(
                rows,
                query=request.query,
                candidate_k=request.get_candidate_k(),
                final_k=request.final_k,
            )
        results = await self.retrieve(request)
        return RetrievalResponse.from_dicts(
            as_dicts(results),
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )
