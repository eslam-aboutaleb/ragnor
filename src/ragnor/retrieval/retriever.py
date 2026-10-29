"""Generic hybrid retrieval orchestrator.

The orchestrator is table-agnostic: the caller
supplies a :class:`RetrieverSpec` naming the table
and the metadata columns to surface, so no table
name is hardcoded anywhere in ragnor.

The retriever embeds the query, delegates the
two-stage search (vector similarity + keyword
search, merged with Reciprocal Rank Fusion) to the
injected vector store, and returns
:class:`ragnor.core.types.SearchResult` dataclasses.
:func:`as_dicts` converts them back to the dict
shape the stores return, so existing dict consumers
keep working during the transition.

A retrieval strategy (see
:mod:`ragnor.retrieval.strategies`) can be
injected instead: when a strategy is supplied, the
constructor builds nothing -- no embedder, no store
-- and retrieval delegates to the strategy, which
owns its capability dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ragnor.core.config import RagSettings
from ragnor.embeddings import get_embedding_dimension, get_embedding_function
from ragnor.retrieval.types import RetrievalRequest, RetrievalResponse
from ragnor.retrieval.capabilities import get_backend_capabilities
from ragnor.retrieval.strategies import (
    _coerce_request,
    _validate_scope_defense_in_depth,
    _query_vector,
)
from ragnor.stores import get_vector_store
from ragnor.telemetry.events import EventKind, NullEventSink, TelemetryEvent
from ragnor.core.types import SearchResult
from ragnor.core.validation import validate_identifier

if TYPE_CHECKING:
    from ragnor.db.session import SessionProvider

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
    :class:`ragnor.stores.base.VectorIndex` plus an
    external lexical engine adapted to
    :class:`ragnor.stores.base.LexicalIndex`, or a
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
                SQL stores require this host-owned provider when
                called without a session.
            strategy: Optional retrieval strategy (see
                :mod:`ragnor.retrieval.strategies`). When
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
                profile = getattr(self._embed_fn, "profile", None)
                embedding_dim = (
                    profile.dimension
                    if profile is not None
                    else get_embedding_dimension(settings.embedding_model)
                )
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
                :class:`~ragnor.retrieval.types.RetrievalRequest`.
            n_results: Maximum number of candidates to fetch.
            distance_threshold: Maximum L2 distance for the vector
                search part. When ``None``, falls back to
                ``settings.rag_distance_threshold``.
            strict: When ``True``, store failures raise
                :class:`ragnor.core.types.RetrievalError`.
            candidate_k: Optional candidate count for initial stages.
            **filters: Additional equality filters.

        Returns:
            Search results sorted by RRF score.
        """
        t0 = time.perf_counter()
        if isinstance(query, RetrievalRequest):
            req = replace(query, filters={**query.filters, **filters})
        else:
            req, _ = _coerce_request(
                query,
                n_results,
                {
                    **filters,
                    "candidate_k": candidate_k,
                    "threshold": distance_threshold,
                    "strict": strict,
                    "text_field": self._spec.text_field,
                    "metadata_fields": self._spec.metadata_fields,
                },
            )
        rows = await self._retrieve_rows(req)
        n_results = req.final_k
        candidate_k = req.get_candidate_k()

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

    async def _retrieve_rows(self, request: RetrievalRequest) -> list[dict[str, Any]]:
        fields = list(request.metadata_fields)
        if request.isolation_scope is not None:
            fields.append(request.scope_field)
        if request.expected_generation is not None:
            fields.append("generation")
        req = replace(
            request,
            threshold=(
                request.threshold
                if request.threshold is not None
                else self._settings.rag_distance_threshold
            ),
            metadata_fields=list(dict.fromkeys(fields)),
        )
        if self._strategy is not None:
            if hasattr(self._strategy, "retrieve_typed"):
                response = await self._strategy.retrieve_typed(req)
                rows = response.to_dicts()
            else:
                caps = get_backend_capabilities(self._strategy)
                caps.validate_constraints(req.mandatory_capabilities)
                caps.validate_filters(req.effective_filters())
                rows = await self._strategy.retrieve(
                    query=req.query,
                    n_results=req.final_k,
                    candidate_k=req.get_candidate_k(),
                    threshold=req.threshold,
                    text_field=req.text_field,
                    metadata_fields=req.metadata_fields,
                    session=req.session,
                    strict=req.strict,
                    **req.effective_filters(),
                )
        else:
            caps = get_backend_capabilities(self._store)
            caps.validate_constraints(req.mandatory_capabilities)
            caps.validate_filters(req.effective_filters())
            query_vector = await _query_vector(self._embed_fn, req.query)
            rows = await self._store.hybrid_search(
                query=req.query,
                embedding=query_vector,
                n_results=req.get_candidate_k(),
                threshold=req.threshold,
                text_field=req.text_field,
                metadata_fields=req.metadata_fields,
                session=req.session,
                strict=req.strict,
                **req.effective_filters(),
            )
        return _validate_scope_defense_in_depth(rows, req)[: req.final_k]

    async def retrieve_typed(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute the same constrained path while preserving reranker scores."""
        rows = await self._retrieve_rows(request)
        return RetrievalResponse.from_dicts(
            rows,
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )
