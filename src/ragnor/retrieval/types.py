"""Typed request and result wrappers for retrieval operations.

Provides typed objects (:class:`RetrievalRequest`,
:class:`RetrievedDocument`, :class:`RetrievalResponse`) that bridge
cleanly to and from the historical ``dict[str, Any]`` shape returned
by the storage backends.

Score semantics are preserved explicitly via :class:`~ragnor.retrieval.scores.RawScore`
and :class:`~ragnor.retrieval.scores.ScoreKind`: distance, similarity, RRF,
and reranker scores are kept distinct and never conflated as probabilities.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragnor.core.exceptions import ScopeDeniedError
from ragnor.retrieval.scores import RawScore, ScoreKind
from ragnor.core.types import RetrievalError, SearchResult


class ScopeViolationError(ScopeDeniedError, RetrievalError):
    """Raised when a retrieved document fails isolation scope validation.

    Defense-in-depth: if a storage backend returns a document belonging
    to a different tenant, partition, or collection than requested,
    retrieval fails closed (or drops the item) to prevent cross-tenant
    evidence leakage.
    """

    default_code = "ERR_SCOPE_VIOLATION"


@dataclass
class RetrievalRequest:
    """A typed request for document retrieval.

    Attributes:
        query: Natural language query string.
        final_k: Final number of top results to return to caller.
        candidate_k: Number of candidates to retrieve in initial dense
            and lexical stages. When ``None``, defaults to ``final_k``.
            Exposing ``candidate_k`` separately ensures an optional
            reranker stage has enough candidate documents to re-score.
        threshold: Distance cutoff for vector search.
        filters: Key-value equality or operator filters to push down.
        mandatory_capabilities: Capability or operator names that the
            storage backend MUST support; fails closed if unsupported.
        isolation_scope: Tenant, namespace, or collection identifier.
            Pushed to backend filter stage and re-validated on results.
        scope_field: Metadata key used for isolation scope (defaults
            to ``"tenant_id"``).
        expected_generation: Target generation integer for active index
            generation scoping.
        text_field: Column / field name for the document text.
        metadata_fields: Additional metadata column names to surface.
        session: Optional external transaction/session handle.
        strict: When ``True``, storage failures raise :class:`RetrievalError`.
    """

    query: str
    final_k: int = 5
    candidate_k: int | None = None
    threshold: float | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    mandatory_capabilities: set[str] = field(default_factory=set)
    isolation_scope: str | None = None
    scope_field: str = "tenant_id"
    expected_generation: int | None = None
    text_field: str = "text"
    metadata_fields: list[str] = field(default_factory=list)
    session: Any = None
    strict: bool = False

    def get_candidate_k(self) -> int:
        """Return the effective candidate count for initial retrieval stages."""
        if self.candidate_k is not None and self.candidate_k > 0:
            return max(self.candidate_k, self.final_k)
        return self.final_k

    def effective_filters(self) -> dict[str, Any]:
        """Return merged filters including isolation scope and generation."""
        merged = dict(self.filters)
        if self.isolation_scope is not None:
            merged[self.scope_field] = self.isolation_scope
        if self.expected_generation is not None:
            merged["generation"] = self.expected_generation
        return merged

    @classmethod
    def from_query(  # noqa: PLR0913
        cls,
        query: str,
        n_results: int = 5,
        *,
        candidate_k: int | None = None,
        threshold: float | None = None,
        strict: bool = False,
        isolation_scope: str | None = None,
        scope_field: str = "tenant_id",
        mandatory_capabilities: set[str] | None = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> RetrievalRequest:
        """Create a request from query string and keyword arguments."""
        return cls(
            query=query,
            final_k=n_results,
            candidate_k=candidate_k,
            threshold=threshold,
            filters=filters,
            mandatory_capabilities=mandatory_capabilities or set(),
            isolation_scope=isolation_scope,
            scope_field=scope_field,
            text_field=text_field,
            metadata_fields=metadata_fields or [],
            session=session,
            strict=strict,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert request to legacy kwargs dictionary."""
        d: dict[str, Any] = {
            "query": self.query,
            "n_results": self.final_k,
            "final_k": self.final_k,
            "candidate_k": self.candidate_k,
            "threshold": self.threshold,
            "text_field": self.text_field,
            "metadata_fields": self.metadata_fields,
            "strict": self.strict,
            "isolation_scope": self.isolation_scope,
            "scope_field": self.scope_field,
            "session": self.session,
            **self.filters,
        }
        return d


@dataclass
class RetrievedDocument:
    """A typed retrieval result with explicit score tracking.

    Attributes:
        id: Unique identifier of the document (or ``None``).
        document: The text content of the retrieved document/chunk.
        metadata: Associated metadata dictionary.
        scores: Explicit dictionary of scores keyed by :class:`ScoreKind`.
        distance: Metric distance from vector search (lower is better).
        keyword_score: Score from keyword/lexical search (higher is better).
        rrf_score: Reciprocal Rank Fusion score (higher is better).
        reranker_score: Score from the reranker stage (higher is better).
    """

    id: str | None
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    scores: dict[ScoreKind, RawScore] = field(default_factory=dict)
    distance: float | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0
    reranker_score: float | None = None

    def __post_init__(self) -> None:
        """Populate scores dictionary from scalar fields if empty."""
        if self.distance is not None and ScoreKind.DISTANCE not in self.scores:
            self.scores[ScoreKind.DISTANCE] = RawScore.distance(self.distance)
        if self.keyword_score is not None and ScoreKind.LEXICAL not in self.scores:
            self.scores[ScoreKind.LEXICAL] = RawScore.lexical(self.keyword_score)
        if self.rrf_score > 0.0 and ScoreKind.RRF not in self.scores:
            self.scores[ScoreKind.RRF] = RawScore.rrf(self.rrf_score)
        if self.reranker_score is not None and ScoreKind.RERANKER not in self.scores:
            self.scores[ScoreKind.RERANKER] = RawScore.reranker(self.reranker_score)

    def to_dict(self) -> dict[str, Any]:
        """Convert to legacy store result dictionary format."""
        d: dict[str, Any] = {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
            "distance": self.distance,
            "keyword_score": self.keyword_score,
            "_rrf_score": self.rrf_score,
        }
        if self.reranker_score is not None:
            d["_reranker_score"] = self.reranker_score
        return d

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RetrievedDocument:
        """Construct from a legacy store result dictionary."""
        distance = row.get("distance")
        keyword_score = row.get("keyword_score")
        rrf_score = row.get("_rrf_score", 0.0)
        reranker_score = row.get("_reranker_score")

        scores: dict[ScoreKind, RawScore] = {}
        if distance is not None:
            scores[ScoreKind.DISTANCE] = RawScore.distance(float(distance))
        if keyword_score is not None:
            scores[ScoreKind.LEXICAL] = RawScore.lexical(float(keyword_score))
        if rrf_score > 0.0:
            scores[ScoreKind.RRF] = RawScore.rrf(float(rrf_score))
        if reranker_score is not None:
            scores[ScoreKind.RERANKER] = RawScore.reranker(float(reranker_score))

        return cls(
            id=str(row["id"]) if row.get("id") is not None else None,
            document=str(row.get("document", "")),
            metadata=dict(row.get("metadata") or {}),
            scores=scores,
            distance=float(distance) if distance is not None else None,
            keyword_score=float(keyword_score) if keyword_score is not None else None,
            rrf_score=float(rrf_score),
            reranker_score=float(reranker_score) if reranker_score is not None else None,
        )

    def to_search_result(self) -> SearchResult:
        """Convert to legacy SearchResult dataclass."""
        return SearchResult(
            id=self.id,
            document=self.document,
            metadata=self.metadata,
            distance=self.distance,
            keyword_score=self.keyword_score,
            rrf_score=self.rrf_score,
        )


@dataclass
class RetrievalResponse:
    """A typed container of retrieved documents with metadata.

    Attributes:
        items: List of retrieved documents ordered best-first.
        query: Query string that produced these results.
        candidate_k: Number of candidate documents fetched initially.
        final_k: Number of final results requested.
    """

    items: list[RetrievedDocument]
    query: str
    candidate_k: int = 5
    final_k: int = 5

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert all items to legacy result dictionaries."""
        return [item.to_dict() for item in self.items]

    @classmethod
    def from_dicts(
        cls,
        rows: Sequence[Mapping[str, Any]],
        query: str,
        *,
        candidate_k: int = 5,
        final_k: int = 5,
    ) -> RetrievalResponse:
        """Construct from a sequence of legacy result dicts."""
        return cls(
            items=[RetrievedDocument.from_dict(row) for row in rows],
            query=query,
            candidate_k=candidate_k,
            final_k=final_k,
        )

    def __iter__(self) -> Iterator[RetrievedDocument]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RetrievedDocument:
        return self.items[index]
