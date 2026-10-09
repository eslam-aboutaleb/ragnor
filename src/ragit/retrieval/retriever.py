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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragit.config import RagSettings
from ragit.embeddings import get_embedding_dimension, get_embedding_function
from ragit.stores import get_vector_store
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
    """Hybrid retrieval over an arbitrary document table."""

    def __init__(
        self,
        spec: RetrieverSpec,
        settings: RagSettings,
        session_factory: SessionProvider | None = None,
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
        """
        self._spec = spec
        self._settings = settings
        self._session_factory = session_factory
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

    async def retrieve(  # noqa: PLR0913
        self,
        query: str,
        n_results: int = 5,
        distance_threshold: float | None = None,
        **filters: Any,
    ) -> list[SearchResult]:
        """Retrieve documents matching ``query``.

        Orchestrates the two-stage retrieval pipeline:
          1. Embeds the query using the configured embedding function.
          2. Delegates to the vector store's ``hybrid_search`` method,
             which executes parallel vector similarity and PostgreSQL
             full-text search CTEs and merges results via Reciprocal
             Rank Fusion.

        Args:
            query: The natural-language search query.
            n_results: Maximum number of candidates to fetch.
            distance_threshold: Maximum L2 distance for the vector
                search part. When ``None``, falls back to
                ``settings.rag_distance_threshold``.
            **filters: Additional equality filters applied to both
                search stages (e.g., ``tenant_id="..."``).

        Returns:
            Search results sorted by RRF score.
        """
        if distance_threshold is None:
            distance_threshold = self._settings.rag_distance_threshold

        embeddings = await self._embed_fn([query])
        query_embedding = embeddings[0]

        rows = await self._store.hybrid_search(
            query=query,
            embedding=query_embedding,
            n_results=n_results,
            threshold=distance_threshold,
            text_field=self._spec.text_field,
            metadata_fields=self._spec.metadata_fields,
            **filters,
        )
        return [_to_search_result(row) for row in rows]
