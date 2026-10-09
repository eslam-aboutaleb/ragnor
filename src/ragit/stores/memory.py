"""In-memory vector store for tests and zero-dependency development.

Implements the :class:`VectorStore` interface entirely in process
memory: brute-force L2 vector search, a token-overlap approximation
of full-text search, and the shared Reciprocal Rank Fusion merge
(:mod:`ragit.stores._scoring`).

Approximation notice: the keyword score is a token-overlap ratio,
not a real ``ts_rank`` over a ``tsvector``. It is intended for
tests and development, not as a production search backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ragit.config import RagSettings
from ragit.stores._scoring import l2_distance, rrf_merge, token_overlap, tokenize
from ragit.stores.base import EmbeddingFunction, VectorStore
from ragit.stores.registry import vector_store_registry
from ragit.validation import validate_embedding, validate_identifier

if TYPE_CHECKING:
    from ragit.db.session import SessionProvider

logger = logging.getLogger(__name__)


@vector_store_registry.register("memory")
class InMemoryVectorStore(VectorStore):
    """In-memory vector store.

    Brute-force L2 vector search with token-overlap keyword matching,
    merged with the shared RRF formula
    (``1.0 / (60 + rank)`` per stage, 1-based ranks, tie-break by
    id). Result dicts carry the same keys as the other backends:
    ``id``, ``document``, ``metadata``, ``distance`` (``None`` for
    keyword-only hits), ``keyword_score`` (``None`` for vector-only
    hits), and ``_rrf_score``.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str = "documents",
        id_field: str = "id",
        embedding_dim: int | None = None,
        session_factory: SessionProvider | None = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
    ):
        """Initialize the in-memory store.

        Args:
            table_name: Logical table name (used for error messages
                and identifier validation only; nothing is persisted).
            id_field: Name of the primary key field.
            embedding_dim: Expected embedding dimensionality. When
                provided, embeddings are validated like the pgvector
                store does.
            session_factory: Accepted for constructor compatibility
                with the pgvector store; ignored (no database).
            settings: Accepted for constructor compatibility with
                the pgvector store; ignored (no database).
            embedding_fn: Async callable that embeds a list of texts.
                Used only when ``hybrid_search`` is called with an
                empty embedding.
        """
        validate_identifier(table_name, "table_name")
        validate_identifier(id_field, "id_field")
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn
        self._records: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}

    def _matches_filters(self, record: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Return whether ``record`` satisfies every equality filter."""
        return all(record.get(key) == value for key, value in filters.items())

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
        """Hybrid search: brute-force L2 plus token-overlap keyword match.

        The two stages are ranked independently (distance ascending,
        overlap ratio descending, each tie-broken by id), truncated to
        ``n_results``, and merged with Reciprocal Rank Fusion — the
        same shape as the other backends' two-stage search.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty, the query is
                embedded automatically using the injected
                ``embedding_fn``.
            n_results: Maximum number of results to return.
            threshold: Maximum L2 distance for the vector search part.
            text_field: Name of the field containing the document text.
            metadata_fields: Fields to include in the result metadata.
            session: Accepted for interface compatibility; ignored
                (no database).
            strict: Accepted for interface compatibility; this store
                has no database failures to surface.
            **filters: Equality filters matched against document
                fields (metadata and extra fields).

            Returns:
                list[dict]: Result dicts sorted by RRF score, with the
                same keys as the other backends' hybrid search.
            """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

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

        query_tokens = set(tokenize(query))

        vector_distances: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        for doc_id, record in self._records.items():
            if not self._matches_filters(record, filters):
                continue
            doc_embedding = self._embeddings.get(doc_id)
            if doc_embedding is not None:
                distance = l2_distance(embedding, doc_embedding)
                if distance < threshold:
                    vector_distances[doc_id] = distance
            score = token_overlap(query_tokens, tokenize(str(record.get(text_field, ""))))
            if score > 0.0:
                keyword_scores[doc_id] = score

        vector_ranked = [
            doc_id
            for doc_id, _ in sorted(
                vector_distances.items(), key=lambda item: (item[1], item[0])
            )[:n_results]
        ]
        keyword_ranked = [
            doc_id
            for doc_id, _ in sorted(
                keyword_scores.items(), key=lambda item: (-item[1], item[0])
            )[:n_results]
        ]

        results = []
        for doc_id, rrf_score in rrf_merge(vector_ranked, keyword_ranked, n_results):
            record = self._records[doc_id]
            metadata = {
                field: record[field]
                for field in metadata_fields
                if field in record and record[field] is not None
            }
            results.append(
                {
                    "id": str(doc_id) if doc_id is not None else None,
                    "document": record.get(text_field),
                    "metadata": metadata,
                    "distance": (
                        float(vector_distances[doc_id]) if doc_id in vector_distances else None
                    ),
                    "keyword_score": (
                        float(keyword_scores[doc_id]) if doc_id in keyword_scores else None
                    ),
                    "_rrf_score": rrf_score,
                }
            )
        return results

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the store.

        Args:
            session: Accepted for interface compatibility; ignored
                (no database).
            strict: Accepted for interface compatibility; this store
                has no database failures to surface.
            **filters: Equality filters matched against document
                fields (metadata and extra fields).

        Returns:
            int: Number of matching documents.
        """
        for key in filters:
            validate_identifier(key, "filter key")
        return sum(1 for record in self._records.values() if self._matches_filters(record, filters))

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

        The in-memory equivalent of ``ON CONFLICT (id) DO UPDATE``:
        a document whose id already exists is replaced wholesale.

        Args:
            documents: List of document dicts with keys: id, text,
                metadata, embedding.
            session: Accepted for interface compatibility; ignored
                (no database).
            text_field: Name of the text field. Defaults to "text".
            embedding_field: Name of the embedding field. Defaults
                to "embedding".
            id_field: Name of the primary key field. Defaults to the
                store's configured ``id_field``.
            extra_fields: Additional static fields to include in every
                document (e.g., ``{"tenant_id": "..."}``).

        Raises:
            ValueError: If any field name contains unsafe characters.
        """
        validate_identifier(text_field, "text_field")
        validate_identifier(embedding_field, "embedding_field")
        if id_field is None:
            id_field = self.id_field
        validate_identifier(id_field, "id_field")
        if extra_fields:
            for key in extra_fields:
                validate_identifier(key, "extra_field")

        if not documents:
            return

        for doc in documents:
            for key in doc.get("metadata", {}):
                validate_identifier(key, "metadata key")

        for doc in documents:
            if self.embedding_dim is not None:
                validate_embedding(
                    doc["embedding"],
                    self.embedding_dim,
                    label="document embedding",
                )

        for doc in documents:
            doc_id = doc.get("id")
            record: dict[str, Any] = {
                id_field: doc_id,
                text_field: doc[text_field],
            }
            record.update(doc.get("metadata", {}))
            if extra_fields:
                record.update(extra_fields)
            self._records[doc_id] = record
            self._embeddings[doc_id] = doc[embedding_field]
        logger.info("Upserted %d documents to in-memory store %s.", len(documents), self.table_name)
