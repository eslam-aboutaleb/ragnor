"""In-memory vector store for tests and zero-dependency development.

Implements the :class:`VectorStore` interface entirely in process
memory: brute-force vector search, a token-overlap approximation
of full-text search, and the shared Reciprocal Rank Fusion merge
(:mod:`ragit.stores._scoring`).

The vector stage ranks by the store's distance metric
(:mod:`ragit.metrics`): ``l2`` by default, ``cosine`` when the
store is constructed with ``metric="cosine"``.

The store subclasses :class:`ragit.stores.base.HybridVectorStore`:
the vector stage lives in :meth:`vector_search` and the keyword
stage in :meth:`keyword_search` (both capability methods that
always raise on failure), and the inherited composite
``hybrid_search`` runs both stages and fuses them with RRF.

Approximation notice: the keyword score is a token-overlap ratio,
not a real ``ts_rank`` over a ``tsvector``. It is intended for
tests and development, not as a production search backend.
"""

from __future__ import annotations

import logging
from typing import Any

from ragit.metrics import cosine_distance
from ragit.stores._scoring import l2_distance, token_overlap, tokenize
from ragit.stores.base import EmbeddingFunction, HybridVectorStore
from ragit.stores.registry import vector_store_registry
from ragit.validation import validate_embedding, validate_identifier

logger = logging.getLogger(__name__)


@vector_store_registry.register("memory")
class InMemoryVectorStore(HybridVectorStore):
    """In-memory vector store.

    Brute-force vector search over the store's distance metric
    (``l2`` by default, ``cosine`` opt-in via ``metric``) with
    token-overlap keyword matching, merged with the shared RRF
    formula (``1.0 / (60 + rank)`` per stage, 1-based ranks,
    tie-break by id). Result dicts carry the same keys as the
    other backends: ``id``, ``document``, ``metadata``,
    ``distance`` (``None`` for keyword-only hits),
    ``keyword_score`` (``None`` for vector-only hits), and
    ``_rrf_score``.
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
                the pgvector store; ignored (no database) except for
                the optional ``rag_distance_metric`` field.
            embedding_fn: Async callable that embeds a list of texts.
                Used only when ``hybrid_search`` is called with an
                empty embedding.
            metric: Distance metric for the vector stage (``"l2"``
                by default, ``"cosine"`` opt-in). When omitted, the
                host settings' optional ``rag_distance_metric``
                field is used.
        """
        super().__init__(
            table_name=table_name,
            id_field=id_field,
            embedding_dim=embedding_dim,
            session_factory=session_factory,
            settings=settings,
            embedding_fn=embedding_fn,
            metric=metric,
        )
        self._records: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}

    def _matches_filters(self, record: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Return whether ``record`` satisfies every equality filter."""
        return all(record.get(key) == value for key, value in filters.items())

    def _metadata_projection(
        self,
        record: dict[str, Any],
        metadata_fields: list[str],
    ) -> dict[str, Any]:
        """Project ``metadata_fields`` out of ``record``."""
        return {
            field: record[field]
            for field in metadata_fields
            if field in record and record[field] is not None
        }

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
        """Brute-force vector search over the record map.

        The candidates are scored with the store's distance metric:
        ``l2_distance`` for ``l2`` (the default) and
        ``cosine_distance`` for ``cosine``.

        Args:
            embedding: Query embedding vector (required; the method
                does not auto-embed).
            n_results: Maximum number of results to return.
            threshold: Maximum distance under the store's metric
                (metric-relative; never comparable across metrics).
            text_field: Name of the field containing the document text.
            metadata_fields: Fields to include in the result metadata.
            session: Accepted for interface compatibility; ignored
                (no database).
            **filters: Equality filters matched against document
                fields (metadata and extra fields).

        Returns:
            list[dict]: Result dicts ordered by distance ascending
                (tie-broken by id), each carrying ``id``, ``document``,
                ``metadata``, ``distance``, and ``keyword_score``
                (``None``).

        Raises:
            ValueError: If identifiers are unsafe or the embedding
                has the wrong dimension.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        if self.embedding_dim is not None:
            validate_embedding(embedding, self.embedding_dim, label="query embedding")

        vector_distances: dict[str, float] = {}
        for doc_id, record in self._records.items():
            if not self._matches_filters(record, filters):
                continue
            doc_embedding = self._embeddings.get(doc_id)
            if doc_embedding is not None:
                if self.metric == "cosine":
                    distance = cosine_distance(embedding, doc_embedding)
                else:
                    distance = l2_distance(embedding, doc_embedding)
                if distance < threshold:
                    vector_distances[doc_id] = distance

        vector_ranked = [
            doc_id
            for doc_id, _ in sorted(vector_distances.items(), key=lambda item: (item[1], item[0]))[
                :n_results
            ]
        ]
        results = []
        for doc_id in vector_ranked:
            record = self._records[doc_id]
            results.append(
                {
                    "id": str(doc_id) if doc_id is not None else None,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                    "distance": float(vector_distances[doc_id]),
                    "keyword_score": None,
                }
            )
        return results

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Token-overlap keyword search over the record map.

        The keyword score is a token-overlap ratio, not a real
        full-text rank -- an approximation intended for tests and
        development.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            text_field: Name of the field containing the document text.
            metadata_fields: Fields to include in the result metadata.
            session: Accepted for interface compatibility; ignored
                (no database).
            **filters: Equality filters matched against document
                fields (metadata and extra fields).

        Returns:
            list[dict]: Result dicts ordered by overlap ratio
            descending (tie-broken by id), each carrying ``id``,
            ``document``, ``metadata``, ``distance`` (``None``), and
            ``keyword_score``.

        Raises:
            ValueError: If identifiers are unsafe.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        query_tokens = set(tokenize(query))
        keyword_scores: dict[str, float] = {}
        for doc_id, record in self._records.items():
            if not self._matches_filters(record, filters):
                continue
            score = token_overlap(query_tokens, tokenize(str(record.get(text_field, ""))))
            if score > 0.0:
                keyword_scores[doc_id] = score

        keyword_ranked = [
            doc_id
            for doc_id, _ in sorted(keyword_scores.items(), key=lambda item: (-item[1], item[0]))[
                :n_results
            ]
        ]
        results = []
        for doc_id in keyword_ranked:
            record = self._records[doc_id]
            results.append(
                {
                    "id": str(doc_id) if doc_id is not None else None,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                    "distance": None,
                    "keyword_score": float(keyword_scores[doc_id]),
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

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete documents selected by ``ids`` and/or ``filters``.

        At least one selector must be supplied; an empty ``ids``
        list is rejected (an empty id list never means "delete
        everything"). When both are supplied, both restrictions
        apply (conjunction).

        Args:
            ids: Document ids to delete.
            filters: Equality filters matched against document
                fields (metadata and extra fields).
            session: Accepted for interface compatibility; ignored
                (no database).

        Returns:
            int: Number of deleted documents.

        Raises:
            ValueError: If no selector is supplied, ``ids`` is
                empty, or a filter key is unsafe.
        """
        if ids is None and filters is None:
            raise ValueError("delete requires at least one of ids or filters")
        if ids is not None and len(ids) == 0:
            raise ValueError(
                "ids must not be empty; an empty id list never means delete everything"
            )
        if filters is not None:
            for key in filters:
                validate_identifier(key, "filter key")

        deleted = 0
        for doc_id in list(self._records):
            if ids is not None and doc_id not in set(ids):
                continue
            if filters is not None and not self._matches_filters(self._records[doc_id], filters):
                continue
            del self._records[doc_id]
            self._embeddings.pop(doc_id, None)
            deleted += 1
        logger.info("Deleted %d documents from in-memory store %s.", deleted, self.table_name)
        return deleted

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``.

        With no selector, every document is returned. When both
        selectors are supplied, both restrictions apply
        (conjunction).

        Args:
            ids: Document ids to fetch.
            filters: Equality filters matched against document
                fields (metadata and extra fields).
            session: Accepted for interface compatibility; ignored
                (no database).
            text_field: Name of the field containing the document
                text. Defaults to "text".
            metadata_fields: Fields to include in the result
                metadata dict.

        Returns:
            list[dict]: ``id``/``document``/``metadata`` dicts
            (score keys are meaningless for direct fetches).

        Raises:
            ValueError: If any identifier is unsafe.
        """
        validate_identifier(text_field, "text_field")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        if filters is not None:
            for key in filters:
                validate_identifier(key, "filter key")

        results = []
        for doc_id, record in self._records.items():
            if ids is not None and doc_id not in set(ids):
                continue
            if filters is not None and not self._matches_filters(record, filters):
                continue
            results.append(
                {
                    "id": str(doc_id) if doc_id is not None else None,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                }
            )
        return results
