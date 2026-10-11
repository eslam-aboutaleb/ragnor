"""FAISS implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, lifecycle operations, and upsert operations
against an in-process FAISS index.

The store subclasses :class:`ragit.stores.base.HybridVectorStore`:
the vector stage lives in :meth:`vector_search` (an exact
FAISS L2 search) and the keyword stage in
:meth:`keyword_search` (an in-process token-overlap scan
over the record map, because FAISS has no text search).
The inherited composite ``hybrid_search`` runs both stages
and fuses them with RRF, so the result shape matches the
other backends.

Keyword-search performance notice: the keyword stage is an
in-process fallback that scans every document in the
record map and scores token overlap. It is intended for
local development, small corpora, and testing -- not for
production-scale keyword retrieval. Hosts at scale combine
a pgvector :class:`VectorIndex` with an external lexical
engine adapted to :class:`ragit.stores.base.LexicalIndex`.

FAISS is a vector index library, not a database: the
store keeps a ``faiss.IndexIDMap`` over
``IndexFlatL2`` for exact L2 search, plus an
in-process record map holding each document's text and
metadata.

Distance metric: the vector stage ranks by the store's
metric (:mod:`ragit.metrics`). The ``l2`` default keeps
``IndexFlatL2`` on the raw vectors. For ``cosine``, every
embedding (documents at upsert time, the query at search
time) is normalized to unit length first and the same
``IndexFlatL2`` index is kept: for unit vectors
``l2_squared = 2 - 2 * cosine_similarity``, so the reported
cosine distance is ``l2_squared / 2.0`` -- exactly
``1 - cosine_similarity``. Cosine ranking therefore matches
the in-memory store's cosine path.

FAISS identifies vectors by 64-bit integers, so string
document ids are mapped to consecutive integers (and
back) inside the store; the original ids are returned
in results.

Configuration (no host imports): the store receives
everything through constructor arguments. The FAISS
index is created lazily on first use, or injected
directly via ``index`` for hosts and tests that manage
their own.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ragit.metrics import normalize
from ragit.stores._scoring import token_overlap, tokenize
from ragit.stores.base import (
    EmbeddingFunction,
    HybridVectorStore,
)
from ragit.stores.registry import vector_store_registry
from ragit.validation import validate_embedding, validate_identifier

logger = logging.getLogger(__name__)


@vector_store_registry.register("faiss")
class FaissStore(HybridVectorStore):
    """FAISS vector store.

    Hybrid search combines an exact FAISS L2 search
    with in-process token-overlap keyword matching,
    merged with the shared Reciprocal Rank Fusion
    formula.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str = "documents",
        id_field: str = "id",
        embedding_dim: int | None = None,
        embedding_fn: EmbeddingFunction | None = None,
        index: Any | None = None,
        settings: Any | None = None,
        session_factory: Any | None = None,
        metric: str | None = None,
    ):
        """Initialize the FAISS store.

        Args:
            table_name: Logical name for the index (used
                for error messages and logging only;
                nothing is persisted).
            id_field: Name of the primary key field.
            embedding_dim: Dimensionality of the embedding
                vectors. Required; pass the dimension
                configured for the active embedding model.
            embedding_fn: Async callable that embeds a
                list of texts, used to auto-embed queries.
            index: Pre-built FAISS index (an
                ``faiss.IndexIDMap`` or any index with
                ``add_with_ids``, ``remove_ids``, and
                ``search``). When omitted, one is created
                lazily.
            settings: Accepted for constructor
                compatibility with the other stores;
                ignored (FAISS is in-process) except for the
                optional ``rag_distance_metric`` field.
            session_factory: Accepted for constructor
                compatibility with the other stores;
                ignored (FAISS is in-process).
            metric: Distance metric for the vector stage
                (``"l2"`` by default, ``"cosine"`` opt-in).
                When omitted, the host settings' optional
                ``rag_distance_metric`` field is used.

        Raises:
            ValueError: If ``table_name`` or ``id_field``
                contain unsafe characters, or
                ``embedding_dim`` is omitted.
        """
        super().__init__(
            table_name=table_name,
            id_field=id_field,
            embedding_dim=embedding_dim,
            embedding_fn=embedding_fn,
            settings=settings,
            session_factory=session_factory,
            metric=metric,
        )
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self._index = index
        self._string_to_int: dict[str, int] = {}
        self._int_to_string: dict[int, str] = {}
        self._next_int = 0
        self._records: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}

    def _resolve_index(self) -> Any:
        """Return the FAISS index, creating it on first use.

        Raises:
            RuntimeError: If no index was injected and
                ``faiss`` is not installed.
        """
        if self._index is None:
            import faiss  # noqa: PLC0415

            self._index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        return self._index

    def _int_id(self, doc_id: str) -> int:
        """Return the FAISS integer id for a document id."""
        if doc_id not in self._string_to_int:
            int_id = self._next_int
            self._next_int += 1
            self._string_to_int[doc_id] = int_id
            self._int_to_string[int_id] = doc_id
        return self._string_to_int[doc_id]

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
        """Exact L2 search over the FAISS index.

        FAISS returns ``-1`` labels for slots beyond the
        index size. The search over-fetches (10×) because
        FAISS cannot filter natively: candidates that do
        not match the filters are dropped in process, so
        matching documents deeper in the ranking are still
        found.

        Under the ``cosine`` metric the query and the
        stored vectors are normalized to unit length, so the
        FAISS L2 distance over the index equals
        ``2 - 2 * cosine_similarity`` and the reported
        distance is half of it (``l2_squared / 2.0`` --
        exactly ``1 - cosine_similarity``).

        Args:
            embedding: Query embedding vector (required;
                the method does not auto-embed).
            n_results: Maximum number of results to return.
            threshold: Maximum distance under the store's
                metric (metric-relative; never comparable
                across metrics).
            text_field: Record key containing the document
                text.
            metadata_fields: Record keys to include in the
                result metadata dict.
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            **filters: Equality filters matched against
                document fields (metadata and extra
                fields). FAISS cannot filter natively, so
                the vector search over-fetches candidates
                and filters them in process.

        Returns:
            list[dict]: Result dicts ordered by distance ascending,
            each carrying ``id``, ``document``, ``metadata``,
            ``distance``, and ``keyword_score`` (``None``).

        Raises:
            ValueError: If identifiers are unsafe or the embedding
                has the wrong dimension.
            Exception: Re-raises FAISS errors.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        validate_embedding(embedding, self.embedding_dim, label="query embedding")

        index = self._resolve_index()

        search_limit = n_results * 10
        query_embedding = normalize(embedding) if self.metric == "cosine" else embedding
        query_vector = np.asarray([query_embedding], dtype="float32")
        faiss_distances, faiss_labels = await asyncio.to_thread(
            index.search, query_vector, search_limit
        )
        vector_ranked: list[str] = []
        distances: dict[str, float] = {}
        for distance, label in zip(faiss_distances[0], faiss_labels[0], strict=False):
            if int(label) < 0:
                continue
            doc_id = self._int_to_string.get(int(label))
            if doc_id is None:
                continue
            record = self._records.get(doc_id, {})
            if not self._matches_filters(record, filters):
                continue
            reported = float(distance)
            if self.metric == "cosine":
                reported /= 2.0
            if reported >= threshold:
                continue
            vector_ranked.append(doc_id)
            distances[doc_id] = reported
        vector_ranked = vector_ranked[:n_results]

        results = []
        for doc_id in vector_ranked:
            record = self._records.get(doc_id, {})
            results.append(
                {
                    "id": doc_id,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                    "distance": distances[doc_id],
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

        FAISS has no text search, so the keyword stage
        scores token overlap over every document in the
        in-process record map -- an in-process fallback
        intended for local development, small corpora,
        and testing, not production-scale keyword
        retrieval.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            text_field: Record key containing the document
                text.
            metadata_fields: Record keys to include in the
                result metadata dict.
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            **filters: Equality filters matched against
                document fields (metadata and extra
                fields).

        Returns:
            list[dict]: Result dicts ordered by keyword score
            descending (tie-broken by id), each carrying ``id``,
            ``document``, ``metadata``, ``distance`` (``None``),
            and ``keyword_score``.

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

        keyword_ranked = sorted(
            keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
        )[:n_results]

        results = []
        for doc_id in keyword_ranked:
            record = self._records.get(doc_id, {})
            results.append(
                {
                    "id": doc_id,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                    "distance": None,
                    "keyword_score": keyword_scores[doc_id],
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
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            strict: Accepted for interface compatibility;
                this store has no external failures to
                surface.
            **filters: Equality filters matched against
                document fields (metadata and extra
                fields).

        Returns:
            int: Number of matching documents.
        """
        for key in filters:
            validate_identifier(key, "filter key")
        return sum(1 for record in self._records.values() if self._matches_filters(record, filters))

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete documents selected by ``ids`` and/or ``filters``.

        At least one selector must be supplied; an empty ``ids``
        list is rejected. When both are supplied, both
        restrictions apply (conjunction). The selected
        documents' vectors are removed from the FAISS index
        (``remove_ids``) and their record-map entries are
        cleaned up.

        Args:
            ids: Document ids to delete.
            filters: Equality filters matched against document
                fields (metadata and extra fields).
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).

        Returns:
            int: Number of deleted documents.

        Raises:
            ValueError: If no selector is supplied, ``ids`` is
                empty, or a filter key is unsafe.
            Exception: Re-raises FAISS errors.
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

        to_delete = [
            doc_id
            for doc_id in list(self._records)
            if (ids is None or doc_id in set(ids))
            and (filters is None or self._matches_filters(self._records[doc_id], filters))
        ]
        if not to_delete:
            return 0

        index = self._resolve_index()
        int_ids = np.asarray(
            [self._string_to_int[doc_id] for doc_id in to_delete],
            dtype="int64",
        )
        await asyncio.to_thread(index.remove_ids, int_ids)
        for doc_id in to_delete:
            del self._records[doc_id]
            self._embeddings.pop(doc_id, None)
            int_id = self._string_to_int.pop(doc_id, None)
            if int_id is not None:
                self._int_to_string.pop(int_id, None)
        logger.info("Deleted %d documents from %s.", len(to_delete), self.table_name)
        return len(to_delete)

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``.

        The selection is a record-map lookup: documents whose
        id is in ``ids`` (when supplied) and that match every
        equality filter (when supplied). With no selector,
        every document is returned.

        Args:
            ids: Document ids to fetch.
            filters: Equality filters matched against document
                fields (metadata and extra fields).
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            text_field: Record key containing the document
                text. Defaults to "text".
            metadata_fields: Record keys to include in the
                result metadata dict.

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
                    "id": doc_id,
                    "document": record.get(text_field),
                    "metadata": self._metadata_projection(record, metadata_fields),
                }
            )
        return results

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

        A document whose id already exists is replaced:
        its old vector is removed from the FAISS index
        and the new vector is added under the same
        integer id.

        Under the ``cosine`` metric the stored vectors are
        normalized to unit length; see the module docstring
        for the ``l2_squared / 2.0`` distance conversion.

        Args:
            documents: List of document dicts with keys:
                id, text, metadata, embedding.
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            text_field: Record key for the document text.
                Defaults to "text".
            embedding_field: Document dict key holding the
                embedding vector. Defaults to "embedding".
            id_field: Name of the primary key field.
                Defaults to the store's configured
                ``id_field``.
            extra_fields: Additional static fields to
                include in every document (e.g.,
                ``{"tenant_id": "..."}``).

        Raises:
            ValueError: If any field name contains unsafe
                characters.
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
            validate_embedding(
                doc["embedding"],
                self.embedding_dim,
                label="document embedding",
            )
            for key in doc.get("metadata", {}):
                validate_identifier(key, "metadata key")

        index = self._resolve_index()

        # Replace existing vectors: FAISS has no update,
        # so an existing id is removed before re-adding.
        existing = [doc.get("id") for doc in documents if doc.get("id") in self._string_to_int]
        if existing:
            int_ids = np.asarray(
                [self._string_to_int[doc_id] for doc_id in existing],
                dtype="int64",
            )
            await asyncio.to_thread(index.remove_ids, int_ids)

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

        stored_vectors = [doc[embedding_field] for doc in documents]
        if self.metric == "cosine":
            stored_vectors = [normalize(vector) for vector in stored_vectors]
        vectors = np.asarray(stored_vectors, dtype="float32")
        int_ids = np.asarray([self._int_id(doc.get("id")) for doc in documents], dtype="int64")
        await asyncio.to_thread(index.add_with_ids, vectors, int_ids)
        logger.info("Upserted %d documents to %s.", len(documents), self.table_name)
