"""FAISS implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, and upsert operations against an in-process
FAISS index.

FAISS is a vector index library, not a database: the
store keeps a ``faiss.IndexIDMap`` over
``IndexFlatL2`` for exact L2 search, plus an
in-process record map holding each document's text and
metadata. The keyword stage scores token overlap over
that record map (the same approximation the in-memory
store uses), because FAISS has no text search. Both
stages are ranked independently and merged with RRF,
so the result shape matches the other backends.

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

from ragit.stores._scoring import rrf_merge, token_overlap, tokenize
from ragit.stores.base import EmbeddingFunction, VectorStore
from ragit.stores.registry import vector_store_registry
from ragit.types import RetrievalError
from ragit.validation import validate_embedding, validate_identifier

logger = logging.getLogger(__name__)


@vector_store_registry.register("faiss")
class FaissStore(VectorStore):
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
                ignored (FAISS is in-process).
            session_factory: Accepted for constructor
                compatibility with the other stores;
                ignored (FAISS is in-process).

        Raises:
            ValueError: If ``table_name`` or ``id_field``
                contain unsafe characters, or
                ``embedding_dim`` is omitted.
        """
        validate_identifier(table_name, "table_name")
        validate_identifier(id_field, "id_field")
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn
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

            self._index = faiss.IndexIDMap(
                faiss.IndexFlatL2(self.embedding_dim)
            )
        return self._index

    def _int_id(self, doc_id: str) -> int:
        """Return the FAISS integer id for a document id."""
        if doc_id not in self._string_to_int:
            int_id = self._next_int
            self._next_int += 1
            self._string_to_int[doc_id] = int_id
            self._int_to_string[int_id] = doc_id
        return self._string_to_int[doc_id]

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
        """Hybrid search with vector similarity and keyword RRF.

        Executes a two-stage retrieval:
          1. Vector search using an exact FAISS L2 search,
             with the distance threshold applied in process.
          2. Keyword search scoring token overlap over the
             in-process record map.

        Results from both stages are merged using
        Reciprocal Rank Fusion (RRF), which combines the
        ranks from each retrieval method to produce a
        unified ranking.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty,
                the query is embedded automatically using
                the injected ``embedding_fn``.
            n_results: Maximum number of results to
                return.
            threshold: Maximum L2 distance for the vector
                search part. Smaller values enforce
                stricter matching.
            text_field: Record key containing the document
                text.
            metadata_fields: Record keys to include in the
                result metadata dict.
            session: Accepted for interface compatibility;
                ignored (FAISS is in-process).
            strict: When ``True``, a failed search raises
                :class:`ragit.types.RetrievalError`. The
                default ``False`` preserves the behavior of
                logging the error and returning an empty
                list.
            **filters: Equality filters matched against
                document fields (metadata and extra
                fields). FAISS cannot filter natively, so
                the vector search over-fetches candidates
                and filters them in process.

        Returns:
            list[dict]: A list of result dicts sorted by
            RRF score, each containing ``id``,
            ``document``, ``metadata``, ``distance`` (L2
            distance, or ``None`` for keyword-only hits),
            ``keyword_score`` (token-overlap ratio, or
            ``None`` for vector-only hits), and
            ``_rrf_score``.

        Raises:
            ragit.types.RetrievalError: If the search
                fails and ``strict=True``.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        if n_results < 1:
            raise ValueError(f"n_results must be a positive integer; got {n_results}")

        if not embedding:
            if self._embedding_fn is None:
                raise ValueError(
                    "embedding_fn is required to auto-embed the query; "
                    "pass an embedding or construct the store with embedding_fn"
                )
            embedding_list = await self._embedding_fn([query])
            embedding = embedding_list[0]

        validate_embedding(embedding, self.embedding_dim, label="query embedding")

        try:
            index = self._resolve_index()

            # Vector stage: exact L2 search. FAISS returns
            # -1 labels for slots beyond the index size.
            # The search over-fetches (10x) because FAISS
            # cannot filter natively: candidates that do
            # not match the filters are dropped in process
            # below, so matching documents deeper in the
            # ranking are still found.
            search_limit = n_results * 10
            query_vector = np.asarray([embedding], dtype="float32")
            faiss_distances, faiss_labels = await asyncio.to_thread(
                index.search, query_vector, search_limit
            )
            vector_ranked: list[str] = []
            distances: dict[str, float] = {}
            for distance, label in zip(
                faiss_distances[0], faiss_labels[0], strict=False
            ):
                if int(label) < 0:
                    continue
                doc_id = self._int_to_string.get(int(label))
                if doc_id is None:
                    continue
                record = self._records.get(doc_id, {})
                if not all(
                    record.get(key) == value for key, value in filters.items()
                ):
                    continue
                if float(distance) >= threshold:
                    continue
                vector_ranked.append(doc_id)
                distances[doc_id] = float(distance)
            vector_ranked = vector_ranked[:n_results]

            # Keyword stage: token-overlap scoring over
            # every document in the record map (FAISS has
            # no text search).
            keyword_scores: dict[str, float] = {}
            query_tokens = set(tokenize(query))
            for doc_id, record in self._records.items():
                if not all(record.get(key) == value for key, value in filters.items()):
                    continue
                score = token_overlap(
                    query_tokens, tokenize(str(record.get(text_field, "")))
                )
                if score > 0.0:
                    keyword_scores[doc_id] = score

            keyword_ranked = sorted(
                keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
            )

            results = []
            for doc_id, rrf_score in rrf_merge(
                vector_ranked, keyword_ranked, n_results
            ):
                record = self._records.get(doc_id, {})
                metadata = {
                    field: record[field]
                    for field in metadata_fields
                    if field in record and record[field] is not None
                }
                results.append(
                    {
                        "id": doc_id,
                        "document": record.get(text_field),
                        "metadata": metadata,
                        "distance": distances.get(doc_id),
                        "keyword_score": keyword_scores.get(doc_id),
                        "_rrf_score": rrf_score,
                    }
                )
            logger.debug(
                "hybrid_search('%s') on %s: found %d results.",
                query,
                self.table_name,
                len(results),
            )
            return results
        except Exception as exc:
            logger.error("Hybrid search failed on %s: %s", self.table_name, exc)
            if strict:
                raise RetrievalError(
                    f"Hybrid search failed on {self.table_name}: {exc}"
                ) from exc
            return []

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
        return sum(
            1
            for record in self._records.values()
            if all(record.get(key) == value for key, value in filters.items())
        )

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
        existing = [
            doc.get("id")
            for doc in documents
            if doc.get("id") in self._string_to_int
        ]
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

        vectors = np.asarray(
            [doc[embedding_field] for doc in documents], dtype="float32"
        )
        int_ids = np.asarray(
            [self._int_id(doc.get("id")) for doc in documents], dtype="int64"
        )
        await asyncio.to_thread(index.add_with_ids, vectors, int_ids)
        logger.info("Upserted %d documents to %s.", len(documents), self.table_name)
