"""Chroma implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, lifecycle operations, and upsert operations against
a Chroma collection.

The store subclasses :class:`ragnor.stores.base.HybridVectorStore`:
the vector stage lives in :meth:`vector_search` (Chroma's
native nearest-neighbor search) and the keyword stage in
:meth:`keyword_search` (an in-process token-overlap scan over
the collection's documents, because Chroma has no built-in
full-text rank). The inherited composite ``hybrid_search``
runs both stages and fuses them with RRF, so the result shape
matches the other backends.

Keyword-search performance notice: the keyword stage is an
in-process fallback that fetches every document in the
collection and scores token overlap. It is intended for local
development, small corpora, and testing -- not for
production-scale keyword retrieval. Hosts at scale combine a
pgvector :class:`VectorIndex` with an external lexical engine
adapted to :class:`ragnor.stores.base.LexicalIndex`.

The store talks to Chroma's synchronous client API and
wraps every call in ``asyncio.to_thread`` so the event
loop is never blocked; the sync client is Chroma's stable
surface (``EphemeralClient``, ``PersistentClient``,
``HttpClient``). A client can also be injected directly
via ``client`` for hosts and tests that manage their own.

Configuration (no host imports): the store receives
everything through constructor arguments. The Chroma
client is created lazily on first use -- an
``EphemeralClient`` by default, or a
``PersistentClient`` when ``path`` is given.

Distance metric: the store accepts the uniform ``metric``
constructor kwarg so the registry can build every backend
the same way, but the cosine metric is NOT supported here
-- only the ``l2`` default is. Constructing the store with
the cosine metric raises ``ValueError``; cosine ranking is
available from the pgvector, memory, and faiss backends.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragnor.core.config import RagSettings
from ragnor.stores._scoring import (
    format_search_hit,
    handle_count_error,
    project_metadata,
    token_overlap,
    tokenize,
    validate_upsert_documents,
)
from ragnor.stores.base import EmbeddingFunction, HybridVectorStore
from ragnor.stores.registry import vector_store_registry
from ragnor.core.validation import (
    validate_delete_selectors,
    validate_embedding,
    validate_identifier,
)

logger = logging.getLogger(__name__)


def _where_filter(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Build a Chroma ``where`` filter from equality filters."""
    if not filters:
        return None
    return {key: {"$eq": value} for key, value in filters.items()}


@vector_store_registry.register("chroma")
class ChromaStore(HybridVectorStore):
    """Chroma vector store.

    Hybrid search combines Chroma's native vector search
    with in-process token-overlap keyword matching, merged
    with the shared Reciprocal Rank Fusion formula.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str,
        id_field: str = "id",
        embedding_dim: int | None = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
        client: Any | None = None,
        path: str | None = None,
        session_factory: Any | None = None,
        metric: str | None = None,
    ):
        """Initialize the Chroma store.

        Args:
            table_name: Name of the Chroma collection.
            id_field: Accepted for constructor compatibility
                with the other stores; Chroma stores the
                document id as the record id.
            embedding_dim: Dimensionality of the embedding
                vectors. Required; pass the dimension
                configured for the active embedding model.
            settings: Host settings; accepted for
                constructor compatibility with the other
                stores, and read for the optional
                ``rag_distance_metric`` field when ``metric``
                is omitted.
            embedding_fn: Async callable that embeds a list
                of texts, used to auto-embed queries.
            client: Pre-built Chroma client. When omitted,
                one is created lazily -- an
                ``EphemeralClient`` by default, or a
                ``PersistentClient`` when ``path`` is given.
            path: Directory for a persistent Chroma database.
                When omitted, an in-memory ephemeral
                database is used.
            session_factory: Accepted for constructor
                compatibility with the other stores; ignored
                (Chroma manages its own connections).
            metric: Distance metric for the vector stage. Only
                ``l2`` (the default) is supported by this
                backend.

        Raises:
            ValueError: If ``table_name`` contains unsafe
                characters, if ``embedding_dim`` is omitted,
                if ``metric`` is not a supported distance
                metric, or if the cosine metric is requested
                (unsupported by this backend).
        """
        super().__init__(
            table_name=table_name,
            id_field=id_field,
            embedding_dim=embedding_dim,
            settings=settings,
            embedding_fn=embedding_fn,
            session_factory=session_factory,
            metric=metric,
        )
        if self.metric == "cosine":
            raise ValueError(
                "the 'cosine' distance metric is unsupported by the Chroma "
                "backend; use l2 (the default), or switch to the pgvector, "
                "memory, or faiss backend for cosine ranking"
            )
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self._settings = settings
        self._client = client
        self._path = path
        self._collection: Any | None = None

    def _resolve_collection(self) -> Any:
        """Return the Chroma collection, creating it on first use.

        Raises:
            RuntimeError: If no client was injected and
                ``chromadb`` is not installed.
        """
        if self._collection is None:
            if self._client is None:
                import chromadb  # noqa: PLC0415

                if self._path is not None:
                    self._client = chromadb.PersistentClient(path=self._path)
                else:
                    self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=self.table_name)
        return self._collection

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
        """Native nearest-neighbor search over the collection.

        Chroma returns L2 distances; the threshold is applied
        in process, matching the other backends' semantics.

        Args:
            embedding: Query embedding vector (required; the
                method does not auto-embed).
            n_results: Maximum number of results to return.
            threshold: Maximum L2 distance.
            text_field: Accepted for interface compatibility;
                Chroma stores document text as the record's
                document.
            metadata_fields: Metadata keys to include in the
                result metadata dict.
            session: Accepted for interface compatibility;
                ignored (Chroma has no SQL session concept).
            **filters: Equality filters applied to the search
                (e.g., ``tenant_id="..."``).

        Returns:
            list[dict]: Result dicts ordered by distance ascending,
            each carrying ``id``, ``document``, ``metadata``,
            ``distance``, and ``keyword_score`` (``None``).

        Raises:
            ValueError: If identifiers are unsafe or the embedding
                has the wrong dimension.
            Exception: Re-raises Chroma errors.
        """
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        validate_embedding(embedding, self.embedding_dim, label="query embedding")

        collection = self._resolve_collection()
        where = _where_filter(filters)

        response = await asyncio.to_thread(
            collection.query,
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )
        results = []
        for raw_id, document, distance, metadata in zip(
            (response.get("ids") or [[]])[0],
            (response.get("documents") or [[]])[0],
            (response.get("distances") or [[]])[0],
            (response.get("metadatas") or [[]])[0],
            strict=False,
        ):
            if distance is not None and distance >= threshold:
                continue
            doc_id = str(raw_id)
            results.append(
                format_search_hit(
                    doc_id=doc_id,
                    document=document,
                    metadata=project_metadata(dict(metadata or {}), metadata_fields),
                    distance=float(distance),
                )
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
        """Token-overlap keyword search over the collection's documents.

        Chroma has no native full-text rank, so the keyword
        stage fetches every document in the collection and
        scores token overlap in process -- an in-process
        fallback intended for local development, small
        corpora, and testing, not production-scale keyword
        retrieval.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            text_field: Ignored for payload retrieval; Chroma stores
                document text natively as the record's document.
            metadata_fields: Keys projected into the returned result metadata.
            session: Ignored (Chroma operations are non-transactional).
            **filters: Metadata equality constraints passed to Chroma.

        Returns:
            list[dict]: Result dicts ordered by keyword score
            descending (tie-broken by id), each carrying ``id``,
            ``document``, ``metadata``, ``distance`` (``None``),
            and ``keyword_score``.

        Raises:
            ValueError: If identifiers are unsafe.
            Exception: Re-raises Chroma errors.
        """
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        collection = self._resolve_collection()
        where = _where_filter(filters)
        query_tokens = set(tokenize(query))

        all_records = await asyncio.to_thread(collection.get, where=where)
        keyword_scores: dict[str, float] = {}
        documents: dict[str, str] = {}
        metadatas: dict[str, dict[str, Any]] = {}
        for raw_id, document, metadata in zip(
            all_records.get("ids") or [],
            all_records.get("documents") or [],
            all_records.get("metadatas") or [],
            strict=False,
        ):
            doc_id = str(raw_id)
            documents[doc_id] = document
            metadatas[doc_id] = dict(metadata or {})
            score = token_overlap(query_tokens, tokenize(str(document)))
            if score > 0.0:
                keyword_scores[doc_id] = score

        keyword_ranked = sorted(
            keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
        )[:n_results]

        results = []
        for doc_id in keyword_ranked:
            record_metadata = metadatas.get(doc_id, {})
            results.append(
                format_search_hit(
                    doc_id=doc_id,
                    document=documents.get(doc_id),
                    metadata=project_metadata(record_metadata, metadata_fields),
                    keyword_score=keyword_scores[doc_id],
                )
            )
        return results

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the collection.

        Args:
            session: Accepted for interface compatibility;
                ignored.
            strict: When ``True``, a failed count raises
                :class:`ragnor.core.types.RetrievalError`.
            **filters: Equality filters (e.g.,
                ``tenant_id="..."``).

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
            ValueError: If a filter key is unsafe.
            ragnor.core.types.RetrievalError: If the count fails
                and ``strict=True``.
        """
        for key in filters:
            validate_identifier(key, "filter key")

        try:
            collection = self._resolve_collection()
            if filters:
                records = await asyncio.to_thread(collection.get, where=_where_filter(filters))
                return len(records.get("ids") or [])
            return int(await asyncio.to_thread(collection.count))
        except Exception as exc:
            return handle_count_error(self.table_name, exc, strict, logger)

    async def get_embeddings(self, ids: list[str], session: Any = None) -> dict[str, list[float]]:
        if not ids:
            return {}
        records = await asyncio.to_thread(
            self._resolve_collection().get, ids=ids, include=["embeddings"]
        )
        vectors = records.get("embeddings")
        if vectors is None:
            return {}
        return {
            str(doc_id): list(vector)
            for doc_id, vector in zip(records["ids"], vectors, strict=True)
        }

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete documents selected by ``ids`` and/or ``filters``.

        At least one selector must be supplied; an empty ``ids``
        list is rejected. When both are supplied, both
        restrictions apply (conjunction): the matching records
        are resolved first, then exactly those ids are deleted,
        so the returned count is the number of documents that
        existed and matched.

        Args:
            ids: Document ids to delete.
            filters: Equality filters (e.g.,
                ``tenant_id="..."``).
            session: Accepted for interface compatibility;
                ignored (Chroma manages its own connections).

        Returns:
            int: Number of deleted documents.

        Raises:
            ValueError: If no selector is supplied, ``ids`` is
                empty, or a filter key is unsafe.
            Exception: Re-raises Chroma errors.
        """
        validate_delete_selectors(ids, filters)

        collection = self._resolve_collection()
        where = _where_filter(filters) if filters else None
        records = await asyncio.to_thread(
            collection.get,
            ids=[str(doc_id) for doc_id in ids] if ids is not None else None,
            where=where,
        )
        matching_ids = [str(doc_id) for doc_id in (records.get("ids") or [])]
        if matching_ids:
            await asyncio.to_thread(collection.delete, ids=matching_ids)
        return len(matching_ids)

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``.

        The selection is resolved by Chroma's ``get`` (by record
        ids and/or ``where`` equality filters). With no selector,
        every document is returned.

        Args:
            ids: Document ids to fetch.
            filters: Equality filters (e.g.,
                ``tenant_id="..."``).
            session: Accepted for interface compatibility;
                ignored (Chroma has no SQL session concept).
            text_field: Accepted for interface compatibility;
                Chroma stores document text as the record's
                document.
            metadata_fields: Metadata keys to include in the
                result metadata dict.

        Returns:
            list[dict]: ``id``/``document``/``metadata`` dicts
            (score keys are meaningless for direct fetches).

        Raises:
            ValueError: If any identifier is unsafe.
            Exception: Re-raises Chroma errors.
        """
        for key in filters or {}:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        collection = self._resolve_collection()
        where = _where_filter(filters) if filters else None
        records = await asyncio.to_thread(
            collection.get,
            ids=[str(doc_id) for doc_id in ids] if ids is not None else None,
            where=where,
        )
        results = []
        for raw_id, document, metadata in zip(
            records.get("ids") or [],
            records.get("documents") or [],
            records.get("metadatas") or [],
            strict=False,
        ):
            record_metadata = dict(metadata or {})
            results.append(
                {
                    "id": str(raw_id),
                    "document": document,
                    "metadata": {
                        field: record_metadata[field]
                        for field in metadata_fields
                        if field in record_metadata and record_metadata[field] is not None
                    },
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
        """Upsert documents into the collection.

        Args:
            documents: List of document dicts with keys: id,
                text, metadata, embedding. Metadata keys are
                included as Chroma metadata; the document text
                is stored as the record's document.
            session: Accepted for interface compatibility;
                ignored.
            text_field: Document dict key holding the text.
                Defaults to "text".
            embedding_field: Document dict key holding the
                embedding vector. Defaults to "embedding".
            id_field: Accepted for interface compatibility;
                Chroma stores the document id as the record
                id.
            extra_fields: Additional static fields to include
                in every record's metadata (e.g.,
                ``{"tenant_id": "..."}``).

        Raises:
            ValueError: If any metadata key contains unsafe
                characters.
            Exception: Re-raises Chroma errors after logging.
        """
        validate_upsert_documents(
            documents,
            text_field,
            embedding_field,
            "id",
            self.id_field,
            extra_fields,
            self.embedding_dim,
        )
        if not documents:
            return

        ids = [str(doc.get("id")) for doc in documents]
        texts = [doc[text_field] for doc in documents]
        embeddings = [list(doc[embedding_field]) for doc in documents]
        metadatas: list[dict[str, Any]] = []
        for doc in documents:
            metadata = dict(doc.get("metadata", {}))
            if extra_fields:
                metadata.update(extra_fields)
            metadatas.append(metadata)

        try:
            collection = self._resolve_collection()
            await asyncio.to_thread(
                collection.upsert,
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Upserted %d documents to %s.", len(documents), self.table_name)
        except Exception as exc:
            logger.error("Upsert failed on %s: %s", self.table_name, type(exc).__name__)
            raise
