"""Chroma implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, lifecycle operations, and upsert operations against
a Chroma collection.

The store subclasses :class:`ragit.stores.base.HybridVectorStore`:
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
adapted to :class:`ragit.stores.base.LexicalIndex`.

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
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragit.config import RagSettings
from ragit.stores._scoring import token_overlap, tokenize
from ragit.stores.base import (
    EmbeddingFunction,
    HybridVectorStore,
)
from ragit.stores.registry import vector_store_registry
from ragit.types import RetrievalError
from ragit.validation import validate_embedding, validate_identifier

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
                stores.
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

        Raises:
            ValueError: If ``table_name`` contains unsafe
                characters, or ``embedding_dim`` is omitted.
        """
        super().__init__(
            table_name=table_name,
            id_field=id_field,
            embedding_dim=embedding_dim,
            settings=settings,
            embedding_fn=embedding_fn,
            session_factory=session_factory,
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
            self._collection = self._client.get_or_create_collection(
                name=self.table_name
            )
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
            record_metadata = dict(metadata or {})
            results.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": {
                        field: record_metadata[field]
                        for field in metadata_fields
                        if field in record_metadata
                        and record_metadata[field] is not None
                    },
                    "distance": float(distance),
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
                {
                    "id": doc_id,
                    "document": documents.get(doc_id),
                    "metadata": {
                        field: record_metadata[field]
                        for field in metadata_fields
                        if field in record_metadata
                        and record_metadata[field] is not None
                    },
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
        """Count documents in the collection.

        Args:
            session: Accepted for interface compatibility;
                ignored.
            strict: When ``True``, a failed count raises
                :class:`ragit.types.RetrievalError`.
            **filters: Equality filters (e.g.,
                ``tenant_id="..."``).

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
            ValueError: If a filter key is unsafe.
            ragit.types.RetrievalError: If the count fails
                and ``strict=True``.
        """
        for key in filters:
            validate_identifier(key, "filter key")

        try:
            collection = self._resolve_collection()
            if filters:
                records = await asyncio.to_thread(
                    collection.get, where=_where_filter(filters)
                )
                return len(records.get("ids") or [])
            return int(await asyncio.to_thread(collection.count))
        except Exception as exc:
            logger.error("Count failed on %s: %s", self.table_name, exc)
            if strict:
                raise RetrievalError(
                    f"Count failed on {self.table_name}: {exc}"
                ) from exc
            return 0

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
        if ids is None and filters is None:
            raise ValueError(
                "delete requires at least one of ids or filters"
            )
        if ids is not None and len(ids) == 0:
            raise ValueError(
                "ids must not be empty; an empty id list never "
                "means delete everything"
            )
        if filters is not None:
            for key in filters:
                validate_identifier(key, "filter key")

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
        for key in (filters or {}):
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
                        if field in record_metadata
                        and record_metadata[field] is not None
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
        validate_identifier(text_field, "text_field")
        validate_identifier(embedding_field, "embedding_field")
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
            logger.info(
                "Upserted %d documents to %s.", len(documents), self.table_name
            )
        except Exception as exc:
            logger.error("Upsert failed on %s: %s", self.table_name, exc)
            raise
