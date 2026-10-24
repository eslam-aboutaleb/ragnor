"""Qdrant implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, lifecycle operations, and upsert operations
against a Qdrant collection.

The store subclasses :class:`ragnor.stores.base.HybridVectorStore`:
the vector stage lives in :meth:`vector_search` (Qdrant's
native nearest-neighbor search) and the keyword stage in
:meth:`keyword_search` (an in-process token-overlap scan over
the collection's payloads, because Qdrant has no built-in
full-text rank). The inherited composite ``hybrid_search``
runs both stages and fuses them with RRF, so the result
shape matches the other backends.

Keyword-search performance notice: the keyword stage is an
in-process fallback that scrolls every document in the
collection and scores token overlap. It is intended for
local development, small corpora, and testing -- not for
production-scale keyword retrieval. Hosts at scale combine a
pgvector :class:`VectorIndex` with an external lexical
engine adapted to :class:`ragnor.stores.base.LexicalIndex`.

Point ids: Qdrant identifies points by unsigned integers or
UUIDs, so string document ids are mapped deterministically to
a UUID (``uuid5``) and the original id is kept in the payload
under the store's ``id_field``.

Configuration (no host imports): the store receives everything
through constructor arguments. The Qdrant client is created
lazily from ``url``/``api_key`` (or ``settings.database_url``)
on first use, or injected directly via ``client`` for hosts
and tests that manage their own client.

Distance metric: the store accepts the uniform ``metric``
constructor kwarg so the registry can build every backend the
same way, but the cosine metric is NOT supported here -- only
the ``l2`` default is. Constructing the store with the cosine
metric raises ``ValueError``; cosine ranking is available from
the pgvector, memory, and faiss backends.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ragnor.core.config import RagSettings
from ragnor.stores._scoring import (
    format_search_hit,
    handle_count_error,
    l2_distance,
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


_SCROLL_BATCH = 100
# Keep the original namespace so existing Qdrant point IDs remain stable.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "ragit-qdrant-point-ids")


def _point_id(doc_id: Any) -> Any:
    """Map a document id to a Qdrant point id.

    Qdrant point ids are unsigned integers or UUIDs. Integer
    and UUID ids pass through; anything else (typically a
    string id) is mapped deterministically to a UUID derived
    from its string form. The original id is stored in the
    payload under the store's ``id_field``.
    """
    if isinstance(doc_id, (int, uuid.UUID)):
        return doc_id
    return uuid.uuid5(_ID_NAMESPACE, str(doc_id))


def _equality_filter(filters: dict[str, Any]) -> Any | None:
    """Build a Qdrant ``Filter`` from equality filters, or ``None``."""
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()
        ]
    )


def _selector_filter(
    ids: list[Any] | None,
    filters: dict[str, Any] | None,
    id_field: str,
) -> Any | None:
    """Build a Qdrant ``Filter`` selecting by ids and/or equality filters.

    The id restriction is a ``MatchAny`` on the payload's
    ``id_field``; equality filters become ``MatchValue``
    conditions. Both restrictions apply (conjunction) when
    both are supplied. Returns ``None`` when neither is
    supplied (matching every point).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # noqa: PLC0415

    must: list[Any] = []
    if ids is not None:
        must.append(FieldCondition(key=id_field, match=MatchAny(any=list(ids))))
    for key, value in (filters or {}).items():
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must) if must else None


@vector_store_registry.register("qdrant")
class QdrantStore(HybridVectorStore):
    """Qdrant vector store.

    Hybrid search combines Qdrant's native vector search with
    in-process token-overlap keyword matching, merged with the
    shared Reciprocal Rank Fusion formula.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str,
        id_field: str = "id",
        embedding_dim: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
        client: Any | None = None,
        session_factory: Any | None = None,
        metric: str | None = None,
    ):
        """Initialize the Qdrant store.

        Args:
            table_name: Name of the Qdrant collection.
            id_field: Payload field holding the document id.
            embedding_dim: Dimensionality of the embedding
                vectors. Required; pass the dimension configured
                for the active embedding model.
            url: Qdrant server URL (e.g. ``http://localhost:6333``).
                Falls back to ``settings.database_url``.
            api_key: API key for authenticated Qdrant servers.
            settings: Host settings; ``database_url`` is used as
                the Qdrant URL when ``url`` is omitted, and the
                optional ``rag_distance_metric`` field selects the
                metric when ``metric`` is omitted.
            embedding_fn: Async callable that embeds a list of
                texts, used to auto-embed queries.
            client: Pre-built Qdrant client (an
                ``AsyncQdrantClient``). When omitted, one is
                created lazily from ``url``/``api_key``.
            session_factory: Accepted for constructor
                compatibility with the other stores; ignored
                (Qdrant manages its own connections).
            metric: Distance metric for the vector stage. Only
                ``l2`` (the default) is supported by this
                backend.

        Raises:
            ValueError: If ``table_name`` or ``id_field`` contain
                unsafe characters, if ``embedding_dim`` is omitted,
                if ``metric`` is not a supported distance metric,
                or if the cosine metric is requested (unsupported
                by this backend).
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
        if self.metric == "cosine":
            raise ValueError(
                "the 'cosine' distance metric is unsupported by the Qdrant "
                "backend; use l2 (the default), or switch to the pgvector, "
                "memory, or faiss backend for cosine ranking"
            )
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self._url = url
        self._api_key = api_key
        self._client = client
        self._collection_ready = False

    def _resolve_client(self) -> Any:
        """Return the Qdrant client, creating it on first use.

        Raises:
            RuntimeError: If no client was injected and no URL
                is available from ``url`` or ``settings``.
        """
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

            url = self._url
            if url is None and self._settings is not None:
                url = self._settings.database_url
            if not url:
                raise RuntimeError(
                    "QdrantStore needs a url, a client, or settings with "
                    "a database_url pointing at the Qdrant server"
                )
            self._client = AsyncQdrantClient(url=url, api_key=self._api_key)
        return self._client

    async def _ensure_collection(self) -> None:
        """Create the collection on first use if it is missing."""
        if self._collection_ready:
            return
        client = self._resolve_client()
        if not await client.collection_exists(self.table_name):
            from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

            await client.create_collection(
                collection_name=self.table_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.EUCLID),
            )
        self._collection_ready = True

    async def _scroll_all_points(self, client: Any, query_filter: Any) -> list[Any]:
        all_points: list[Any] = []
        offset: Any = None
        while True:
            points, offset = await client.scroll(
                collection_name=self.table_name,
                scroll_filter=query_filter,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
            )
            all_points.extend(points)
            if offset is None:
                break
        return all_points

    async def _keyword_stage(
        self,
        client: Any,
        query: str,
        text_field: str,
        query_filter: Any,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Score token overlap over every document in the collection.

        Qdrant has no native full-text rank, so the keyword
        stage paginates through the collection's payloads and
        scores token overlap in process. Returns the keyword
        scores and the payloads of every scanned document.
        """
        keyword_scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        query_tokens = set(tokenize(query))
        points = await self._scroll_all_points(client, query_filter)
        for point in points:
            payload = dict(point.payload or {})
            doc_id = str(payload.get(self.id_field, point.id))
            payloads.setdefault(doc_id, payload)
            score = token_overlap(query_tokens, tokenize(str(payload.get(text_field, ""))))
            if score > 0.0:
                keyword_scores[doc_id] = score
        return keyword_scores, payloads

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

        The ``n_results`` nearest points are fetched with their
        vectors so the L2 distance threshold is applied in
        process, matching the pgvector backend's semantics.

        Args:
            embedding: Query embedding vector (required; the
                method does not auto-embed).
            n_results: Maximum number of results to return.
            threshold: Maximum L2 distance.
            text_field: Payload field containing the document text.
            metadata_fields: Payload fields to include in the
                result metadata dict.
            session: Accepted for interface compatibility; ignored
                (Qdrant has no SQL session concept).
            **filters: Equality filters applied to the search
                (e.g., ``tenant_id="..."``).

        Returns:
            list[dict]: Result dicts ordered by distance ascending,
            each carrying ``id``, ``document``, ``metadata``,
            ``distance``, and ``keyword_score`` (``None``).

        Raises:
            ValueError: If identifiers are unsafe or the embedding
                has the wrong dimension.
            Exception: Re-raises Qdrant errors.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        validate_embedding(embedding, self.embedding_dim, label="query embedding")

        await self._ensure_collection()
        client = self._resolve_client()
        query_filter = _equality_filter(filters)

        response = await client.query_points(
            collection_name=self.table_name,
            query=embedding,
            limit=n_results,
            with_payload=True,
            with_vectors=True,
            query_filter=query_filter,
        )
        vector_ranked: list[str] = []
        distances: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for point in response.points:
            point_vector = list(point.vector or [])
            distance = l2_distance(embedding, point_vector)
            if distance >= threshold:
                continue
            doc_id = str(point.payload.get(self.id_field, point.id))
            vector_ranked.append(doc_id)
            distances[doc_id] = distance
            payloads[doc_id] = dict(point.payload or {})

        results = []
        for doc_id in vector_ranked:
            payload = payloads[doc_id]
            results.append(
                format_search_hit(
                    doc_id=doc_id,
                    document=payload.get(text_field),
                    metadata=project_metadata(payload, metadata_fields),
                    distance=distances[doc_id],
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
        """Token-overlap keyword search over the collection's payloads.

        Qdrant has no native full-text rank, so the keyword
        stage scrolls every document in the collection and
        scores token overlap in process -- an in-process
        fallback intended for local development, small
        corpora, and testing, not production-scale keyword
        retrieval.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            text_field: Payload field containing the document text.
            metadata_fields: Payload fields to include in the
                result metadata dict.
            session: Accepted for interface compatibility; ignored
                (Qdrant has no SQL session concept).
            **filters: Equality filters applied to the search
                (e.g., ``tenant_id="..."``).

        Returns:
            list[dict]: Result dicts ordered by keyword score
            descending (tie-broken by id), each carrying ``id``,
            ``document``, ``metadata``, ``distance`` (``None``),
            and ``keyword_score``.

        Raises:
            ValueError: If identifiers are unsafe.
            Exception: Re-raises Qdrant errors.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        await self._ensure_collection()
        client = self._resolve_client()
        query_filter = _equality_filter(filters)

        keyword_scores, payloads = await self._keyword_stage(
            client, query, text_field, query_filter
        )
        keyword_ranked = sorted(
            keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
        )[:n_results]

        results = []
        for doc_id in keyword_ranked:
            payload = payloads.get(doc_id, {})
            results.append(
                format_search_hit(
                    doc_id=doc_id,
                    document=payload.get(text_field),
                    metadata=project_metadata(payload, metadata_fields),
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
            session: Accepted for interface compatibility; ignored.
            strict: When ``True``, a failed count raises
                :class:`ragnor.core.types.RetrievalError`. The default
                ``False`` preserves the behavior of logging the
                error and returning ``0``.
            **filters: Equality filters (e.g., ``tenant_id="..."``).

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
            ValueError: If a filter key is unsafe.
            ragnor.core.types.RetrievalError: If the count fails and
                ``strict=True``.
            Exception: Re-raises Qdrant errors when ``strict=True``.
        """
        for key in filters:
            validate_identifier(key, "filter key")

        try:
            await self._ensure_collection()
            client = self._resolve_client()
            response = await client.count(
                collection_name=self.table_name,
                count_filter=_equality_filter(filters),
            )
            return int(response.count)
        except Exception as exc:
            return handle_count_error(self.table_name, exc, strict, logger)

    async def get_embeddings(self, ids: list[str], session: Any = None) -> dict[str, list[float]]:
        if not ids:
            return {}
        await self._ensure_collection()
        points = await self._resolve_client().retrieve(
            collection_name=self.table_name,
            ids=[_point_id(doc_id) for doc_id in ids],
            with_vectors=True,
            with_payload=True,
        )
        return {str(point.payload[self.id_field]): list(point.vector) for point in points}

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete points selected by ``ids`` and/or ``filters``.

        At least one selector must be supplied; an empty ``ids``
        list is rejected. When both are supplied, both
        restrictions apply (conjunction). Qdrant's ``delete``
        does not report how many points it removed, so the
        deletion count is the collection count before the
        delete minus the count after it.

        Args:
            ids: Document ids to delete (mapped to Qdrant point
                ids with the store's ``_point_id`` mapping).
            filters: Equality filters (e.g., ``tenant_id="..."``).
            session: Accepted for interface compatibility; ignored
                (Qdrant manages its own connections).

        Returns:
            int: Number of deleted documents.

        Raises:
            ValueError: If no selector is supplied, ``ids`` is
                empty, or a filter key is unsafe.
            Exception: Re-raises Qdrant errors.
        """
        validate_delete_selectors(ids, filters)

        await self._ensure_collection()
        client = self._resolve_client()

        if ids is not None and not filters:
            # A point-ids selector carries the id restriction alone, so
            # the deletion count is the whole-collection count delta.
            before_response = await client.count(collection_name=self.table_name)
            before = int(before_response.count)
            from qdrant_client.models import PointIdsList  # noqa: PLC0415

            selector: Any = PointIdsList(points=[_point_id(doc_id) for doc_id in ids])
            await client.delete(collection_name=self.table_name, points_selector=selector)
            after_response = await client.count(collection_name=self.table_name)
            after = int(after_response.count)
        else:
            # A filter selector carries the id restriction (a MatchAny on
            # the payload's id_field) and/or the equality filters, so both
            # restrictions apply when both are supplied.
            query_filter = _selector_filter(ids, filters, self.id_field)
            before_response = await client.count(
                collection_name=self.table_name, count_filter=query_filter
            )
            before = int(before_response.count)
            from qdrant_client.models import FilterSelector  # noqa: PLC0415

            selector = FilterSelector(filter=query_filter)
            await client.delete(collection_name=self.table_name, points_selector=selector)
            after_response = await client.count(
                collection_name=self.table_name, count_filter=query_filter
            )
            after = int(after_response.count)
        return before - after

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``.

        The selection scrolls the collection with a ``MatchAny``
        filter on the payload's ``id_field`` (for ``ids``) and/or
        equality filters. With no selector, every document is
        returned.

        Args:
            ids: Document ids to fetch.
            filters: Equality filters (e.g., ``tenant_id="..."``).
            session: Accepted for interface compatibility; ignored
                (Qdrant has no SQL session concept).
            text_field: Payload field containing the document text.
            metadata_fields: Payload fields to include in the
                result metadata dict.

        Returns:
            list[dict]: ``id``/``document``/``metadata`` dicts
            (score keys are meaningless for direct fetches).

        Raises:
            ValueError: If any identifier is unsafe.
            Exception: Re-raises Qdrant errors.
        """
        validate_identifier(text_field, "text_field")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")
        if filters is not None:
            for key in filters:
                validate_identifier(key, "filter key")

        await self._ensure_collection()
        client = self._resolve_client()
        query_filter = _selector_filter(ids, filters, self.id_field)

        points = await self._scroll_all_points(client, query_filter)
        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            doc_id = str(payload.get(self.id_field, point.id))
            results.append(
                {
                    "id": doc_id,
                    "document": payload.get(text_field),
                    "metadata": project_metadata(payload, metadata_fields),
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
            documents: List of document dicts with keys: id, text,
                metadata, embedding. Metadata keys are included as
                additional payload fields.
            session: Accepted for interface compatibility; ignored.
            text_field: Payload field for the document text.
                Defaults to "text".
            embedding_field: Document dict key holding the
                embedding vector. Defaults to "embedding".
            id_field: Payload field for the document id. Defaults
                to the store's configured ``id_field``.
            extra_fields: Additional static fields to include in
                every payload (e.g., ``{"tenant_id": "..."}``).

        Raises:
            ValueError: If any field name contains unsafe
                characters.
            Exception: Re-raises Qdrant errors after logging.
        """
        id_field = validate_upsert_documents(
            documents,
            text_field,
            embedding_field,
            id_field,
            self.id_field,
            extra_fields,
            self.embedding_dim,
        )
        if not documents:
            return

        await self._ensure_collection()
        client = self._resolve_client()

        from qdrant_client.models import PointStruct  # noqa: PLC0415

        points = []
        for doc in documents:
            payload: dict[str, Any] = {
                id_field: doc.get("id"),
                text_field: doc[text_field],
            }
            payload.update(doc.get("metadata", {}))
            if extra_fields:
                payload.update(extra_fields)
            points.append(
                PointStruct(
                    id=_point_id(doc.get("id")),
                    vector=list(doc[embedding_field]),
                    payload=payload,
                )
            )
        await client.upsert(collection_name=self.table_name, points=points)
        logger.info("Upserted %d documents to %s.", len(documents), self.table_name)
