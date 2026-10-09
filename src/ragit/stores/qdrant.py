"""Qdrant implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, and upsert operations against a Qdrant collection.

The vector stage uses Qdrant's native nearest-neighbor search;
the keyword stage scores token overlap in process over the
collection's payloads (the same approximation the in-memory
store uses), because Qdrant has no built-in full-text rank.
Both stages are ranked independently and merged with RRF, so
the result shape matches the other backends.

Point ids: Qdrant identifies points by unsigned integers or
UUIDs, so string document ids are mapped deterministically to
a UUID (``uuid5``) and the original id is kept in the payload
under the store's ``id_field``.

Configuration (no host imports): the store receives everything
through constructor arguments. The Qdrant client is created
lazily from ``url``/``api_key`` (or ``settings.database_url``)
on first use, or injected directly via ``client`` for hosts
and tests that manage their own client.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ragit.config import RagSettings
from ragit.stores._scoring import l2_distance, rrf_merge, token_overlap, tokenize
from ragit.stores.base import EmbeddingFunction, VectorStore
from ragit.stores.registry import vector_store_registry
from ragit.types import RetrievalError
from ragit.validation import validate_embedding, validate_identifier

logger = logging.getLogger(__name__)

_SCROLL_BATCH = 100
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
    from qdrant_client import FieldCondition, Filter, MatchValue  # noqa: PLC0415

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


@vector_store_registry.register("qdrant")
class QdrantStore(VectorStore):
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
                the Qdrant URL when ``url`` is omitted.
            embedding_fn: Async callable that embeds a list of
                texts, used to auto-embed queries.
            client: Pre-built Qdrant client (an
                ``AsyncQdrantClient``). When omitted, one is
                created lazily from ``url``/``api_key``.
            session_factory: Accepted for constructor
                compatibility with the other stores; ignored
                (Qdrant manages its own connections).

        Raises:
            ValueError: If ``table_name`` or ``id_field`` contain
                unsafe characters, or ``embedding_dim`` is omitted.
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
        self._url = url
        self._api_key = api_key
        self._settings = settings
        self._embedding_fn = embedding_fn
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
            from qdrant_client import Distance, VectorParams  # noqa: PLC0415

            await client.create_collection(
                collection_name=self.table_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.EUCLID
                ),
            )
        self._collection_ready = True

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
        offset: Any = None
        while True:
            points, offset = await client.scroll(
                collection_name=self.table_name,
                scroll_filter=query_filter,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = dict(point.payload or {})
                doc_id = str(payload.get(self.id_field, point.id))
                payloads.setdefault(doc_id, payload)
                score = token_overlap(
                    query_tokens, tokenize(str(payload.get(text_field, "")))
                )
                if score > 0.0:
                    keyword_scores[doc_id] = score
            if offset is None:
                break
        return keyword_scores, payloads

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
          1. Vector search using Qdrant's native nearest-neighbor
             search (Euclid distance), with the distance threshold
             applied in process.
          2. Keyword search scoring token overlap over the
             collection's payloads.

        Results from both stages are merged using Reciprocal Rank
        Fusion (RRF), which combines the ranks from each retrieval
        method to produce a unified ranking.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty, the query
                is embedded automatically using the injected
                ``embedding_fn``.
            n_results: Maximum number of results to return.
            threshold: Maximum L2 distance for the vector search
                part. Smaller values enforce stricter matching.
            text_field: Payload field containing the document text.
            metadata_fields: Payload fields to include in the result
                metadata dict.
            session: Accepted for interface compatibility; ignored
                (Qdrant has no SQL session concept).
            strict: When ``True``, a failed search raises
                :class:`ragit.types.RetrievalError`. The default
                ``False`` preserves the behavior of logging the
                error and returning an empty list.
            **filters: Equality filters applied to both search
                stages (e.g., ``tenant_id="..."``).

        Returns:
            list[dict]: A list of result dicts sorted by RRF score,
            each containing ``id``, ``document``, ``metadata``,
            ``distance`` (L2 distance, or ``None`` for keyword-only
            hits), ``keyword_score`` (token-overlap ratio, or
            ``None`` for vector-only hits), and ``_rrf_score``.

        Raises:
            ragit.types.RetrievalError: If the search fails and
                ``strict=True``.
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
            await self._ensure_collection()
            client = self._resolve_client()
            query_filter = _equality_filter(filters)

            # Vector stage: native nearest-neighbor search. The
            # n_results nearest points are fetched with their
            # vectors so the L2 distance threshold is applied in
            # process, matching the pgvector backend's semantics.
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

            # Keyword stage: token-overlap scoring over every
            # document in the collection (Qdrant has no native
            # full-text rank).
            keyword_scores, keyword_payloads = await self._keyword_stage(
                client, query, text_field, query_filter
            )
            for doc_id, payload in keyword_payloads.items():
                payloads.setdefault(doc_id, payload)

            keyword_ranked = sorted(
                keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
            )

            results = []
            for doc_id, rrf_score in rrf_merge(
                vector_ranked, keyword_ranked, n_results
            ):
                payload = payloads.get(doc_id, {})
                metadata = {
                    field: payload[field]
                    for field in metadata_fields
                    if field in payload and payload[field] is not None
                }
                results.append(
                    {
                        "id": doc_id,
                        "document": payload.get(text_field),
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
        """Count documents in the collection.

        Args:
            session: Accepted for interface compatibility; ignored.
            strict: When ``True``, a failed count raises
                :class:`ragit.types.RetrievalError`.
            **filters: Equality filters (e.g., ``tenant_id="..."``).

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
            ragit.types.RetrievalError: If the count fails and
                ``strict=True``.
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
            logger.error("Count failed on %s: %s", self.table_name, exc)
            if strict:
                raise RetrievalError(
                    f"Count failed on {self.table_name}: {exc}"
                ) from exc
            return 0

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

        await self._ensure_collection()
        client = self._resolve_client()

        from qdrant_client import PointStruct  # noqa: PLC0415

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
