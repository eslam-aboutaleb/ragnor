"""Chroma implementation of the VectorStore interface.

Provides hybrid search (vector similarity + token-overlap
keyword matching with Reciprocal Rank Fusion), document
counting, and upsert operations against a Chroma collection.

The vector stage uses Chroma's native nearest-neighbor
search; the keyword stage scores token overlap in process
over the collection's documents (the same approximation
the in-memory store uses), because Chroma has no built-in
full-text rank. Both stages are ranked independently and
merged with RRF, so the result shape matches the other
backends.

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
from ragit.stores._scoring import rrf_merge, token_overlap, tokenize
from ragit.stores.base import EmbeddingFunction, VectorStore
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
class ChromaStore(VectorStore):
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
        validate_identifier(table_name, "table_name")
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self._settings = settings
        self._embedding_fn = embedding_fn
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
          1. Vector search using Chroma's native
             nearest-neighbor search (L2 distance), with
             the distance threshold applied in process.
          2. Keyword search scoring token overlap over the
             collection's documents.

        Results from both stages are merged using Reciprocal
        Rank Fusion (RRF), which combines the ranks from each
        retrieval method to produce a unified ranking.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty, the
                query is embedded automatically using the
                injected ``embedding_fn``.
            n_results: Maximum number of results to return.
            threshold: Maximum L2 distance for the vector
                search part. Smaller values enforce stricter
                matching.
            text_field: Accepted for interface compatibility;
                Chroma stores document text as the record's
                document.
            metadata_fields: Metadata keys to include in the
                result metadata dict.
            session: Accepted for interface compatibility;
                ignored (Chroma has no SQL session concept).
            strict: When ``True``, a failed search raises
                :class:`ragit.types.RetrievalError`. The
                default ``False`` preserves the behavior of
                logging the error and returning an empty list.
            **filters: Equality filters applied to both
                search stages (e.g., ``tenant_id="..."``).

        Returns:
            list[dict]: A list of result dicts sorted by RRF
            score, each containing ``id``, ``document``,
            ``metadata``, ``distance`` (L2 distance, or
            ``None`` for keyword-only hits),
            ``keyword_score`` (token-overlap ratio, or
            ``None`` for vector-only hits), and ``_rrf_score``.

        Raises:
            ragit.types.RetrievalError: If the search fails
                and ``strict=True``.
        """
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
            collection = self._resolve_collection()
            where = _where_filter(filters)

            # Vector stage: native nearest-neighbor search.
            # Chroma returns L2 distances; the threshold is
            # applied in process, matching the other
            # backends' semantics.
            response = await asyncio.to_thread(
                collection.query,
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
            )
            vector_ranked: list[str] = []
            distances: dict[str, float] = {}
            documents: dict[str, str] = {}
            metadatas: dict[str, dict[str, Any]] = {}
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
                vector_ranked.append(doc_id)
                distances[doc_id] = float(distance)
                documents[doc_id] = document
                metadatas[doc_id] = dict(metadata or {})

            # Keyword stage: token-overlap scoring over every
            # document in the collection (Chroma has no
            # native full-text rank).
            keyword_scores: dict[str, float] = {}
            query_tokens = set(tokenize(query))
            all_records = await asyncio.to_thread(collection.get, where=where)
            for raw_id, document in zip(
                all_records.get("ids") or [],
                all_records.get("documents") or [],
                strict=False,
            ):
                doc_id = str(raw_id)
                documents.setdefault(doc_id, document)
                score = token_overlap(query_tokens, tokenize(str(document)))
                if score > 0.0:
                    keyword_scores[doc_id] = score

            keyword_ranked = sorted(
                keyword_scores, key=lambda doc_id: (-keyword_scores[doc_id], doc_id)
            )

            results = []
            for doc_id, rrf_score in rrf_merge(
                vector_ranked, keyword_ranked, n_results
            ):
                metadata = {
                    field: metadatas.get(doc_id, {}).get(field)
                    for field in metadata_fields
                    if metadatas.get(doc_id, {}).get(field) is not None
                }
                results.append(
                    {
                        "id": doc_id,
                        "document": documents.get(doc_id),
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
            session: Accepted for interface compatibility;
                ignored.
            strict: When ``True``, a failed count raises
                :class:`ragit.types.RetrievalError`.
            **filters: Equality filters (e.g.,
                ``tenant_id="..."``).

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
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
