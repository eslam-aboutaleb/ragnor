"""Vector store abstraction for ragit.

Defines the interface for hybrid retrieval (vector + keyword
search) operations, allowing different backends (pgvector,
in-memory, etc.) to be swapped via the store registry without
changing the retrieval or ingestion code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

EmbeddingFunction = Callable[[list[str]], Awaitable[list[list[float]]]]
"""Async callable that embeds a list of texts into vectors."""


class VectorStore(ABC):
    """Abstract interface for vector store operations.

    Session ownership rule: when the caller supplies a ``session``
    to a method, the caller owns the transaction and is responsible
    for committing or rolling it back. When no session is supplied,
    the store opens its own session and commits where the operation
    is transactional.
    """

    @abstractmethod
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
        """Perform hybrid vector + PostgreSQL full-text search using Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector.
            n_results: Maximum number of results to return.
            threshold: Maximum distance for vector search part.
            text_field: Name of the column containing the document text.
            metadata_fields: Additional column names to include in the
                result metadata dict.
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            strict: When ``True``, retrieval failures raise
                :class:`ragit.types.RetrievalError` instead of being
                swallowed into an empty result list.
            **filters: Additional filters (e.g., tenant_id).

        Returns:
            List of result dicts with keys: id, document, metadata,
            distance, keyword_score, _rrf_score.
        """
        ...

    @abstractmethod
    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the store.

        Args:
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            strict: When ``True``, failures raise
                :class:`ragit.types.RetrievalError` instead of
                being swallowed into ``0``.
            **filters: Optional filters (e.g., tenant_id).

        Returns:
            Number of matching documents.
        """
        ...

    @abstractmethod
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

        Args:
            documents: List of document dicts with keys: id, text,
                metadata, embedding.
            session: Optional external session. When supplied, the
                caller owns the transaction (see the class docstring).
            text_field: Name of the text column.
            embedding_field: Name of the embedding column.
            id_field: Name of the primary key column. Defaults to the
                store's configured id field.
            extra_fields: Additional static fields to include in every
                INSERT (e.g., ``{"tenant_id": "..."}``).
        """
        ...
