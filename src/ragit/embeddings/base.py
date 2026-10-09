"""Embedding function seam for ragit.

The :class:`EmbeddingFunction` protocol is the boundary every
embedding provider implements. Host applications depend on the
protocol, not on a concrete provider, so providers can be swapped
through the registry in :mod:`ragit.embeddings.registry`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Callable embedding function.

    Implementations embed batches of texts and expose the
    ``embed_query`` / ``embed_documents`` convenience surface used by
    the retrieval and ingestion pipelines.
    """

    async def __call__(self, input: list[str]) -> list[list[float]]: ...

    async def embed_query(self, input: str | list[str]) -> list[float]: ...

    async def embed_documents(self, input: list[str]) -> list[list[float]]: ...
