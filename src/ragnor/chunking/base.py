"""Chunker seam for ragnor.

The :class:`Chunker` protocol is the boundary every chunking
strategy implements. ``chunk`` produces :class:`ragnor.core.types.Chunk`
objects; ``version`` feeds the ingestion snapshot comparison
that makes ingestion idempotent; ``config`` exposes the
chunker's tunables (chunk size, overlap) for the same
snapshot.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ragnor.core.types import Chunk


@runtime_checkable
class Chunker(Protocol):
    """Text chunking strategy."""

    def chunk(self, text: str) -> list[Chunk]: ...

    @property
    def version(self) -> str:
        """Version tag feeding ingestion snapshot comparison."""
        ...

    @property
    def config(self) -> dict[str, Any]:
        """Chunker tunables (chunk_size/overlap) for snapshots."""
        ...
