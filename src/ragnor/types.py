"""Shared value types for ragnor.

Dataclasses are the canonical boundary across chunking, embedding,
storage, and retrieval. They replace the raw ``dict[str, Any]``
plumbing used by earlier iterations; the dict-based store API keeps
working during the transition, and later plans adapt it to these
types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text fragment produced by a chunker."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document with an optional embedding vector."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A single hit from a (hybrid) search.

    Mirrors the dict keys the pgvector store's hybrid search returns
    today: ``document``, ``metadata``, ``distance``,
    ``keyword_score``, ``_rrf_score``, and ``id``. ``distance`` is
    ``None`` for keyword-only hits and ``keyword_score`` is ``None``
    for vector-only hits.
    """

    id: str | None
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0


class RetrievalError(Exception):
    """Typed error for retrieval failures (used by strict-mode search)."""
