"""Shared scoring helpers for vector store backends.

The hybrid-search contract ranks two independent stages -- a
dense vector stage and a sparse keyword stage -- and merges
them with Reciprocal Rank Fusion. Backends without a native
full-text search (everything except PostgreSQL) reuse these
token-overlap and RRF helpers so their result shape and
ranking stay comparable to the pgvector backend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Return the lowercase alphanumeric tokens of ``text``."""
    return _TOKEN_PATTERN.findall(text.lower())


def l2_distance(left: list[float], right: list[float]) -> float:
    """Return the L2 distance between two vectors.

    Vectors of different lengths are compared positionally up to
    the shorter length; stores validate dimensions up front
    whenever ``embedding_dim`` is configured.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=False)))


def token_overlap(query_tokens: set[str], text_tokens: list[str]) -> float:
    """Return the fraction of query tokens present in ``text_tokens``."""
    if not query_tokens:
        return 0.0
    present = set(text_tokens)
    return len(query_tokens & present) / len(query_tokens)


def rrf_merge(  # noqa: PLR0913
    vector_ranked: Sequence[str],
    keyword_ranked: Sequence[str],
    n_results: int,
) -> list[tuple[str, float]]:
    """Merge two ranked document-id lists with Reciprocal Rank Fusion.

    Both stages rank independently; a document's score is the sum
    of ``1 / (60 + rank)`` over the stages that found it (1-based
    ranks). Ties break by document id so merges are deterministic.

    Args:
        vector_ranked: Document ids ranked by vector similarity
            (best first).
        keyword_ranked: Document ids ranked by keyword score
            (best first).
        n_results: Maximum number of merged results to return.

    Returns:
        At most ``n_results`` ``(doc_id, rrf_score)`` pairs
        sorted by score descending.
    """
    vector_ranks = {doc_id: rank for rank, doc_id in enumerate(vector_ranked, start=1)}
    keyword_ranks = {doc_id: rank for rank, doc_id in enumerate(keyword_ranked, start=1)}

    def _rrf(doc_id: str) -> float:
        score = 0.0
        if doc_id in vector_ranks:
            score += 1.0 / (60 + vector_ranks[doc_id])
        if doc_id in keyword_ranks:
            score += 1.0 / (60 + keyword_ranks[doc_id])
        return score

    merged = sorted(
        set(vector_ranks) | set(keyword_ranks),
        key=lambda doc_id: (-_rrf(doc_id), doc_id),
    )
    return [(doc_id, _rrf(doc_id)) for doc_id in merged[:n_results]]
