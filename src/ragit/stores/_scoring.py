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
from typing import Any

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


def fuse_hybrid(  # noqa: PLR0913
    vector_results: Sequence[dict[str, Any]],
    keyword_results: Sequence[dict[str, Any]],
    n_results: int,
) -> list[dict[str, Any]]:
    """Merge the two truncated stage results into hybrid results.

    Both stages are already ranked and truncated to
    ``n_results`` by their capability methods; this function
    merges the two result lists by document id via
    :func:`rrf_merge`, preserving ``distance`` from the
    vector row and ``keyword_score`` from the keyword row.
    A document found by both stages outranks a single-stage
    hit; ties break by document id.

    Args:
        vector_results: The vector stage's result dicts
            (``id``, ``document``, ``metadata``,
            ``distance``, ``keyword_score``), best first.
        keyword_results: The keyword stage's result dicts
            (same keys), best first.
        n_results: Maximum number of merged results to
            return.

    Returns:
        At most ``n_results`` result dicts sorted by RRF
        score, each carrying ``id``, ``document``,
        ``metadata``, ``distance`` (``None`` for
        keyword-only hits), ``keyword_score`` (``None`` for
        vector-only hits), and ``_rrf_score``.
    """
    vector_ranked = [row["id"] for row in vector_results]
    keyword_ranked = [row["id"] for row in keyword_results]
    vector_by_id = {row["id"]: row for row in vector_results}
    keyword_by_id = {row["id"]: row for row in keyword_results}

    results: list[dict[str, Any]] = []
    for doc_id, rrf_score in rrf_merge(vector_ranked, keyword_ranked, n_results):
        vector_row = vector_by_id.get(doc_id)
        keyword_row = keyword_by_id.get(doc_id)
        source = vector_row if vector_row is not None else keyword_row
        results.append(
            {
                "id": doc_id,
                "document": source.get("document"),
                "metadata": source.get("metadata") or {},
                "distance": (
                    vector_row.get("distance") if vector_row is not None else None
                ),
                "keyword_score": (
                    keyword_row.get("keyword_score")
                    if keyword_row is not None
                    else None
                ),
                "_rrf_score": rrf_score,
            }
        )
    return results
