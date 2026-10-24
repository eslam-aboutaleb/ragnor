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
from collections.abc import Mapping, Sequence
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
            format_search_hit(
                doc_id=doc_id,
                document=str(source.get("document", "")),
                metadata=source.get("metadata") or {},
                distance=vector_row.get("distance") if vector_row is not None else None,
                keyword_score=(
                    keyword_row.get("keyword_score") if keyword_row is not None else None
                ),
                rrf_score=rrf_score,
            )
        )
    return results


def project_metadata(
    record: dict[str, Any],
    metadata_fields: Sequence[str] | None,
) -> dict[str, Any]:
    """Project ``metadata_fields`` out of ``record``."""
    if not metadata_fields:
        return {}
    return {
        field: record[field]
        for field in metadata_fields
        if field in record and record[field] is not None
    }


def format_search_hit(  # noqa: PLR0913
    doc_id: str,
    document: str,
    metadata: dict[str, Any],
    *,
    distance: float | None = None,
    keyword_score: float | None = None,
    rrf_score: float | None = None,
) -> dict[str, Any]:
    """Construct a standard search hit dictionary across store backends."""
    hit: dict[str, Any] = {
        "id": doc_id,
        "document": document,
        "metadata": metadata,
        "distance": distance,
        "keyword_score": keyword_score,
    }
    if rrf_score is not None:
        hit["_rrf_score"] = rrf_score
    return hit


def filter_record_map(
    records: Mapping[str, dict[str, Any]],
    *,
    ids: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    text_field: str = "text",
    metadata_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter in-memory records by id and equality filters."""
    id_set = set(ids) if ids is not None else None
    results = []
    for doc_id, record in records.items():
        if id_set is not None and doc_id not in id_set:
            continue
        if filters:
            match = True
            for key, value in filters.items():
                record_val = record.get(key)
                if isinstance(value, (list, tuple, set, frozenset)):
                    if record_val not in value:
                        match = False
                        break
                elif record_val != value:
                    match = False
                    break
            if not match:
                continue
        results.append(
            {
                "id": str(doc_id),
                "document": record.get(text_field),
                "metadata": project_metadata(record, metadata_fields),
            }
        )
    return results


def build_upsert_record(
    doc: dict[str, Any],
    id_field: str,
    text_field: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a stored record dict from an input document."""
    doc_id = doc.get("id")
    record: dict[str, Any] = {
        id_field: doc_id,
        text_field: doc[text_field],
    }
    for key, value in doc.get("metadata", {}).items():
        record[key] = value
    if extra_fields:
        record.update(extra_fields)
    return record


def validate_upsert_documents(  # noqa: PLR0913, PLR0917
    documents: list[dict[str, Any]],
    text_field: str,
    embedding_field: str,
    id_field: str | None,
    default_id_field: str,
    extra_fields: list[str] | None = None,
    dimension: int | None = None,
    metric: Any = None,
) -> str:
    """Validate identifiers and embeddings for an upsert operation."""
    from ragnor.core.validation import (  # noqa: PLC0415
        validate_embedding,
        validate_identifier,
        validate_upsert_identifiers,
    )

    effective_id = validate_upsert_identifiers(
        text_field, embedding_field, id_field, default_id_field, extra_fields
    )
    if documents:
        for doc in documents:
            if dimension is not None:
                validate_embedding(
                    doc[embedding_field],
                    expected_dim=dimension,
                    label="document embedding",
                )
            for key in doc.get("metadata", {}):
                validate_identifier(key, "metadata key")
    return effective_id


def handle_count_error(
    table_name: str,
    exc: Exception,
    strict: bool,
    logger: Any,
) -> int:
    """Log count failure and raise RetrievalError if strict, else return 0."""
    from ragnor.core.exceptions import BackendUnavailableError, OperationTimeoutError  # noqa: PLC0415

    logger.error("Count failed on %s: %s", table_name, type(exc).__name__)
    if strict:
        if isinstance(exc, TimeoutError):
            raise OperationTimeoutError(
                f"Count timed out on {table_name} ({type(exc).__name__})"
            ) from exc
        raise BackendUnavailableError(
            f"Count failed on {table_name} ({type(exc).__name__})"
        ) from exc
    return 0


def validate_delete_args(ids: Sequence[str] | None, filters: Mapping[str, Any] | None) -> bool:
    """Validate delete arguments. Return False if ids is empty list (no-op)."""
    if ids is None and filters is None:
        raise ValueError("delete requires at least one of ids or filters")
    if ids is not None:
        if len(ids) == 0:
            return False
        for item in ids:
            if not isinstance(item, str):
                raise TypeError(f"All ids must be strings; got {type(item).__name__}")
    return True


def score_records_overlap(
    records: Mapping[str, dict[str, Any]],
    query: str,
    text_field: str,
    filters: Mapping[str, Any] | None,
    matches_fn: Any,
) -> dict[str, float]:
    """Calculate token overlap keyword scores across filtered records."""
    query_tokens = set(tokenize(query))
    scores: dict[str, float] = {}
    for doc_id, record in records.items():
        if filters and not matches_fn(record, filters):
            continue
        score = token_overlap(query_tokens, tokenize(str(record.get(text_field, ""))))
        if score > 0.0:
            scores[doc_id] = score
    return scores
