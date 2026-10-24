"""Distance metrics for the vector stores.

A store's distance metric decides how the vector stage scores
documents: ``l2`` (Euclidean distance) is ragnor's historical
default, and ``cosine`` (``1 - cosine_similarity``) is an opt-in
for hosts whose indexes and thresholds are cosine end-to-end.

Thresholds are metric-relative: a threshold tuned for one metric
means something different under another, so a host that switches
metrics must re-tune ``rag_distance_threshold``.

``rag_distance_metric`` is an OPTIONAL host settings field, read
through ``getattr`` exactly like the other optional host fields
(``embedding_provider``, ``llm_model``). ragnor never adds it to
the :class:`ragnor.core.config.RagSettings` protocol, so a host settings
object keeps satisfying the protocol whether or not it declares
the field.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

DistanceMetric = Literal["l2", "cosine"]
"""Distance metric a store ranks by (``l2`` or ``cosine``)."""

DEFAULT_DISTANCE_METRIC: DistanceMetric = "l2"
"""Metric used when neither an argument nor the host settings pick one."""

SUPPORTED_DISTANCE_METRICS: tuple[DistanceMetric, ...] = ("l2", "cosine")
"""Every metric the stores accept."""


def resolve_distance_metric(metric: str | None = None, settings: Any = None) -> DistanceMetric:
    """Resolve the distance metric from an argument or host settings.

    Resolution order: the explicit ``metric`` argument, then the
    optional ``rag_distance_metric`` host settings field, then
    :data:`DEFAULT_DISTANCE_METRIC`. Values are normalized
    case-insensitively (mirroring ragnor's provider registries) and
    surrounding whitespace is ignored.

    Args:
        metric: Requested metric, or ``None`` to fall back to the
            host settings.
        settings: Host settings object. The optional
            ``rag_distance_metric`` field is read with ``getattr``,
            so a settings object that does not declare it keeps
            working; ``None`` is accepted.

    Returns:
        The resolved :class:`DistanceMetric`.

    Raises:
        ValueError: If the resolved value is not a supported
            metric. The message lists the supported names.
    """
    if metric is None and settings is not None:
        metric = getattr(settings, "rag_distance_metric", None)
    if metric is None:
        return DEFAULT_DISTANCE_METRIC
    normalized = str(metric).strip().lower()
    for supported in SUPPORTED_DISTANCE_METRICS:
        if normalized == supported:
            return supported
    supported_names = ", ".join(SUPPORTED_DISTANCE_METRICS)
    raise ValueError(
        f"Unknown distance metric {metric!r}; supported metrics are: {supported_names}"
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors.

    The similarity is the dot product divided by the product of
    the vector norms, in ``[-1.0, 1.0]``. A zero-norm vector has
    no direction, so the similarity is defined as ``0.0``.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The cosine similarity of the two vectors.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(left) != len(right):
        raise ValueError(f"vectors must have the same length; got {len(left)} and {len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine distance between two vectors.

    The distance is ``1.0 - cosine_similarity(left, right)``, in
    ``[0.0, 2.0]``: identical vectors score ``0.0``, orthogonal
    vectors score ``1.0``, and opposite vectors score ``2.0``.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The cosine distance between the two vectors.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    return 1.0 - cosine_similarity(left, right)


def normalize(vector: Sequence[float]) -> list[float]:
    """Return the unit-length form of ``vector``.

    A zero vector has no direction to normalize to, so it is
    returned unchanged.

    Args:
        vector: Vector to scale to unit length.

    Returns:
        A new list holding the unit vector (``[0.0, ...]`` for a
        zero vector). The input is not modified.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]
