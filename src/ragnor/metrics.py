"""Compatibility import for :mod:`ragnor.stores.metrics`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.stores import metrics as _implementation

if TYPE_CHECKING:
    from ragnor.stores.metrics import (
        Any,
        DEFAULT_DISTANCE_METRIC,
        DistanceMetric,
        Literal,
        SUPPORTED_DISTANCE_METRICS,
        Sequence,
        cosine_distance,
        cosine_similarity,
        math,
        normalize,
        resolve_distance_metric,
    )

__all__ = [
    "Any",
    "DEFAULT_DISTANCE_METRIC",
    "DistanceMetric",
    "Literal",
    "SUPPORTED_DISTANCE_METRICS",
    "Sequence",
    "cosine_distance",
    "cosine_similarity",
    "math",
    "normalize",
    "resolve_distance_metric",
]

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
