"""Compatibility import for :mod:`ragnor.stores.metrics`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.stores import metrics as _implementation

if TYPE_CHECKING:
    from ragnor.stores.metrics import (
        Any as Any,
        DEFAULT_DISTANCE_METRIC as DEFAULT_DISTANCE_METRIC,
        DistanceMetric as DistanceMetric,
        Literal as Literal,
        SUPPORTED_DISTANCE_METRICS as SUPPORTED_DISTANCE_METRICS,
        Sequence as Sequence,
        cosine_distance as cosine_distance,
        cosine_similarity as cosine_similarity,
        math as math,
        normalize as normalize,
        resolve_distance_metric as resolve_distance_metric,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
