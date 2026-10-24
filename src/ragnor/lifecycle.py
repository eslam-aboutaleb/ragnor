"""Compatibility import for :mod:`ragnor.core.lifecycle`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import lifecycle as _implementation

if TYPE_CHECKING:
    from ragnor.core.lifecycle import (
        AsyncCloseable,
        Protocol,
        runtime_checkable,
    )

__all__ = [
    "AsyncCloseable",
    "Protocol",
    "runtime_checkable",
]

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
