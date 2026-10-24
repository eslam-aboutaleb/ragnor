"""Compatibility import for :mod:`ragnor.core.lifecycle`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import lifecycle as _implementation

if TYPE_CHECKING:
    from ragnor.core.lifecycle import (
        AsyncCloseable as AsyncCloseable,
        Protocol as Protocol,
        runtime_checkable as runtime_checkable,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
