"""Compatibility import for :mod:`ragnor.core.registry`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import registry as _implementation

if TYPE_CHECKING:
    from ragnor.core.registry import (
        Callable as Callable,
        Generic as Generic,
        Registry as Registry,
        T as T,
        TypeVar as TypeVar,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
