"""Compatibility import for :mod:`ragnor.core.registry`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import registry as _implementation

if TYPE_CHECKING:
    from ragnor.core.registry import (
        Callable,
        Generic,
        Registry,
        T,
        TypeVar,
    )

__all__ = [
    "Callable",
    "Generic",
    "Registry",
    "T",
    "TypeVar",
]

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
