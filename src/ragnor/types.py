"""Compatibility import for :mod:`ragnor.core.types`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import types as _implementation

if TYPE_CHECKING:
    from ragnor.core.types import (
        Chunk as Chunk,
        Document as Document,
        RagnorError as RagnorError,
        RetrievalError as RetrievalError,
        SearchResult as SearchResult,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
