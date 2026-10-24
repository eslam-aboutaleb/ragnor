"""Compatibility import for :mod:`ragnor.core.validation`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import validation as _implementation

if TYPE_CHECKING:
    from ragnor.core.validation import (
        Any as Any,
        Iterable as Iterable,
        Mapping as Mapping,
        Sequence as Sequence,
        math as math,
        re as re,
        validate_delete_selectors as validate_delete_selectors,
        validate_embedding as validate_embedding,
        validate_identifier as validate_identifier,
        validate_search_identifiers as validate_search_identifiers,
        validate_upsert_identifiers as validate_upsert_identifiers,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
