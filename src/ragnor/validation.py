"""Compatibility import for :mod:`ragnor.core.validation`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import validation as _implementation

if TYPE_CHECKING:
    from ragnor.core.validation import (
        Any,
        Iterable,
        Mapping,
        Sequence,
        math,
        re,
        validate_delete_selectors,
        validate_embedding,
        validate_identifier,
        validate_search_identifiers,
        validate_upsert_identifiers,
    )

__all__ = [
    "Any",
    "Iterable",
    "Mapping",
    "Sequence",
    "math",
    "re",
    "validate_delete_selectors",
    "validate_embedding",
    "validate_identifier",
    "validate_search_identifiers",
    "validate_upsert_identifiers",
]

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
