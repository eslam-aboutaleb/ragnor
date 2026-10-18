"""Unit tests for ragnor.validation."""

from __future__ import annotations

import math

import pytest

from ragnor.validation import validate_embedding, validate_identifier


def test_validate_identifier_accepts_safe_identifiers() -> None:
    for name in ("id", "_private", "table_1", "A", "a" * 100):
        validate_identifier(name, "column")


def test_validate_identifier_rejects_unsafe_identifiers() -> None:
    unsafe = ("", "1abc", "a-b", "a b", "a;b", "a.b", "drop table", 'a"b', "a'b")
    for name in unsafe:
        with pytest.raises(ValueError, match="Unsafe column"):
            validate_identifier(name, "column")


def test_validate_embedding_accepts_valid_embedding() -> None:
    validate_embedding([1.0, -2.5, 0.0], 3)


def test_validate_embedding_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="has 2 dimensions; expected 3"):
        validate_embedding([1.0, 2.0], 3)


def test_validate_embedding_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match=r"embedding\[1\] is not finite"):
        validate_embedding([1.0, math.inf], 2)
    with pytest.raises(ValueError, match="is not finite"):
        validate_embedding([math.nan], 1)


def test_validate_embedding_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError, match="is not a number"):
        validate_embedding(["nope"], 1)
