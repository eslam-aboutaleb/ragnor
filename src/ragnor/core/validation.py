"""Input validation helpers shared across ragnor providers.

The stores rely on these validators: SQL identifiers are
checked against a strict allowlist before interpolation, and
embedding vectors are checked for the expected dimension and
finite values before they reach the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import re
from typing import Any

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_identifier(name: str, label: str) -> None:
    """Validate that a string is a safe SQL identifier (table, column, etc.).

    Only allows alphanumeric characters and underscores, starting with a
    letter or underscore. This is a strict whitelist that prevents SQL
    injection through identifier interpolation.

    Args:
        name: The identifier string to validate.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the identifier contains unsafe characters.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(name):
        raise ValueError(f"Unsafe {label}: {name!r}")


def validate_embedding(
    embedding: list[float],
    expected_dim: int,
    label: str = "embedding",
) -> None:
    """Validate that an embedding vector has the expected dimension and contains only finite values.

    Args:
        embedding: The embedding vector to validate.
        expected_dim: Expected dimensionality of the embedding.
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the embedding has the wrong dimension or contains non-finite values.
        TypeError: If a value in the embedding is not a number.
    """
    if len(embedding) != expected_dim:
        raise ValueError(f"{label} has {len(embedding)} dimensions; expected {expected_dim}")
    for i, value in enumerate(embedding):
        try:
            if not math.isfinite(value):
                raise ValueError(f"{label}[{i}] is not finite: {value!r}")
        except TypeError as exc:
            raise TypeError(f"{label}[{i}] is not a number: {value!r}") from exc


def validate_upsert_identifiers(
    text_field: str,
    embedding_field: str,
    id_field: str | None = None,
    default_id_field: str = "id",
    extra_fields: Iterable[str] | None = None,
) -> str:
    """Validate identifiers used in upsert operations and return the resolved id field."""
    validate_identifier(text_field, "text_field")
    validate_identifier(embedding_field, "embedding_field")
    resolved_id = id_field if id_field is not None else default_id_field
    validate_identifier(resolved_id, "id_field")
    if extra_fields:
        for key in extra_fields:
            validate_identifier(key, "extra_field")
    return resolved_id


def validate_search_identifiers(
    text_field: str,
    filters: Iterable[str] | None = None,
    metadata_fields: Iterable[str] | None = None,
) -> None:
    """Validate identifiers used in search operations."""
    validate_identifier(text_field, "text_field")
    if filters:
        for key in filters:
            validate_identifier(key, "filter key")
    if metadata_fields:
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")


def validate_delete_selectors(
    ids: Sequence[Any] | None,
    filters: Mapping[str, Any] | None,
) -> None:
    """Validate deletion selectors (ids and/or filters)."""
    if ids is None and filters is None:
        raise ValueError("delete requires at least one of ids or filters")
    if ids is not None and len(ids) == 0:
        raise ValueError("ids must not be empty; an empty id list never means delete everything")
    if filters is not None:
        for key in filters:
            validate_identifier(key, "filter key")
