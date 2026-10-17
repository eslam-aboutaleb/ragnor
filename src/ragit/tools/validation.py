"""Input validation and security sanitization for tool bridges and adapters.

Prevents arbitrary filesystem paths, credentials/secrets, and unrestricted
fetch URLs from reaching internal retrieval, storage, or execution engines.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# Pattern matching potential credentials or API keys
_CREDENTIAL_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[a-zA-Z0-9_\-\.]{15,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(ghp_[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(glpat-[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(xoxb-[a-zA-Z0-9-]{20,})\b"),
    re.compile(r"(?i)\b(password|passwd|secret_key|api_key)\s*[:=]\s*['\"]?[^\s'\"]{6,}"),
]

# Patterns detecting path traversal or arbitrary filesystem paths
_PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\.[/\\]"),
    re.compile(r"^[/\\]etc[/\\]"),
    re.compile(r"^[/\\]var[/\\]"),
    re.compile(r"^[/\\]usr[/\\]"),
    re.compile(r"^[/\\]root"),
    re.compile(r"^[A-Za-z]:[/\\]"),  # Windows drive letter C:\
    re.compile(r"^~[/\\]"),
    re.compile(r"(?i)^file://"),
]

# Prohibited SSRF hostnames and IP ranges
_PROHIBITED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # noqa: S104
    "::1",
    "169.254.169.254",  # AWS/GCP/Azure link-local metadata service
    "metadata.google.internal",
}

_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
_SQL_INJECTION_PATTERN = re.compile(r"(?i)(--|;|\bUNION\b|\bDROP\b|\bDELETE\b|\bTRUNCATE\b|/\*)")


class SecurityValidationError(ValueError):
    """Raised when tool arguments contain unsafe paths, credentials, or URLs."""


def validate_safe_string(
    val: str,
    field_name: str,
    *,
    allow_path_traversal: bool = False,
    max_length: int = 10000,
) -> None:
    """Validate that a string contains no null bytes, path traversal, or credentials.

    Args:
        val: String value to inspect.
        field_name: Name of the argument or field for error reporting.
        allow_path_traversal: If True, skips path traversal checks.
        max_length: Maximum allowed character length.

    Raises:
        SecurityValidationError: If the value violates security constraints.
    """
    if not isinstance(val, str):
        raise SecurityValidationError(f"{field_name} must be a string, got {type(val).__name__}")

    if len(val) > max_length:
        raise SecurityValidationError(
            f"{field_name} exceeds maximum allowed length of {max_length} characters"
        )

    if "\0" in val:
        raise SecurityValidationError(f"{field_name} contains prohibited null byte")

    # Check for credentials
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(val):
            raise SecurityValidationError(
                f"{field_name} contains prohibited credential or secret pattern"
            )

    # Check for path traversal if not explicitly allowed
    if not allow_path_traversal:
        for pattern in _PATH_TRAVERSAL_PATTERNS:
            if pattern.search(val):
                raise SecurityValidationError(
                    f"{field_name} contains prohibited arbitrary filesystem path or traversal"
                )


def validate_safe_identifier(val: str, field_name: str) -> None:
    """Validate that an identifier contains only alphanumeric, dash, dot, or underscore.

    Prevents SQL injection, path traversal, and command injection in collection/document IDs.

    Args:
        val: Identifier to validate.
        field_name: Field name for error reporting.

    Raises:
        SecurityValidationError: If identifier contains unsafe characters.
    """
    validate_safe_string(val, field_name, max_length=128)
    if not _SAFE_IDENTIFIER_PATTERN.match(val) or ".." in val:
        raise SecurityValidationError(
            f"{field_name} must be an alphanumeric identifier (1-128 chars, [a-zA-Z0-9_.-]); "
            f"got {val!r}"
        )


def validate_safe_uri(uri: str | None, field_name: str = "uri") -> None:
    """Validate that a URI uses an approved scheme and does not target internal services.

    Args:
        uri: The URI string to inspect.
        field_name: Argument name for error reporting.

    Raises:
        SecurityValidationError: If scheme is unsafe or target is an internal SSRF host.
    """
    if uri is None:
        return

    validate_safe_string(uri, field_name, max_length=2048, allow_path_traversal=False)

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if not scheme:
        # Bare safe identifier or relative safe ref
        return

    if scheme not in ("https", "http", "urn", "doc"):
        raise SecurityValidationError(
            f"{field_name} specifies prohibited scheme {scheme!r}; "
            "only https, http, urn, and doc schemes are permitted"
        )

    if scheme in ("http", "https"):
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise SecurityValidationError(f"{field_name} is missing a valid hostname")

        if hostname in _PROHIBITED_HOSTS or hostname.endswith(".internal"):
            raise SecurityValidationError(
                f"{field_name} targets prohibited internal or link-local host {hostname!r}"
            )

        # Check private IP ranges
        if hostname.startswith("10.") or hostname.startswith("192.168."):
            raise SecurityValidationError(
                f"{field_name} targets prohibited private network address {hostname!r}"
            )
        if re.match(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", hostname):
            raise SecurityValidationError(
                f"{field_name} targets prohibited private network address {hostname!r}"
            )


def validate_safe_filters(filters: dict[str, Any] | None) -> None:
    """Validate filter dictionary keys and values.

    Args:
        filters: Filter key-value mapping.

    Raises:
        SecurityValidationError: If any key or value is unsafe.
    """
    if filters is None:
        return

    if not isinstance(filters, dict):
        raise SecurityValidationError(f"filters must be a dict, got {type(filters).__name__}")

    for k, v in filters.items():
        validate_safe_identifier(str(k), f"filter key {k!r}")
        if isinstance(v, str):
            validate_safe_string(v, f"filter value for {k!r}", max_length=1000)
            if _SQL_INJECTION_PATTERN.search(v):
                raise SecurityValidationError(
                    f"filter value for {k!r} contains prohibited SQL injection characters"
                )
        elif isinstance(v, (int, float, bool)):
            continue
        elif isinstance(v, (list, tuple, set)):
            for item in v:
                if isinstance(item, str):
                    validate_safe_string(item, f"filter item in {k!r}", max_length=1000)
                    if _SQL_INJECTION_PATTERN.search(item):
                        raise SecurityValidationError(
                            f"filter list item for {k!r} contains prohibited SQL syntax"
                        )
