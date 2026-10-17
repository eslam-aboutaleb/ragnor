"""Unit tests for tool input validation and security sanitization."""

from __future__ import annotations

import pytest

from ragit.tools.validation import (
    SecurityValidationError,
    validate_safe_filters,
    validate_safe_identifier,
    validate_safe_string,
    validate_safe_uri,
)


def test_validate_safe_string_valid() -> None:
    validate_safe_string("safe query text", "query")
    validate_safe_string("hello world", "text", max_length=50)


def test_validate_safe_string_non_string() -> None:
    with pytest.raises(SecurityValidationError, match="must be a string"):
        validate_safe_string(123, "field")  # type: ignore[arg-type]


def test_validate_safe_string_length_exceeded() -> None:
    with pytest.raises(SecurityValidationError, match="exceeds maximum allowed length"):
        validate_safe_string("a" * 20, "field", max_length=10)


def test_validate_safe_string_null_byte() -> None:
    with pytest.raises(SecurityValidationError, match="contains prohibited null byte"):
        validate_safe_string("hello\0world", "field")


@pytest.mark.parametrize(
    "secret",
    [
        "bearer abcdefghijklmnopqrstuvwxyz",
        "-----BEGIN RSA PRIVATE KEY-----",
        "sk-123456789012345678901234",
        "ghp_123456789012345678901234",
        "glpat-12345678901234567890",
        "xoxb-123456789012-123456789012-abcdefghij",
        "password = 'mysecretpassword'",
        "api_key: 'supersecretkey123'",
    ],
)
def test_validate_safe_string_credentials(secret: str) -> None:
    with pytest.raises(SecurityValidationError, match="prohibited credential or secret pattern"):
        validate_safe_string(secret, "field")


@pytest.mark.parametrize(
    "path",
    [
        "../traversal",
        "..\\traversal",
        "/etc/passwd",
        "/var/log",
        "/usr/bin",
        "/root/.ssh",
        "C:\\Windows",
        "~/secrets",
        "file:///etc/hosts",
    ],
)
def test_validate_safe_string_path_traversal(path: str) -> None:
    with pytest.raises(SecurityValidationError, match="prohibited arbitrary filesystem path"):
        validate_safe_string(path, "path")

    # Allowed when explicitly set
    validate_safe_string(path, "path", allow_path_traversal=True)


def test_validate_safe_identifier() -> None:
    validate_safe_identifier("default", "collection")
    validate_safe_identifier("coll-123_abc.v1", "collection")

    with pytest.raises(SecurityValidationError, match="must be an alphanumeric identifier"):
        validate_safe_identifier("coll/sub", "collection")

    with pytest.raises(SecurityValidationError, match="must be an alphanumeric identifier"):
        validate_safe_identifier("coll..sub", "collection")

    with pytest.raises(SecurityValidationError, match="must be an alphanumeric identifier"):
        validate_safe_identifier("", "collection")


def test_validate_safe_uri() -> None:
    validate_safe_uri(None)
    validate_safe_uri("doc:manual-section-1")
    validate_safe_uri("urn:isbn:0451450523")
    validate_safe_uri("https://example.com/docs/item")
    validate_safe_uri("http://example.org/resource")

    with pytest.raises(SecurityValidationError, match="prohibited scheme"):
        validate_safe_uri("ftp://ftp.example.com/file")

    with pytest.raises(SecurityValidationError, match="missing a valid hostname"):
        validate_safe_uri("https:///no-host")

    with pytest.raises(SecurityValidationError, match="prohibited internal or link-local host"):
        validate_safe_uri("https://localhost/api")

    with pytest.raises(SecurityValidationError, match="prohibited internal or link-local host"):
        validate_safe_uri("http://127.0.0.1:8080/metrics")

    with pytest.raises(SecurityValidationError, match="prohibited internal or link-local host"):
        validate_safe_uri("http://169.254.169.254/latest/meta-data/")

    with pytest.raises(SecurityValidationError, match="prohibited internal or link-local host"):
        validate_safe_uri("http://service.internal/status")

    with pytest.raises(SecurityValidationError, match="prohibited private network address"):
        validate_safe_uri("http://10.0.1.5/admin")

    with pytest.raises(SecurityValidationError, match="prohibited private network address"):
        validate_safe_uri("https://192.168.1.100/router")

    with pytest.raises(SecurityValidationError, match="prohibited private network address"):
        validate_safe_uri("http://172.16.0.2/intranet")


def test_validate_safe_filters() -> None:
    validate_safe_filters(None)
    validate_safe_filters(
        {
            "category": "manual",
            "version": 2,
            "score": 0.95,
            "active": True,
            "tags": ["v1", "core"],
        }
    )

    with pytest.raises(SecurityValidationError, match="filters must be a dict"):
        validate_safe_filters(["category", "manual"])  # type: ignore[arg-type]

    with pytest.raises(SecurityValidationError, match="filter key 'invalid;key'"):
        validate_safe_filters({"invalid;key": "value"})

    with pytest.raises(SecurityValidationError, match="prohibited SQL injection characters"):
        validate_safe_filters({"category": "value; DROP TABLE users; --"})

    with pytest.raises(SecurityValidationError, match="prohibited SQL syntax"):
        validate_safe_filters({"tags": ["normal", "UNION SELECT * FROM accounts"]})
