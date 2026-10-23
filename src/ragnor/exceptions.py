"""Consistent SDK error taxonomy, retryability, and fail-closed semantics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_DSN_PATTERN = re.compile(r"://([^:@\s]+):([^@\s]+)@")
_KEY_VALUE_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|bearer)\b\s*[:= ]\s*([^\s,;'\"]+)"),
]


def sanitize_message(msg: str) -> str:
    """Redact sensitive credentials, tokens, and DSNs from error messages."""
    if not msg:
        return msg
    # Redact credentials in connection strings
    msg = _DSN_PATTERN.sub(r"://\1:***@", msg)
    # Redact key-value secrets
    for pat in _KEY_VALUE_PATTERNS:
        msg = pat.sub(r"\1=***", msg)
    return msg


class RagnorError(Exception):
    """Base exception for all Ragnor SDK domain errors.

    Attributes:
        code: Machine-readable error code.
        retryable: Whether this error can be safely retried with backoff.
        details: Sanitized contextual key-value diagnostics.
    """

    default_code: str = "ERR_RAGNOR"
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        sanitized = sanitize_message(str(message))
        super().__init__(sanitized)
        self.message = sanitized
        self.code = code or self.default_code
        self.retryable = self.default_retryable if retryable is None else retryable
        clean_details: dict[str, Any] = {}
        if details:
            for k, v in details.items():
                if isinstance(v, str):
                    clean_details[k] = sanitize_message(v)
                else:
                    clean_details[k] = v
        self.details = clean_details

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"code={self.code!r}, retryable={self.retryable!r})"
        )


class ConfigurationError(RagnorError):
    """Raised when SDK configuration, options, or environment parameters are invalid."""

    default_code = "ERR_CONFIG_INVALID"
    default_retryable = False


class BackendUnavailableError(RagnorError):
    """Raised when an external vector store, database, or network backend is unreachable."""

    default_code = "ERR_BACKEND_UNAVAILABLE"
    default_retryable = True


class EmbeddingProviderError(RagnorError):
    """Raised when an embedding provider fails during vector generation."""

    default_code = "ERR_EMBEDDING_FAILED"
    default_retryable = False


class IndexConsistencyError(RagnorError):
    """Raised when version, snapshot, or lease validation detects an inconsistency."""

    default_code = "ERR_INDEX_INCONSISTENT"
    default_retryable = False


class UnsupportedCapabilityError(RagnorError):
    """Raised when a requested feature or constraint is not supported by the backend."""

    default_code = "ERR_UNSUPPORTED_CAPABILITY"
    default_retryable = False


class ScopeDeniedError(RagnorError):
    """Raised when access to a tenant, document scope, or partition is denied."""

    default_code = "ERR_SCOPE_DENIED"
    default_retryable = False


class OperationTimeoutError(RagnorError):
    """Raised when an SDK operation times out while waiting on an external dependency."""

    default_code = "ERR_OPERATION_TIMEOUT"
    default_retryable = True
