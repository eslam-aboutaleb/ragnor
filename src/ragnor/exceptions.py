"""Consistent SDK error taxonomy, retryability, and fail-closed semantics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_DSN_PATTERN = re.compile(r"://([^:@\s]+):([^@\s]+)@")
_AUTH_HEADER_PATTERN = re.compile(r"(?i)(authorization\s*[:=]\s*bearer)\s+([^\s,;'\"]+)")
_PROVIDER_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_KEY_VALUE_PATTERNS = [
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?key|private[_-]?key|token|secret|password|bearer)"
        r"\b\s*[:= ]\s*([^\s,;'\"]+)"
    ),
    re.compile(
        r"(?i)\b(document[_-]?text|raw[_-]?text|document[_-]?body|prompt[_-]?content|customer[_-]?text)"
        r"\b\s*[:=]\s*([^\n,;]+)"
    ),
]


def sanitize_message(msg: str) -> str:
    """Redact sensitive credentials, tokens, DSNs, and document snippets from error messages."""
    if not msg:
        return msg
    msg = _DSN_PATTERN.sub(r"://\1:***@", msg)
    msg = _AUTH_HEADER_PATTERN.sub(r"\1 ***", msg)
    msg = _PROVIDER_KEY_PATTERN.sub("sk-***", msg)
    for pat in _KEY_VALUE_PATTERNS:
        msg = pat.sub(r"\1=***", msg)
    return msg


_SENSITIVE_DETAIL_KEYS = frozenset(
    {
        "apikey",
        "accesskey",
        "privatekey",
        "token",
        "secret",
        "password",
        "authorization",
        "bearer",
        "documenttext",
        "rawtext",
        "documentbody",
        "promptcontent",
        "customertext",
    }
)


def _sanitize_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "***"
            if re.sub(r"[^a-z]", "", str(key).lower()) in _SENSITIVE_DETAIL_KEYS
            else _sanitize_detail(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_detail(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_detail(item) for item in value)
    return sanitize_message(value) if isinstance(value, str) else value


class RagnorError(Exception):
    """Base exception for all Ragnor SDK domain errors.

    Attributes:
        message: Sanitized human-readable message.
        code: Machine-readable error code.
        retryable: Whether this error can be safely retried with backoff.
        details: Sanitized contextual key-value diagnostics.
        trace_id: Optional caller-supplied correlation identifier.
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
        trace_id: str | None = None,
    ) -> None:
        sanitized = sanitize_message(str(message))
        super().__init__(sanitized)
        self.message = sanitized
        self.code = code or self.default_code
        self.retryable = self.default_retryable if retryable is None else retryable
        self.trace_id = sanitize_message(trace_id) if trace_id is not None else None
        clean_details: dict[str, Any] = _sanitize_detail(details or {})
        if self.trace_id is not None and "trace_id" not in clean_details:
            clean_details["trace_id"] = self.trace_id
        self.details = clean_details

    def __repr__(self) -> str:
        if self.trace_id is not None:
            return (
                f"{self.__class__.__name__}(message={self.message!r}, "
                f"code={self.code!r}, retryable={self.retryable!r}, "
                f"trace_id={self.trace_id!r})"
            )
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"code={self.code!r}, retryable={self.retryable!r})"
        )


class RetrievalError(RagnorError):
    """Typed error for retrieval failures (used by strict-mode search)."""

    default_code = "ERR_RETRIEVAL"
    default_retryable = True


class ConfigurationError(RagnorError, ValueError):
    """Raised when SDK configuration, options, or environment parameters are invalid."""

    default_code = "ERR_CONFIG_INVALID"
    default_retryable = False


class BackendUnavailableError(RetrievalError):
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


class ScopeDeniedError(RagnorError, PermissionError):
    """Raised when access to a tenant, document scope, or partition is denied."""

    default_code = "ERR_SCOPE_DENIED"
    default_retryable = False


class OperationTimeoutError(RetrievalError, TimeoutError):
    """Raised when an SDK operation times out while waiting on an external dependency."""

    default_code = "ERR_OPERATION_TIMEOUT"
    default_retryable = True


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    return None


def _attach_cause(err: RagnorError, cause: Exception) -> RagnorError:
    if err is not cause and err.__cause__ is None:
        err.__cause__ = cause
    return err


def classify_exception(  # noqa: PLR0911, PLR0912
    exc: Exception,
    *,
    default_stage: str = "operation",
    trace_id: str | None = None,
) -> RagnorError:
    """Classify an arbitrary exception into the SDK-03 RagnorError taxonomy.

    Never double-wraps an existing RagnorError. Distinguishes transient
    failures (timeouts, connection errors, HTTP 429/5xx) from permanent
    failures (auth 401/403, invalid configuration, capability/scope errors).
    """
    error_type = type(exc).__name__
    if isinstance(exc, RagnorError):
        if exc.trace_id is None and trace_id is not None:
            exc.trace_id = sanitize_message(trace_id)
            exc.details.setdefault("trace_id", exc.trace_id)
        return exc

    if isinstance(exc, TimeoutError):
        return _attach_cause(
            OperationTimeoutError(
                f"{default_stage} timed out ({error_type})",
                retryable=True,
                trace_id=trace_id,
            ),
            exc,
        )

    if isinstance(exc, PermissionError):
        return _attach_cause(
            ScopeDeniedError(
                f"{default_stage} permission denied ({error_type})",
                retryable=False,
                trace_id=trace_id,
            ),
            exc,
        )

    status = _extract_status_code(exc)
    if status is not None:
        if status in {401, 403}:
            cls: type[RagnorError] = (
                EmbeddingProviderError if default_stage == "embedding" else BackendUnavailableError
            )
            return _attach_cause(
                cls(
                    f"{default_stage} authentication failed (status {status}) ({error_type})",
                    code="ERR_AUTH_FAILED",
                    retryable=False,
                    details={"status_code": status},
                    trace_id=trace_id,
                ),
                exc,
            )
        if status == 429:
            cls = (
                EmbeddingProviderError if default_stage == "embedding" else BackendUnavailableError
            )
            return _attach_cause(
                cls(
                    f"{default_stage} rate limited (status 429) ({error_type})",
                    code="ERR_RATE_LIMITED",
                    retryable=True,
                    details={"status_code": 429},
                    trace_id=trace_id,
                ),
                exc,
            )
        if 500 <= status < 600:
            cls = (
                EmbeddingProviderError if default_stage == "embedding" else BackendUnavailableError
            )
            return _attach_cause(
                cls(
                    f"{default_stage} upstream server error (status {status}) ({error_type})",
                    code="ERR_UPSTREAM_5XX",
                    retryable=True,
                    details={"status_code": status},
                    trace_id=trace_id,
                ),
                exc,
            )
        if 400 <= status < 500:
            return _attach_cause(
                ConfigurationError(
                    f"{default_stage} rejected invalid request (status {status}) ({error_type})",
                    code="ERR_INVALID_REQUEST",
                    retryable=False,
                    details={"status_code": status},
                    trace_id=trace_id,
                ),
                exc,
            )

    if isinstance(exc, (ConnectionError, OSError)):
        return _attach_cause(
            BackendUnavailableError(
                f"{default_stage} backend connection failed ({error_type})",
                retryable=True,
                trace_id=trace_id,
            ),
            exc,
        )

    if isinstance(exc, (ValueError, TypeError)):
        return _attach_cause(
            ConfigurationError(
                f"{default_stage} configuration or validation error ({error_type})",
                retryable=False,
                trace_id=trace_id,
            ),
            exc,
        )

    lower_msg = str(exc).lower()
    if any(token in lower_msg for token in ("401", "403", "unauthorized", "invalid api key")):
        cls = EmbeddingProviderError if default_stage == "embedding" else BackendUnavailableError
        return _attach_cause(
            cls(
                f"{default_stage} authentication failed ({error_type})",
                code="ERR_AUTH_FAILED",
                retryable=False,
                trace_id=trace_id,
            ),
            exc,
        )
    if any(token in lower_msg for token in ("429", "rate limit", "too many requests")):
        cls = EmbeddingProviderError if default_stage == "embedding" else BackendUnavailableError
        return _attach_cause(
            cls(
                f"{default_stage} rate limited ({error_type})",
                code="ERR_RATE_LIMITED",
                retryable=True,
                trace_id=trace_id,
            ),
            exc,
        )

    if default_stage in {"retrieval", "backend", "search"}:
        return _attach_cause(
            BackendUnavailableError(
                f"{default_stage} failed ({error_type})",
                retryable=True,
                trace_id=trace_id,
            ),
            exc,
        )
    if default_stage == "embedding":
        return _attach_cause(
            EmbeddingProviderError(
                f"embedding failed ({error_type})",
                retryable=False,
                trace_id=trace_id,
            ),
            exc,
        )
    return _attach_cause(
        RagnorError(
            f"{default_stage} failed ({error_type})",
            retryable=False,
            trace_id=trace_id,
        ),
        exc,
    )


def is_retryable_exception(exc: Exception) -> bool:
    """Return True if ``exc`` represents a transient, safely retryable failure."""
    return classify_exception(exc).retryable
