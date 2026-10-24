"""Typed events, EventSink protocols, and privacy-preserving redaction.

Core emits typed events for lifecycle monitoring, sync, embedding, and retrieval.
Raw text, queries, credentials, and user identifiers are excluded by default;
host-controlled opt-in telemetry requires explicit redaction rules.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class EventKind(StrEnum):
    """Standardized event kind identifiers emitted across core ragnor stages."""

    SCAN_STARTED = "scan_started"
    DOCUMENT_LOADED = "document_loaded"
    DOCUMENT_SKIPPED = "document_skipped"
    BATCH_EMBEDDED = "batch_embedded"
    GENERATION_ACTIVATED = "generation_activated"
    RETRY_SCHEDULED = "retry_scheduled"
    SYNC_FAILED = "sync_failed"
    SEARCH_COMPLETED = "search_completed"


def sanitize_error_category(exc: Any) -> str:  # noqa: PLR0911
    """Classify an exception into a sanitized, secret-free category string.

    Prevents sensitive strings (queries, passwords, file paths, connection URLs)
    from leaking into logs or event streams.
    """
    if exc is None:
        return "none"

    type_name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if any(k in type_name for k in ("auth", "credential", "permission")) or any(
        k in msg for k in ("unauthorized", "401", "forbidden", "403", "api key", "apikey", "sk-")
    ):
        return "authentication_error"
    if "rate" in msg or "429" in msg or "quota" in msg:
        return "rate_limit_error"
    if "timeout" in type_name or "timeout" in msg:
        return "timeout_error"
    if (
        any(k in type_name for k in ("connect", "network", "socket"))
        or "connection" in msg
        or "refused" in msg
    ):
        return "connection_error"
    if "dimension" in type_name or "dimension" in msg:
        return "dimension_error"
    if "notfound" in type_name or "not found" in msg or "404" in msg:
        return "not_found_error"
    if any(k in type_name for k in ("validation", "valueerror", "typeerror")) or "invalid" in msg:
        return "validation_error"
    if any(k in msg for k in ("fence", "lease", "stale", "superseded")):
        return "lease_or_fence_error"
    if "transient" in type_name:
        return "transient_error"

    return type(exc).__name__


@dataclass
class RedactionConfig:
    """Privacy configuration for telemetry attribute redaction.

    By default, raw text, query strings, credentials, and user identifiers
    are excluded from all telemetry events. Hosts can explicitly opt-in to
    capturing specific fields with designated redaction rules.
    """

    allow_queries: bool = False
    allow_document_text: bool = False
    allow_user_identifiers: bool = False
    allow_unmasked_errors: bool = False
    mask_credentials: bool = True

    def __init__(  # noqa: PLR0913
        self,
        allow_queries: bool = False,
        allow_document_text: bool = False,
        allow_user_identifiers: bool = False,
        allow_unmasked_errors: bool = False,
        mask_credentials: bool = True,
        *,
        allow_raw_query: bool | None = None,
        allow_raw_text: bool | None = None,
        allow_user_id: bool | None = None,
        scrub_credentials: bool | None = None,
    ) -> None:
        self.allow_queries = allow_raw_query if allow_raw_query is not None else allow_queries
        self.allow_document_text = (
            allow_raw_text if allow_raw_text is not None else allow_document_text
        )
        self.allow_user_identifiers = (
            allow_user_id if allow_user_id is not None else allow_user_identifiers
        )
        self.allow_unmasked_errors = allow_unmasked_errors
        self.mask_credentials = (
            scrub_credentials if scrub_credentials is not None else mask_credentials
        )

    _SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
        "password",
        "secret",
        "token",
        "key",
        "auth",
        "credential",
        "dsn",
        "database_url",
    )

    def scrub_text(self, text: str) -> str:
        """Scrub known credential patterns from arbitrary text."""
        # Replace Bearer tokens, passwords, keys
        pattern = re.compile(
            r"(?i)(bearer\s+[A-Za-z0-9_\-\.]+)|"
            r"((?:password|token|secret|key)\s*[=:]\s*['\"]?[^\s'\"]+['\"]?)"
        )
        return pattern.sub("[REDACTED]", text)

    def sanitize_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Sanitize an attributes dictionary according to redaction rules."""
        sanitized: dict[str, Any] = {}
        for k, v in attributes.items():
            k_lower = k.lower()

            # Mask credentials
            if self.mask_credentials and any(p in k_lower for p in self._SENSITIVE_KEY_PATTERNS):
                sanitized[k] = "[REDACTED]"
                continue

            # Query text
            if "query" in k_lower and not self.allow_queries:
                sanitized[k] = "[QUERY_REDACTED]"
                continue

            # Document raw text
            if (
                ("text" in k_lower or "document" in k_lower or "content" in k_lower)
                and not self.allow_document_text
                and isinstance(v, str)
                and len(v) > 20
            ):
                sanitized[k] = f"[TEXT_REDACTED_LEN_{len(v)}]"
                continue

            # User identifiers
            if (
                any(u in k_lower for u in ("user_id", "email", "username", "account"))
                and not self.allow_user_identifiers
            ):
                sanitized[k] = "[USER_REDACTED]"
                continue

            sanitized[k] = v
        return sanitized


DEFAULT_REDACTION_CONFIG = RedactionConfig()


@dataclass
class TelemetryEvent:
    """A strongly typed event emitted by ragnor components.

    Attributes:
        event_kind: Classification from :class:`EventKind` or custom string.
        timestamp: Unix epoch timestamp in seconds.
        event_id: Unique event UUID string.
        run_id: Pipeline or sync execution run identifier.
        request_id: Search or query request identifier.
        stage: Component stage (e.g. "sync", "embedding", "retrieval").
        source_id: Scoped source identifier.
        document_id: Scoped document identifier.
        isolation_scope: Tenant or namespace scope.
        counts: Stage-specific numeric measurements (e.g. batch size, tokens).
        duration_ms: Duration in milliseconds for timed operations.
        error_category: Sanitized error classification.
        raw_text: Optional document text, excluded by default.
        raw_query: Optional query text, excluded by default.
        raw_credentials: Always excluded or None.
        user_id: Optional user identifier, excluded by default.
        attributes: Additional sanitized event key-values.
    """

    event_kind: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str | None = None
    request_id: str | None = None
    stage: str | None = None
    source_id: str | None = None
    document_id: str | None = None
    isolation_scope: str | None = None
    generation: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    duration_ms: float | None = None
    error_category: str | None = None
    error: Any = None
    raw_text: str | None = None
    raw_query: str | None = None
    raw_credentials: str | None = None
    user_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Sanitize error to error_category and scrub error object."""
        if self.error is not None and self.error_category is None:
            self.error_category = sanitize_error_category(self.error)
        self.error = None

    @property
    def sanitized_error_category(self) -> str | None:
        """Alias for error_category for clarity."""
        return self.error_category

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        event_kind: str | EventKind,
        *,
        run_id: str | None = None,
        request_id: str | None = None,
        stage: str | None = None,
        source_id: str | None = None,
        document_id: str | None = None,
        isolation_scope: str | None = None,
        generation: int | None = None,
        counts: dict[str, int] | None = None,
        duration_ms: float | None = None,
        error: Any = None,
        raw_text: str | None = None,
        raw_query: str | None = None,
        raw_credentials: str | None = None,
        user_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        redaction: RedactionConfig | None = None,
        **extra_attrs: Any,
    ) -> TelemetryEvent:
        """Create a TelemetryEvent with automatic sanitization and redaction."""
        cfg = redaction or DEFAULT_REDACTION_CONFIG
        err_cat = sanitize_error_category(error) if error is not None else None

        merged_attrs = dict(attributes or {})
        merged_attrs.update(extra_attrs)
        safe_attrs = cfg.sanitize_attributes(merged_attrs)

        # Apply default exclusions & redactions
        safe_query: str | None = None
        if cfg.allow_queries and raw_query is not None:
            safe_query = cfg.scrub_text(raw_query) if cfg.mask_credentials else raw_query

        safe_text: str | None = None
        if cfg.allow_document_text and raw_text is not None:
            safe_text = cfg.scrub_text(raw_text) if cfg.mask_credentials else raw_text

        safe_user: str | None = None
        if cfg.allow_user_identifiers and user_id is not None:
            safe_user = user_id

        return cls(
            event_kind=str(event_kind),
            run_id=run_id,
            request_id=request_id,
            stage=stage,
            source_id=source_id,
            document_id=document_id,
            isolation_scope=isolation_scope,
            counts=dict(counts or {}),
            duration_ms=duration_ms,
            error_category=err_cat,
            raw_text=safe_text,
            raw_query=safe_query,
            raw_credentials=None,  # credentials never allowed
            user_id=safe_user,
            attributes=safe_attrs,
        )


@runtime_checkable
class EventSink(Protocol):
    """Protocol for consuming telemetry events emitted by the core system."""

    def emit(self, event: TelemetryEvent) -> None:
        """Process or forward a telemetry event."""
        ...


class NullEventSink:
    """No-op EventSink that discards all emitted events with zero overhead."""

    def emit(self, event: TelemetryEvent) -> None:
        """Discard event."""


class MemoryEventSink:
    """In-memory EventSink capturing events for inspection, assertions, and testing."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        """Record event in memory."""
        self.events.append(event)

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()

    def get_by_kind(self, event_kind: str | EventKind) -> list[TelemetryEvent]:
        """Return all recorded events matching ``event_kind``."""
        target = str(event_kind)
        return [e for e in self.events if e.event_kind == target]

    def get_by_stage(self, stage: str) -> list[TelemetryEvent]:
        """Return all recorded events matching ``stage``."""
        return [e for e in self.events if e.stage == stage]


class LoggingEventSink:
    """EventSink emitting event summaries via Python's standard logging module."""

    def __init__(self, logger_name: str = "ragnor.events", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def emit(self, event: TelemetryEvent) -> None:
        """Log event summary."""
        parts = [f"[{event.event_kind}]"]
        if event.stage:
            parts.append(f"stage={event.stage}")
        if event.run_id:
            parts.append(f"run_id={event.run_id}")
        if event.request_id:
            parts.append(f"request_id={event.request_id}")
        if event.duration_ms is not None:
            parts.append(f"duration_ms={event.duration_ms:.2f}")
        if event.error_category:
            parts.append(f"error={event.error_category}")
        if event.counts:
            parts.append(f"counts={event.counts}")

        self._logger.log(self._level, " ".join(parts))


class CompositeEventSink:
    """EventSink forwarding emitted events to multiple underlying sinks."""

    def __init__(self, sinks: Sequence[EventSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, event: TelemetryEvent) -> None:
        """Forward event to all configured child sinks."""
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EventSink %r failed to emit event: %s", sink, exc)


__all__ = [
    "CompositeEventSink",
    "EventKind",
    "EventSink",
    "LoggingEventSink",
    "MemoryEventSink",
    "NullEventSink",
    "RedactionConfig",
    "TelemetryEvent",
    "sanitize_error_category",
]
