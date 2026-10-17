"""Telemetry, event emission, privacy redaction, and health reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragit.telemetry.events import (
    CompositeEventSink,
    EventKind,
    EventSink,
    LoggingEventSink,
    MemoryEventSink,
    NullEventSink,
    RedactionConfig,
    TelemetryEvent,
    sanitize_error_category,
)
from ragit.telemetry.health import (
    ComponentHealth,
    HealthChecker,
    HealthReport,
    HealthState,
)

if TYPE_CHECKING:
    from ragit.telemetry.otel import OpenTelemetryEventSink

__all__ = [
    "ComponentHealth",
    "CompositeEventSink",
    "EventKind",
    "EventSink",
    "HealthChecker",
    "HealthReport",
    "HealthState",
    "LoggingEventSink",
    "MemoryEventSink",
    "NullEventSink",
    "OpenTelemetryEventSink",
    "RedactionConfig",
    "TelemetryEvent",
    "sanitize_error_category",
]


def __getattr__(name: str) -> Any:
    if name == "OpenTelemetryEventSink":
        from ragit.telemetry.otel import OpenTelemetryEventSink  # noqa: PLC0415

        return OpenTelemetryEventSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
