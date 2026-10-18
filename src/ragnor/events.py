"""Convenience re-exports for ragnor events and sinks."""

from __future__ import annotations

import ragnor.telemetry.events as _events

CompositeEventSink = _events.CompositeEventSink
EventKind = _events.EventKind
EventSink = _events.EventSink
LoggingEventSink = _events.LoggingEventSink
MemoryEventSink = _events.MemoryEventSink
NullEventSink = _events.NullEventSink
RedactionConfig = _events.RedactionConfig
TelemetryEvent = _events.TelemetryEvent
sanitize_error_category = _events.sanitize_error_category


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
