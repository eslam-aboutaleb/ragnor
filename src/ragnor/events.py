"""Compatibility import for :mod:`ragnor.telemetry.events`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.telemetry import events as _implementation

if TYPE_CHECKING:
    from ragnor.telemetry.events import (
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

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
