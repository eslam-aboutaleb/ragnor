"""Compatibility import for :mod:`ragnor.telemetry.events`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.telemetry import events as _implementation

if TYPE_CHECKING:
    from ragnor.telemetry.events import (
        CompositeEventSink as CompositeEventSink,
        EventKind as EventKind,
        EventSink as EventSink,
        LoggingEventSink as LoggingEventSink,
        MemoryEventSink as MemoryEventSink,
        NullEventSink as NullEventSink,
        RedactionConfig as RedactionConfig,
        TelemetryEvent as TelemetryEvent,
        sanitize_error_category as sanitize_error_category,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
