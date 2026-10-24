"""Optional OpenTelemetry bridge for event tracing and metric emission.

This module provides an optional bridge from :class:`~ragnor.telemetry.events.TelemetryEvent`
to OpenTelemetry spans, span events, and metrics. It is an optional extra
(``pip install 'ragnor[opentelemetry]'``) and never a core dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from ragnor.telemetry.events import TelemetryEvent

logger = logging.getLogger(__name__)


class OpenTelemetryEventSink:
    """EventSink forwarding ragnor events to active OpenTelemetry spans and meters.

    Requires the ``opentelemetry`` extra (``pip install 'ragnor[opentelemetry]'``).
    """

    def __init__(
        self,
        tracer: Any = None,
        meter: Any = None,
        *,
        service_name: str = "ragnor",
    ) -> None:
        """Initialize the OpenTelemetry bridge sink.

        Args:
            tracer: Optional opentelemetry.trace.Tracer. If None, acquires the
                global tracer.
            meter: Optional opentelemetry.metrics.Meter. If None, acquires the
                global meter.
            service_name: Instrumentation scope name.

        Raises:
            ImportError: If ``opentelemetry-api`` is not installed.
        """
        try:
            from opentelemetry import metrics, trace  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry integration requires the 'opentelemetry' extra: "
                "pip install 'ragnor[opentelemetry]'"
            ) from exc

        self._trace = trace
        self._metrics = metrics
        self._tracer = tracer or trace.get_tracer(service_name)
        self._meter = meter or metrics.get_meter(service_name)

        # Initialize standard instruments
        self._event_counter = self._meter.create_counter(
            name="ragnor.telemetry.events.total",
            description="Total count of ragnor telemetry events emitted",
            unit="1",
        )
        self._duration_histogram = self._meter.create_histogram(
            name="ragnor.duration_ms",
            description="Duration of timed operations in milliseconds",
            unit="ms",
        )

    def emit(self, event: TelemetryEvent) -> None:
        """Forward a TelemetryEvent to OpenTelemetry span events and metrics."""
        base_attributes: dict[str, Any] = {
            "ragnor.event_kind": event.event_kind,
        }
        if event.stage:
            base_attributes["ragnor.stage"] = event.stage
        if event.run_id:
            base_attributes["ragnor.run_id"] = event.run_id
        if event.request_id:
            base_attributes["ragnor.request_id"] = event.request_id
        if event.source_id:
            base_attributes["ragnor.source_id"] = event.source_id
        if event.document_id:
            base_attributes["ragnor.document_id"] = event.document_id
        if event.isolation_scope:
            base_attributes["ragnor.isolation_scope"] = event.isolation_scope
        if event.error_category:
            base_attributes["ragnor.error_category"] = event.error_category

        # Add event to active OpenTelemetry span if available
        span = self._trace.get_current_span()
        if span is not None and span.is_recording():
            span_attrs = dict(base_attributes)
            for k, v in event.attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    span_attrs[f"ragnor.attr.{k}"] = v
            span.add_event(name=event.event_kind, attributes=span_attrs)

        # Record metrics
        metric_attrs = {
            "event_kind": event.event_kind,
            "stage": event.stage or "unknown",
            "error_category": event.error_category or "none",
        }
        self._event_counter.add(1, attributes=metric_attrs)

        if event.duration_ms is not None:
            self._duration_histogram.record(event.duration_ms, attributes=metric_attrs)
