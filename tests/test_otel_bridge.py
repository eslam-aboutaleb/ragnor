"""Unit tests for the optional OpenTelemetry bridge."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ragnor.telemetry.events import EventKind, TelemetryEvent


def test_otel_sink_missing_dependency() -> None:
    """When opentelemetry is not installed, instantiation raises a clear ImportError."""
    with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
        from ragnor.telemetry.otel import OpenTelemetryEventSink

        with pytest.raises(ImportError) as exc_info:
            OpenTelemetryEventSink()
        assert "ragnor[opentelemetry]" in str(exc_info.value)


def test_otel_sink_with_mocked_otel() -> None:
    """When opentelemetry is available, events are recorded to tracer and meter."""
    mock_trace = MagicMock()
    mock_metrics = MagicMock()
    mock_tracer = MagicMock()
    mock_meter = MagicMock()
    mock_counter = MagicMock()
    mock_histogram = MagicMock()

    mock_trace.get_tracer.return_value = mock_tracer
    mock_metrics.get_meter.return_value = mock_meter
    mock_meter.create_counter.return_value = mock_counter
    mock_meter.create_histogram.return_value = mock_histogram

    mock_otel = MagicMock()
    mock_otel.trace = mock_trace
    mock_otel.metrics = mock_metrics

    with patch.dict(
        sys.modules,
        {
            "opentelemetry": mock_otel,
            "opentelemetry.trace": mock_trace,
            "opentelemetry.metrics": mock_metrics,
        },
    ):
        from ragnor.telemetry.otel import OpenTelemetryEventSink

        sink = OpenTelemetryEventSink(service_name="test-service")
        event = TelemetryEvent(
            event_kind=EventKind.SEARCH_COMPLETED,
            stage="retrieval",
            source_id="scope-abc",
            duration_ms=45.2,
            counts={"final_k": 5, "result_count": 3},
        )
        sink.emit(event)

        # Counter should be incremented
        mock_counter.add.assert_called_once()
        args, kwargs = mock_counter.add.call_args
        assert args[0] == 1
        assert kwargs["attributes"]["event_kind"] == "search_completed"

        # Histogram should record duration
        mock_histogram.record.assert_called_once()
        h_args, h_kwargs = mock_histogram.record.call_args
        assert h_args[0] == 45.2

        # Active span should receive the event
        mock_span = mock_trace.get_current_span.return_value
        mock_span.add_event.assert_called_once()
        s_args, s_kwargs = mock_span.add_event.call_args
        assert s_kwargs["name"] == "search_completed"
