"""Unit tests for telemetry events, redaction, error sanitization, and sinks."""

from __future__ import annotations

import pytest

from ragnor.telemetry.events import (
    CompositeEventSink,
    EventKind,
    LoggingEventSink,
    MemoryEventSink,
    NullEventSink,
    RedactionConfig,
    TelemetryEvent,
    sanitize_error_category,
)


def test_telemetry_event_default_exclusion() -> None:
    """By default, raw text, query, credentials, and user id are excluded."""
    event = TelemetryEvent.create(
        event_kind=EventKind.SEARCH_COMPLETED,
        stage="retrieval",
        source_id="tenant-1",
        run_id="run-42",
        raw_text="Sensitive document text",
        raw_query="SELECT password FROM users WHERE secret='123'",
        raw_credentials="Bearer secret-token-xyz",
        user_id="alice@example.com",
    )
    assert event.event_kind == EventKind.SEARCH_COMPLETED
    assert event.stage == "retrieval"
    assert event.source_id == "tenant-1"
    assert event.run_id == "run-42"
    # By default, redacted/excluded fields are None
    assert event.raw_text is None
    assert event.raw_query is None
    assert event.raw_credentials is None
    assert event.user_id is None


def test_telemetry_event_opt_in_with_redaction() -> None:
    """Host-controlled opt-in allows fields but applies explicit redaction."""
    redaction = RedactionConfig(
        allow_raw_query=True,
        allow_user_id=True,
        scrub_credentials=True,
    )
    event = TelemetryEvent.create(
        event_kind=EventKind.SEARCH_COMPLETED,
        stage="retrieval",
        raw_query="find user with token Bearer secret_token_12345",
        raw_credentials="password123",
        user_id="user-123",
        redaction=redaction,
    )
    assert event.raw_query is not None
    assert "secret_token_12345" not in event.raw_query
    assert "[REDACTED]" in event.raw_query
    # raw_credentials is never allowed
    assert event.raw_credentials is None
    assert event.user_id == "user-123"


def test_sanitize_error_category_no_credentials_leaked() -> None:
    """Sanitize error category strips credentials and classifies known errors."""
    auth_exc = PermissionError("Invalid API key sk-proj-1234567890abcdef")
    assert sanitize_error_category(auth_exc) == "authentication_error"

    timeout_exc = TimeoutError("Connection to postgresql://user:pass@db:5432 timed out")
    assert sanitize_error_category(timeout_exc) == "timeout_error"

    conn_exc = ConnectionResetError("Connection refused by host")
    assert sanitize_error_category(conn_exc) == "connection_error"

    dim_exc = ValueError("Vector dimension 1536 does not match profile dimension 768")
    assert sanitize_error_category(dim_exc) == "dimension_error"

    custom_exc = RuntimeError("Some internal processing failure")
    assert sanitize_error_category(custom_exc) == "RuntimeError"


def test_memory_event_sink() -> None:
    """MemoryEventSink records events and supports filtering and clearing."""
    sink = MemoryEventSink()
    assert len(sink.events) == 0

    ev1 = TelemetryEvent(event_kind=EventKind.SCAN_STARTED, stage="sync", source_id="src1")
    ev2 = TelemetryEvent(event_kind=EventKind.DOCUMENT_LOADED, stage="sync", document_id="doc1")
    ev3 = TelemetryEvent(event_kind=EventKind.SEARCH_COMPLETED, stage="retrieval")

    sink.emit(ev1)
    sink.emit(ev2)
    sink.emit(ev3)

    assert len(sink.events) == 3
    assert len(sink.get_by_kind(EventKind.DOCUMENT_LOADED)) == 1
    assert sink.get_by_kind(EventKind.DOCUMENT_LOADED)[0].document_id == "doc1"
    assert len(sink.get_by_stage("sync")) == 2

    sink.clear()
    assert len(sink.events) == 0


def test_null_event_sink() -> None:
    """NullEventSink does not raise and discards events."""
    sink = NullEventSink()
    ev = TelemetryEvent(event_kind=EventKind.SCAN_STARTED)
    sink.emit(ev)


def test_logging_event_sink(caplog: pytest.LogCaptureFixture) -> None:
    """LoggingEventSink writes structured log records."""
    import logging

    sink = LoggingEventSink(logger_name="test.telemetry", level=logging.INFO)
    with caplog.at_level(logging.INFO, logger="test.telemetry"):
        sink.emit(
            TelemetryEvent(
                event_kind=EventKind.SCAN_STARTED,
                stage="sync",
                source_id="scope-1",
            )
        )
    assert any("scan_started" in record.message for record in caplog.records)


def test_composite_event_sink() -> None:
    """CompositeEventSink fans out events to all nested sinks."""
    s1 = MemoryEventSink()
    s2 = MemoryEventSink()
    composite = CompositeEventSink([s1, s2])

    ev = TelemetryEvent(event_kind=EventKind.BATCH_EMBEDDED, counts={"batch_size": 10})
    composite.emit(ev)

    assert len(s1.events) == 1
    assert len(s2.events) == 1
    assert s1.events[0].counts["batch_size"] == 10
    assert s2.events[0].counts["batch_size"] == 10


@pytest.mark.asyncio
async def test_sync_coordinator_emits_telemetry() -> None:
    from ragnor.chunking.base import Chunker
    from ragnor.embeddings.dimensions import register_dimension
    from ragnor.ingestion.sources import RawDocument
    from ragnor.stores.memory import InMemoryVectorStore
    from ragnor.sync.coordinator import DefaultSyncCoordinator
    from ragnor.sync.manifest import InMemoryManifestStore
    from ragnor.sync.state import InMemorySyncStateStore
    from ragnor.types import Chunk

    register_dimension("test-event-model", 4)

    class _Chunker(Chunker):
        def chunk(self, text: str) -> list[Chunk]:
            return [Chunk(id="c1", text=text, metadata={})]

    class _Settings:
        embedding_model = "test-event-model"

    async def _embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0, 4.0] for _ in texts]

    sink = MemoryEventSink()
    coordinator = DefaultSyncCoordinator(
        manifest_store=InMemoryManifestStore(),
        state_store=InMemorySyncStateStore(),
        vector_store=InMemoryVectorStore(),
        embedder=_embed_fn,
        chunker=_Chunker(),
        settings=_Settings(),  # type: ignore[arg-type]
        event_sink=sink,
    )

    doc1 = RawDocument(source_name="doc-1.txt", content="Hello world", external_id="doc-1")
    await coordinator.run_sync("scope-test", [doc1])

    kinds = [e.event_kind for e in sink.events]
    assert EventKind.SCAN_STARTED in kinds
    assert EventKind.DOCUMENT_LOADED in kinds
    assert EventKind.GENERATION_ACTIVATED in kinds

    # Run again with identical scan -> should emit DOCUMENT_SKIPPED
    sink.clear()
    await coordinator.run_sync("scope-test", [doc1])
    kinds2 = [e.event_kind for e in sink.events]
    assert EventKind.SCAN_STARTED in kinds2
    assert EventKind.DOCUMENT_SKIPPED in kinds2


@pytest.mark.asyncio
async def test_sync_coordinator_sync_failed_event() -> None:
    from ragnor.chunking.base import Chunker
    from ragnor.embeddings.dimensions import register_dimension
    from ragnor.ingestion.sources import RawDocument
    from ragnor.stores.memory import InMemoryVectorStore
    from ragnor.sync.coordinator import ActivationMode, DefaultSyncCoordinator
    from ragnor.sync.manifest import InMemoryManifestStore
    from ragnor.sync.state import InMemorySyncStateStore
    from ragnor.types import Chunk

    register_dimension("test-fail-model", 4)

    class _Chunker(Chunker):
        def chunk(self, text: str) -> list[Chunk]:
            return [Chunk(id="c1", text=text, metadata={})]

    class _Settings:
        embedding_model = "test-fail-model"

    async def _embed_fail(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding backend network failure")

    sink = MemoryEventSink()
    coordinator = DefaultSyncCoordinator(
        manifest_store=InMemoryManifestStore(),
        state_store=InMemorySyncStateStore(),
        vector_store=InMemoryVectorStore(),
        embedder=_embed_fail,
        chunker=_Chunker(),
        settings=_Settings(),  # type: ignore[arg-type]
        activation_mode=ActivationMode.COLLECTION,  # collection mode raises on doc failure
        event_sink=sink,
    )

    doc = RawDocument(source_name="doc-fail.txt", content="Fail content", external_id="doc-fail")
    with pytest.raises(RuntimeError):
        await coordinator.run_sync("scope-fail", [doc])

    failed_events = sink.get_by_kind(EventKind.SYNC_FAILED)
    assert len(failed_events) == 1
    assert failed_events[0].sanitized_error_category == "RuntimeError"


@pytest.mark.asyncio
async def test_embedding_executor_telemetry() -> None:
    from ragnor.embeddings.executor import EmbeddingExecutor
    from ragnor.embeddings.profile import EmbeddingProfile

    profile = EmbeddingProfile(
        provider_id="test",
        model_id="test-model",
        dimension=4,
    )

    calls = 0

    async def _failing_then_success(batch: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("Connection reset by peer")
        return [[0.1, 0.2, 0.3, 0.4] for _ in batch]

    sink = MemoryEventSink()
    executor = EmbeddingExecutor(
        fn=_failing_then_success,
        profile=profile,
        max_retries=2,
        base_backoff=0.001,
        event_sink=sink,
    )

    vectors, metrics = await executor.embed(["item1", "item2"])
    assert len(vectors) == 2
    assert metrics.total_retry_count == 1

    retry_events = sink.get_by_kind(EventKind.RETRY_SCHEDULED)
    assert len(retry_events) == 1
    assert retry_events[0].sanitized_error_category == "connection_error"

    batch_events = sink.get_by_kind(EventKind.BATCH_EMBEDDED)
    assert len(batch_events) == 1
    assert batch_events[0].counts.get("batch_size") == 2


@pytest.mark.asyncio
async def test_retriever_telemetry() -> None:
    from unittest.mock import AsyncMock

    from ragnor.retrieval.retriever import HybridRetriever, RetrieverSpec

    class _FakeSettings:
        rag_distance_threshold = 0.5
        embedding_model = "test-model"

    spec = RetrieverSpec(table_name="test_table")
    settings = _FakeSettings()

    from ragnor.retrieval.strategies import RetrievalStrategy

    mock_strategy = AsyncMock(spec=RetrievalStrategy)
    mock_strategy.retrieve.return_value = [
        {"id": "doc1", "document": "Hello", "metadata": {}, "_rrf_score": 0.8}
    ]

    sink = MemoryEventSink()
    retriever = HybridRetriever(
        spec=spec,
        settings=settings,  # type: ignore[arg-type]
        strategy=mock_strategy,
        event_sink=sink,
    )

    results = await retriever.retrieve("query text", n_results=3)
    assert len(results) == 1

    search_events = sink.get_by_kind(EventKind.SEARCH_COMPLETED)
    assert len(search_events) == 1
    ev = search_events[0]
    assert ev.stage == "retrieval"
    assert ev.counts.get("final_k") == 3
    assert ev.counts.get("result_count") == 1
    assert ev.duration_ms is not None
    # Verify query text is excluded by default
    assert ev.raw_query is None
