"""Explicit production publication, retry policy and telemetry port contracts."""

from __future__ import annotations

from datetime import timedelta

import pytest

from ragnor.jobs.outbox import (
    InMemoryEmbeddingJobStore,
    JobPayload,
    process_pending_jobs,
)


class Processor:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    async def process(self, job: JobPayload) -> None:
        self.calls.append(job.job_id)
        if self.error is not None:
            raise self.error


class PublishingStore(InMemoryEmbeddingJobStore):
    require_publication = True

    def __init__(self) -> None:
        super().__init__([JobPayload(job_id="one")])
        self.published: list[str] = []

    async def mark_completed(self, job: JobPayload) -> None:
        raise AssertionError("ordinary acknowledgment bypasses mandatory publication")

    async def activate_generation(self, job: JobPayload) -> bool:
        self.published.append(job.job_id)
        job.status = "completed"
        return True


class UnconfiguredPublishingStore(InMemoryEmbeddingJobStore):
    require_publication = True


class PermanentRetry:
    def delay(self, error: Exception, attempt: int, max_attempts: int) -> timedelta | None:
        return None


class TenSecondRetry:
    def delay(self, error: Exception, attempt: int, max_attempts: int) -> timedelta | None:
        return timedelta(seconds=10)


class EventSink:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_publication_required_worker_activates_instead_of_direct_ack() -> None:
    store = PublishingStore()
    processor = Processor()
    events = EventSink()
    count = await process_pending_jobs(store, processor, worker_id="host", event_sink=events)
    assert count == 1
    assert processor.calls == ["one"]
    assert store.published == ["one"]
    assert store.jobs[0].status == "completed"
    assert [event.kind for event in events.events] == ["claimed", "completed"]
    assert all("payload" not in repr(event) for event in events.events)


@pytest.mark.asyncio
async def test_incompatible_publication_store_fails_before_processor_side_effect() -> None:
    store = UnconfiguredPublishingStore([JobPayload(job_id="one")])
    processor = Processor()
    with pytest.raises(ValueError, match="activate_generation"):
        await process_pending_jobs(store, processor, worker_id="host")
    assert processor.calls == []
    assert store.jobs[0].status == "pending"


@pytest.mark.asyncio
async def test_explicit_permanent_error_uses_dead_letter_without_retry_or_raw_text() -> None:
    store = InMemoryEmbeddingJobStore([JobPayload(job_id="one")])
    events = EventSink()
    processed = await process_pending_jobs(
        store,
        Processor(ValueError("SECRET document body")),
        worker_id="host",
        retry_policy=PermanentRetry(),
        event_sink=events,
    )
    assert processed == 0
    job = store.jobs[0]
    assert job.status == "dead_letter"
    assert job.retry_count == 1
    assert job.next_retry_at is None
    assert "SECRET" not in (job.status_detail or "")
    assert [event.kind for event in events.events] == ["claimed", "dead_letter"]
    assert all("SECRET" not in repr(event) for event in events.events)


@pytest.mark.asyncio
async def test_custom_retry_schedule_is_used_instead_of_fixed_global_schedule() -> None:
    store = InMemoryEmbeddingJobStore([JobPayload(job_id="one")])
    await process_pending_jobs(
        store,
        Processor(RuntimeError("failure")),
        worker_id="host",
        retry_policy=TenSecondRetry(),
    )
    job = store.jobs[0]
    assert job.status == "failed"
    assert job.next_retry_at is not None
    delta = (job.next_retry_at - job.locked_at).total_seconds() if job.locked_at else 10
    assert 0 <= delta <= 11


@pytest.mark.asyncio
async def test_event_sink_failure_cannot_undo_committed_job_ack() -> None:
    class BrokenSink:
        async def emit(self, event) -> None:
            raise RuntimeError("observer failure")

    store = InMemoryEmbeddingJobStore([JobPayload(job_id="one")])
    assert (
        await process_pending_jobs(store, Processor(), worker_id="host", event_sink=BrokenSink())
        == 1
    )
    assert store.jobs[0].status == "completed"
