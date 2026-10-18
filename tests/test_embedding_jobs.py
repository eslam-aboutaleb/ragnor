"""
Tests for the ragnor job outbox engine.

The retry/backoff/dead-letter engine is driven here
through a fake in-memory
:class:`~ragnor.jobs.outbox.EmbeddingJobStore`
and :class:`~ragnor.jobs.outbox.JobProcessor` —
no host dependency. The host adapter's database
behavior is covered by the host application's own
test suite.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragnor.jobs.outbox import (
    MAX_ATTEMPTS,
    RETRY_SCHEDULE,
    process_pending_jobs,
)
from tests.job_doubles import FakeJobProcessor, FakeJobStore


def _pending_job(store: FakeJobStore, **overrides) -> object:
    return store.add(**overrides)


@pytest.mark.asyncio
async def test_failed_job_retries_with_backoff():
    """Failing a job 3 times keeps it failed with a future next_retry_at."""
    store = FakeJobStore()
    job = _pending_job(store)
    processor = FakeJobProcessor(error=Exception("embed error"))

    for attempt in range(3):
        await process_pending_jobs(store, processor, limit=10, worker_id="w")
        # The backoff elapses between passes, making the job retry-due.
        if attempt < 2 and job.next_retry_at is not None:
            job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)

    assert job.status == "failed"
    assert job.retry_count == 3
    assert job.next_retry_at is not None
    assert job.next_retry_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_failed_job_moves_to_dead_letter_after_max_attempts():
    """Failing a job 4 times moves it to dead_letter."""
    store = FakeJobStore()
    job = _pending_job(store)
    processor = FakeJobProcessor(error=Exception("embed error"))

    for _ in range(4):
        await process_pending_jobs(store, processor, limit=10, worker_id="w")
        # The backoff elapses between passes, making the job retry-due.
        if job.next_retry_at is not None:
            job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)

    assert job.status == "dead_letter"
    assert job.retry_count == 4
    assert job.next_retry_at is None
    assert "Exhausted 4 attempts" in job.status_detail


@pytest.mark.asyncio
async def test_retriable_failed_job_is_processed_after_backoff():
    """A failed job whose next_retry_at has passed is retried and can succeed."""
    past_time = datetime.now(UTC) - timedelta(minutes=2)
    store = FakeJobStore()
    job = _pending_job(
        store,
        status="failed",
        status_detail="previous failure",
        retry_count=1,
        next_retry_at=past_time,
        max_attempts=4,
    )
    processor = FakeJobProcessor()

    processed = await process_pending_jobs(store, processor, limit=10, worker_id="w")

    assert processed == 1
    assert job.status == "completed"
    assert job.completed_at is not None
    assert processor.processed == [job]


@pytest.mark.asyncio
async def test_dead_letter_job_is_never_selected():
    """A dead_letter job is never picked up by the worker."""
    store = FakeJobStore()
    _pending_job(store, status="dead_letter", status_detail="gave up")
    processor = FakeJobProcessor()

    processed = await process_pending_jobs(store, processor, limit=10, worker_id="w")

    assert processed == 0
    assert processor.processed == []


@pytest.mark.asyncio
async def test_retry_schedule_constants():
    """Retry schedule has the expected number of intervals and MAX_ATTEMPTS matches."""
    assert len(RETRY_SCHEDULE) == MAX_ATTEMPTS - 1
    assert RETRY_SCHEDULE[0] == timedelta(minutes=1)
    assert RETRY_SCHEDULE[1] == timedelta(minutes=5)
    assert RETRY_SCHEDULE[2] == timedelta(minutes=30)


@pytest.mark.asyncio
async def test_processor_receives_the_full_job_payload():
    """The processor sees the job's payload (document text and metadata)."""
    store = FakeJobStore()
    job = _pending_job(
        store,
        payload={
            "claim_id": "CLM-8821",
            "claim_type": "Water Damage",
            "description": "Pipe burst",
            "claim_status": "Denied",
        },
    )
    processor = FakeJobProcessor()

    await process_pending_jobs(store, processor, limit=10, worker_id="w")

    assert processor.processed == [job]
    assert job.payload["claim_status"] == "Denied"


@pytest.mark.asyncio
async def test_claimed_jobs_are_stamped_with_lock_fields():
    """Claimed jobs record which worker took them and when."""
    store = FakeJobStore()
    job = _pending_job(store)
    processor = FakeJobProcessor()

    await process_pending_jobs(store, processor, limit=10, worker_id="worker-1")

    assert job.status == "completed"
    assert job.locked_by == "worker-1"
    assert job.locked_at is not None


@pytest.mark.asyncio
async def test_max_attempts_comes_from_the_job_record():
    """The job record's max_attempts is authoritative: a job with
    max_attempts=2 dead-letters after 2 failures."""
    store = FakeJobStore()
    job = _pending_job(store, max_attempts=2)
    processor = FakeJobProcessor(error=Exception("embed error"))

    for _ in range(2):
        await process_pending_jobs(store, processor, limit=10, worker_id="w")
        # The backoff elapses between passes, making the job retry-due.
        if job.next_retry_at is not None:
            job.next_retry_at = datetime.now(UTC) - timedelta(seconds=1)

    assert job.status == "dead_letter"
    assert job.retry_count == 2
    assert "Exhausted 2 attempts" in job.status_detail


@pytest.mark.asyncio
async def test_failed_job_releases_its_lock():
    """A failed job clears its lock so a later pass can reclaim it."""
    store = FakeJobStore()
    job = _pending_job(store)
    processor = FakeJobProcessor(error=Exception("embed error"))

    await process_pending_jobs(store, processor, limit=10, worker_id="worker-1")

    assert job.status == "failed"
    assert job.locked_at is None
    assert job.locked_by is None


@pytest.mark.asyncio
async def test_failed_job_with_exhausted_attempts_is_not_requeued():
    """A failed job past its attempt ceiling is not claimable again."""
    store = FakeJobStore()
    _pending_job(
        store,
        status="failed",
        status_detail="old failure",
        retry_count=4,
        max_attempts=4,
        next_retry_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    processor = FakeJobProcessor()

    processed = await process_pending_jobs(store, processor, limit=10, worker_id="w")

    assert processed == 0
    assert processor.processed == []


@pytest.mark.asyncio
async def test_claim_is_limited_and_ordered_by_creation():
    """Only up to ``limit`` jobs are claimed, oldest first."""
    store = FakeJobStore()
    oldest = _pending_job(store, job_id="job-1")
    middle = _pending_job(store, job_id="job-2")
    newest = _pending_job(store, job_id="job-3")
    processor = FakeJobProcessor()

    processed = await process_pending_jobs(store, processor, limit=2, worker_id="w")

    assert processed == 2
    assert oldest in processor.processed
    assert middle in processor.processed
    assert newest not in processor.processed
    assert store.claim_limits == [2]
    assert store.claim_worker_ids == ["w"]
