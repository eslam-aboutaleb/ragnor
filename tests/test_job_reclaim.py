"""Tier B: stale-job reclaim for the ragnor job outbox.

The reclaim engine lives in ragnor; these tests drive it
through the fake in-memory store. The host adapter's
database behavior (real ``embedding_jobs`` rows,
atomic claim behavior remains covered by the host
application's own integration suite.

A drainer that dies mid-upsert leaves its rows locked in
``processing`` forever. ``reclaim_stale_jobs`` must reset
exactly the abandoned rows — old lock timestamp, or a NULL
lock from before the column existed — and leave live locks
and other statuses alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragnor.jobs.outbox import reclaim_stale_jobs
from tests.job_doubles import FakeJobStore


@pytest.mark.asyncio
async def test_reclaim_resets_only_abandoned_locks():
    """Stale and NULL-locked processing rows return to pending; live locks stay."""
    store = FakeJobStore()
    stale = store.add(
        status="processing",
        status_detail="in flight",
        locked_at=datetime.now(UTC) - timedelta(hours=1),
        locked_by="dead-worker",
    )
    null_lock = store.add(
        status="processing",
        status_detail="in flight",
        locked_at=None,
        locked_by="unknown",
    )
    live = store.add(
        status="processing",
        status_detail="in flight",
        locked_at=datetime.now(UTC),
        locked_by="live-worker",
    )
    pending = store.add(status="pending", status_detail="pending")

    reclaimed = await reclaim_stale_jobs(store, stale_after_seconds=300)

    assert reclaimed == 2
    assert stale.status == "pending"
    assert stale.locked_at is None
    assert stale.locked_by is None
    assert stale.status_detail is None
    assert null_lock.status == "pending"
    assert null_lock.locked_by is None
    assert live.status == "processing"
    assert live.locked_by == "live-worker"
    assert live.locked_at is not None
    assert pending.status == "pending"


@pytest.mark.asyncio
async def test_reclaim_with_no_stale_jobs_is_a_no_op():
    """A queue with only live locks reclaims nothing."""
    store = FakeJobStore()
    live = store.add(
        status="processing",
        status_detail="in flight",
        locked_at=datetime.now(UTC),
        locked_by="live-worker",
    )

    reclaimed = await reclaim_stale_jobs(store, stale_after_seconds=300)

    assert reclaimed == 0
    assert live.status == "processing"
    assert live.locked_by == "live-worker"
    assert live.locked_at is not None


@pytest.mark.asyncio
async def test_reclaim_uses_the_store_default_threshold():
    """A NULL stale_after_seconds defers to the store's own default."""
    store = FakeJobStore()
    stale = store.add(
        status="processing",
        status_detail="in flight",
        locked_at=datetime.now(UTC) - timedelta(hours=1),
        locked_by="dead-worker",
    )

    reclaimed = await reclaim_stale_jobs(store)

    assert reclaimed == 1
    assert stale.status == "pending"


@pytest.mark.asyncio
async def test_reclaim_ignores_failed_and_completed_rows():
    """Only ``processing`` rows are reclaim candidates."""
    store = FakeJobStore()
    failed = store.add(
        status="failed",
        status_detail="boom",
        retry_count=1,
        locked_at=datetime.now(UTC) - timedelta(hours=1),
        locked_by="dead-worker",
    )
    completed = store.add(
        status="completed",
        status_detail=None,
        completed_at=datetime.now(UTC),
        locked_at=datetime.now(UTC) - timedelta(hours=1),
        locked_by="dead-worker",
    )

    reclaimed = await reclaim_stale_jobs(store, stale_after_seconds=300)

    assert reclaimed == 0
    assert failed.status == "failed"
    assert completed.status == "completed"
