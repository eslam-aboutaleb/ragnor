"""Optional lease heartbeat tests for slow durable outbox processors."""

from __future__ import annotations

import asyncio

import pytest

from ragnor.jobs.outbox import (
    InMemoryEmbeddingJobStore,
    JobPayload,
    LostJobOwnershipError,
    process_pending_jobs,
)


class HostLeaseStore(InMemoryEmbeddingJobStore):
    def __init__(self) -> None:
        super().__init__([JobPayload(job_id="one")])
        self.renewed = asyncio.Event()
        self.lose_ownership = False

    async def renew_lease(self, job: JobPayload, *, lease_seconds: float | None = None) -> None:
        if self.lose_ownership:
            raise LostJobOwnershipError("claim expired")
        self.renewed.set()


class WaitingProcessor:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.cancelled = False

    async def process(self, job: JobPayload) -> None:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_opt_in_heartbeat_renews_until_processor_completes() -> None:
    store = HostLeaseStore()
    processor = WaitingProcessor(store.renewed)
    assert (
        await process_pending_jobs(
            store, processor, worker_id="host", heartbeat_interval_seconds=0.01
        )
        == 1
    )
    assert store.renewed.is_set()
    assert store.jobs[0].status == "completed"


@pytest.mark.asyncio
async def test_expired_heartbeat_cancels_processing_without_ack() -> None:
    store = HostLeaseStore()
    store.lose_ownership = True
    processor = WaitingProcessor(asyncio.Event())
    with pytest.raises(LostJobOwnershipError):
        await process_pending_jobs(
            store, processor, worker_id="host", heartbeat_interval_seconds=0.01
        )
    assert processor.cancelled
    assert store.jobs[0].status != "completed"


@pytest.mark.asyncio
async def test_requesting_heartbeat_without_renewal_support_fails_closed() -> None:
    store = InMemoryEmbeddingJobStore([JobPayload(job_id="one")])
    processor = WaitingProcessor(asyncio.Event())
    with pytest.raises(ValueError, match="renew_lease"):
        await process_pending_jobs(
            store, processor, worker_id="host", heartbeat_interval_seconds=0.01
        )


@pytest.mark.asyncio
async def test_invalid_heartbeat_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        await process_pending_jobs(
            HostLeaseStore(),
            WaitingProcessor(asyncio.Event()),
            heartbeat_interval_seconds=0,
        )


@pytest.mark.asyncio
async def test_multi_host_claim_requirement_rejects_non_atomic_local_store() -> None:
    store = InMemoryEmbeddingJobStore([JobPayload(job_id="local")])
    processor = WaitingProcessor(asyncio.Event())
    with pytest.raises(ValueError, match="atomic"):
        await process_pending_jobs(
            store,
            processor,
            worker_id="multiple-hosts",
            require_atomic_claims=True,
        )


@pytest.mark.asyncio
async def test_heartbeat_propagates_processor_failure_instead_of_acknowledging() -> None:
    class FailingProcessor:
        async def process(self, job: JobPayload) -> None:
            raise ValueError("permanent processor failure")

    store = HostLeaseStore()
    assert (
        await process_pending_jobs(
            store, FailingProcessor(), worker_id="host", heartbeat_interval_seconds=0.01
        )
        == 0
    )
    assert store.jobs[0].status != "completed"
    assert store.jobs[0].retry_count == 1
