"""Generic embedding-job outbox.

The outbox pattern
decouples row insertion from embedding generation: a
job row is written in the same transaction as the row
it refers to, and a background worker claims pending
jobs, processes them, and records the outcome.

The job table itself is owned by the host
application: :class:`EmbeddingJobStore` is a
:class:`~typing.Protocol` implemented against any job
table, and :class:`JobProcessor` describes the work
done per job (embedding and upserting — that
processor lives in the host application).

Outbox invariant
----------------
A job row must be committed in the **same transaction**
as the row it refers to. ``enqueue_job(store, payload,
session=...)`` therefore only flushes when a caller
session is supplied; without a session the store opens
its own transaction and commits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

RETRY_SCHEDULE = [
    timedelta(minutes=1),
    timedelta(minutes=5),
    timedelta(minutes=30),
]
"""Backoff applied after the 1st, 2nd and 3rd failure."""

MAX_ATTEMPTS = 4
"""Default attempt ceiling; the job table's column is authoritative."""


@dataclass
class JobPayload:
    """A claimed job, decoupled from any job table.

    ``payload`` carries the document text and metadata
    to embed; every other field is job bookkeeping.
    """

    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    status_detail: str | None = None
    retry_count: int = 0
    max_attempts: int = MAX_ATTEMPTS
    next_retry_at: datetime | None = None
    locked_at: datetime | None = None
    locked_by: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    claim_generation: int = 0
    claim_owner: str | None = None


def claim_jobs(jobs: list[JobPayload], worker_id: str, now: datetime) -> list[JobPayload]:
    """Transition claimed jobs to ``processing`` and stamp the lock.

    A failed job is first reset to ``pending`` so its retry bookkeeping
    starts clean; the lock columns then record which worker claimed it
    and when, for :func:`reclaim_stale_jobs`. Returns the same job list
    for call chaining.
    """
    for job in jobs:
        if job.status == "failed":
            job.status = "pending"
            job.status_detail = None
            job.next_retry_at = None
        job.status = "processing"
        job.locked_at = now
        job.locked_by = worker_id
    return jobs


@runtime_checkable
class EmbeddingJobStore(Protocol):
    """Storage seam for the job outbox, implemented against any job table."""

    async def claim_pending(self, limit: int, worker_id: str) -> list[JobPayload]:
        """Select up to ``limit`` claimable jobs (pending, or failed with
        a due retry and attempts remaining), ordered by creation time,
        using ``FOR UPDATE SKIP LOCKED`` so concurrent workers never
        claim the same row."""
        ...

    async def mark_processing(self, jobs: list[JobPayload], worker_id: str) -> list[JobPayload]:
        """Transition jobs to ``processing`` and stamp the lock.

        Returns the subset that was successfully claimed; a job another
        worker claimed in the meantime is dropped so it is never
        processed twice.
        """
        ...

    async def mark_completed(self, job: JobPayload) -> None:
        """Persist a successfully processed job as ``completed``."""
        ...

    async def mark_failed(self, job: JobPayload, error: Exception) -> None:
        """Persist a failed job's retry/dead-letter transition."""
        ...

    async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
        """Reset jobs abandoned in ``processing`` back to ``pending``."""
        ...

    async def enqueue(self, payload: dict[str, Any], session: Any = None) -> None:
        """Write a new pending job row.

        When ``session`` is supplied the row is only added and flushed —
        the caller owns the commit, so the job and the row it refers to
        land in one transaction. Without a session the store opens its
        own transaction and commits.
        """
        ...


class LostJobOwnershipError(RuntimeError):
    """Stale owner/lease/generation cannot alter an outbox job."""


@runtime_checkable
class AtomicClaimJobStore(Protocol):
    """Optional stronger claim contract for durable multi-worker adapters."""

    async def claim_and_mark_processing(self, limit: int, worker_id: str) -> list[JobPayload]:
        """Claim and transition jobs in one durable transaction."""
        ...


@runtime_checkable
class JobProcessor(Protocol):
    """Per-job work: embed the job's payload and upsert it."""

    async def process(self, job: JobPayload) -> None:
        """Process one claimed job. Raises on failure."""
        ...


async def process_pending_jobs(
    store: EmbeddingJobStore,
    processor: JobProcessor,
    limit: int = 10,
    worker_id: str | None = None,
) -> int:
    """Process pending jobs.

    Claims up to ``limit`` jobs, runs each through ``processor``,
    and records the outcome. Failed jobs are marked with an error
    detail and a retry timestamp; jobs that exhaust their attempts
    move to the terminal ``dead_letter`` state.

    Claimed jobs are stamped with ``locked_at``/``locked_by`` for
    the whole processing transaction. A worker that dies mid-pass
    therefore leaves a visible lock, which :func:`reclaim_stale_jobs`
    later resets.

    Args:
        store: The job store to claim from and update.
        processor: The per-job processor.
        limit: Maximum number of jobs to process in one batch.
        worker_id: Identity stamped into ``locked_by``.

    Returns:
        int: Number of jobs successfully processed.
    """
    if isinstance(store, AtomicClaimJobStore):
        jobs = await store.claim_and_mark_processing(limit, worker_id or "")
    else:
        # Backward compatibility with local/legacy single-host stores. Two
        # separate calls do not promise distributed atomicity.
        jobs = await store.claim_pending(limit, worker_id or "")
        jobs = await store.mark_processing(jobs, worker_id or "")

    processed = 0
    for job in jobs:
        try:
            await processor.process(job)

            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            await store.mark_completed(job)
            processed += 1
            logger.info("Processed job '%s'.", job.job_id)
        except LostJobOwnershipError:
            # Do not turn an ownership conflict into a write by a stale worker.
            raise
        except Exception as exc:
            job.retry_count += 1
            if job.retry_count >= job.max_attempts:
                job.status = "dead_letter"
                job.status_detail = (
                    f"Exhausted {job.max_attempts} attempts ({type(exc).__name__})"
                )
                job.next_retry_at = None
                job.locked_at = None
                job.locked_by = None
            else:
                backoff = RETRY_SCHEDULE[min(job.retry_count - 1, len(RETRY_SCHEDULE) - 1)]
                job.next_retry_at = datetime.now(UTC) + backoff
                job.status = "failed"
                job.status_detail = f"Retry scheduled ({type(exc).__name__})"
                job.locked_at = None
                job.locked_by = None
            await store.mark_failed(job, exc)
            logger.error(
                "Job '%s' failed (retry %d): %s",
                job.job_id,
                job.retry_count,
                type(exc).__name__,
            )

    return processed


async def reclaim_stale_jobs(
    store: EmbeddingJobStore,
    stale_after_seconds: float | None = None,
) -> int:
    """Reset jobs abandoned in ``processing`` back to ``pending``.

    A drainer that dies mid-upsert leaves its rows locked in
    ``processing`` forever: the claim query only selects ``pending``
    or retry-due ``failed`` rows, so no live worker can pick them
    up again. This reclaims rows whose ``locked_at`` is older than
    ``stale_after_seconds`` — or whose lock timestamp is NULL, which
    marks rows locked before the column existed — so a replacement
    worker can process them.

    Args:
        store: The job store to reclaim within.
        stale_after_seconds: Lock age in seconds that counts as
            abandoned. Defaults to the store's own default.

    Returns:
        int: Number of jobs reclaimed.
    """
    return await store.reclaim_stale(stale_after_seconds)


async def enqueue_job(
    store: EmbeddingJobStore,
    payload: dict[str, Any],
    session: Any = None,
) -> None:
    """Add a job to the outbox.

    When ``session`` is supplied the job is only added and flushed;
    the caller owns the commit, so the job and the row it refers to
    land in one transaction. When it is omitted the store opens its
    own session and commits.

    Callers that are writing the referred row must pass their
    session. See the module docstring for why.

    Args:
        store: The job store to enqueue into.
        payload: The document text and metadata to embed.
        session: Optional caller-owned session. Not committed when
            provided.
    """
    await store.enqueue(payload, session=session)


class InMemoryEmbeddingJobStore:
    """In-memory reference implementation of EmbeddingJobStore."""

    def __init__(self, jobs: list[JobPayload] | None = None) -> None:
        self.jobs: list[JobPayload] = list(jobs or [])

    async def enqueue(self, payload: dict[str, Any], session: Any = None) -> None:
        """Enqueue a new job payload."""
        job_id = payload.get("idempotency_key") or f"job-{len(self.jobs) + 1}"
        if any(job.job_id == job_id for job in self.jobs):
            return
        status = payload.get("status", "pending")
        job = JobPayload(
            job_id=job_id,
            payload=payload,
            status=status,
            status_detail=payload.get("status_detail"),
            retry_count=payload.get("retries", 0),
            max_attempts=payload.get("max_attempts", MAX_ATTEMPTS),
            created_at=datetime.now(UTC),
        )
        self.jobs.append(job)

    async def claim_pending(self, limit: int, worker_id: str) -> list[JobPayload]:
        """Claim pending jobs up to limit."""
        now = datetime.now(UTC)
        claimable = [
            j
            for j in self.jobs
            if j.status == "pending"
            or (
                j.status == "failed"
                and j.next_retry_at is not None
                and j.next_retry_at <= now
                and j.retry_count < j.max_attempts
            )
        ]
        return claimable[:limit]

    async def mark_processing(self, jobs: list[JobPayload], worker_id: str) -> list[JobPayload]:
        """Mark jobs as processing with lock stamped."""
        return claim_jobs(jobs, worker_id, datetime.now(UTC))

    async def mark_completed(self, job: JobPayload) -> None:
        """Mark job as completed."""
        job.status = "completed"
        job.completed_at = datetime.now(UTC)

    async def mark_failed(self, job: JobPayload, error: Exception) -> None:
        """Persist the transition already computed by process_pending_jobs."""
        job.status_detail = job.status_detail or str(error)

    async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
        """Reclaim stale jobs."""
        reclaimed = 0
        now = datetime.now(UTC)
        threshold = timedelta(seconds=stale_after_seconds or 300)
        for j in self.jobs:
            if j.status == "processing" and j.locked_at and (now - j.locked_at) > threshold:
                j.status = "pending"
                j.locked_at = None
                j.locked_by = None
                reclaimed += 1
        return reclaimed

    async def get_status_counts(self) -> dict[str, int]:
        """Return counts of jobs by status."""
        counts: dict[str, int] = {}
        for j in self.jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        return counts
