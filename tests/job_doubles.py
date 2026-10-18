"""In-memory doubles for the ragnor job outbox tests.

Shared by the evaluation/jobs test modules: a fake
:class:`~ragnor.jobs.outbox.EmbeddingJobStore` and
:class:`~ragnor.jobs.outbox.JobProcessor` that prove
the job engine has no host dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ragnor.jobs.outbox import JobPayload, claim_jobs


class FakeJobStore:
    """In-memory ``EmbeddingJobStore`` backed by a list of payloads."""

    def __init__(self, jobs: list[JobPayload] | None = None) -> None:
        self.jobs: list[JobPayload] = list(jobs or [])
        self.enqueued_payloads: list[dict[str, Any]] = []
        self.committed_payloads: list[dict[str, Any]] = []
        self.claim_limits: list[int] = []
        self.claim_worker_ids: list[str] = []
        self.reclaim_calls: list[float | None] = []

    def add(self, **overrides: Any) -> JobPayload:
        """Append a new pending job and return it."""
        defaults: dict[str, Any] = {
            "job_id": f"job-{len(self.jobs) + 1}",
            "payload": {"claim_id": "CLM-1"},
            "status": "pending",
            "status_detail": "pending",
            "retry_count": 0,
            "max_attempts": 4,
            "next_retry_at": None,
            "locked_at": None,
            "locked_by": None,
            "created_at": datetime.now(UTC),
            "completed_at": None,
        }
        defaults.update(overrides)
        job = JobPayload(**defaults)
        self.jobs.append(job)
        return job

    async def claim_pending(self, limit: int, worker_id: str) -> list[JobPayload]:
        self.claim_limits.append(limit)
        self.claim_worker_ids.append(worker_id)
        now = datetime.now(UTC)
        claimable = [
            job
            for job in self.jobs
            if job.status == "pending"
            or (
                job.status == "failed"
                and job.next_retry_at is not None
                and job.next_retry_at <= now
                and job.retry_count < job.max_attempts
            )
        ]
        claimable.sort(key=lambda job: job.created_at or datetime.min)
        return claimable[:limit]

    async def mark_processing(self, jobs: list[JobPayload], worker_id: str) -> list[JobPayload]:
        claim_jobs(jobs, worker_id, datetime.now(UTC))
        return jobs

    async def mark_completed(self, job: JobPayload) -> None:
        # The payload is already transitioned by the orchestrator.
        return None

    async def mark_failed(self, job: JobPayload, error: Exception) -> None:
        # The payload is already transitioned by the orchestrator.
        return None

    async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
        self.reclaim_calls.append(stale_after_seconds)
        cutoff = datetime.now(UTC) - timedelta(
            seconds=stale_after_seconds if stale_after_seconds is not None else 300.0
        )
        reclaimed = 0
        for job in self.jobs:
            if job.status == "processing" and (job.locked_at is None or job.locked_at <= cutoff):
                job.status = "pending"
                job.status_detail = None
                job.locked_at = None
                job.locked_by = None
                reclaimed += 1
        return reclaimed

    async def enqueue(self, payload: dict[str, Any], session: Any = None) -> None:
        self.enqueued_payloads.append(payload)
        if session is not None:
            # Outbox invariant: a caller-owned session is only flushed,
            # never committed, by the enqueue.
            await session.flush()
        else:
            # Without a session the store opens its own transaction and
            # commits; the in-memory double records the durable write.
            self.committed_payloads.append(payload)


class FakeJobProcessor:
    """In-memory ``JobProcessor`` that records processed jobs."""

    def __init__(self, error: Exception | None = None) -> None:
        self.processed: list[JobPayload] = []
        self.error = error

    async def process(self, job: JobPayload) -> None:
        if self.error is not None:
            raise self.error
        self.processed.append(job)


class RecordingSession:
    """Session double that records add/flush/commit calls."""

    def __init__(self) -> None:
        self.commits = 0
        self.flushes = 0
        self.added: list[object] = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1
