"""Generic job outbox and drainer.

The outbox pattern decouples row insertion from
background processing: job rows are written in the
same transaction as the row they refer to, and a
drainer worker atomically claims pending jobs, processes them, and
records the outcome with retry/backoff/dead-letter
semantics.
"""

from ragnor.jobs.drainer import drain_forever, main
from ragnor.jobs.outbox import (
    MAX_ATTEMPTS,
    RETRY_SCHEDULE,
    EmbeddingJobStore,
    InMemoryEmbeddingJobStore,
    JobPayload,
    JobProcessor,
    claim_jobs,
    enqueue_job,
    process_pending_jobs,
    reclaim_stale_jobs,
)

__all__ = [
    "MAX_ATTEMPTS",
    "RETRY_SCHEDULE",
    "EmbeddingJobStore",
    "InMemoryEmbeddingJobStore",
    "JobPayload",
    "JobProcessor",
    "claim_jobs",
    "drain_forever",
    "enqueue_job",
    "main",
    "process_pending_jobs",
    "reclaim_stale_jobs",
]
