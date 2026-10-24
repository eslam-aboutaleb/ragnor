"""Compatibility name and migration assets for the original PostgreSQL outbox.

The implementation now uses the same SQLAlchemy Core statements as other dialects.
The host still applies these legacy migrations to existing PostgreSQL deployments.
"""

from __future__ import annotations

from ragnor.jobs.adapters.relational import RelationalEmbeddingJobStore

DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS ragnor_embedding_jobs (
        job_id text PRIMARY KEY,
        scope text NOT NULL,
        idempotency_key text NOT NULL,
        payload jsonb NOT NULL,
        status text NOT NULL DEFAULT 'pending',
        retry_count integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 4,
        next_retry_at timestamptz,
        locked_at timestamptz,
        lease_expires_at timestamptz,
        locked_by text,
        claim_generation bigint NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        status_detail text,
        UNIQUE(scope, idempotency_key)
    )""",
    """CREATE INDEX IF NOT EXISTS ragnor_job_claim_idx
       ON ragnor_embedding_jobs(scope, status, next_retry_at, created_at, job_id)""",
    """CREATE INDEX IF NOT EXISTS ragnor_job_lease_idx
       ON ragnor_embedding_jobs(scope, status, lease_expires_at)""",
    """CREATE TABLE IF NOT EXISTS ragnor_active_generations (
        scope text NOT NULL,
        document_id text NOT NULL,
        revision bigint NOT NULL,
        generation_key text NOT NULL,
        activated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY(scope, document_id)
    )""",
    """CREATE TABLE IF NOT EXISTS ragnor_generation_reservations (
    scope text NOT NULL,
    generation_key text NOT NULL,
    content_digest text NOT NULL,
    PRIMARY KEY(scope, generation_key)
)""",
    """CREATE TABLE IF NOT EXISTS ragnor_ingestion_locks (
    lock_key text PRIMARY KEY,
    generation integer NOT NULL DEFAULT 0
)""",
)


class PostgresEmbeddingJobStore(RelationalEmbeddingJobStore):
    """Compatibility alias for the host-bound SQLAlchemy job store."""
