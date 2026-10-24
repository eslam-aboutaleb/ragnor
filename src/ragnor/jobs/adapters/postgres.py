"""Optional SQLAlchemy/PostgreSQL embedding job outbox adapter.

Requires host-supplied asynchronous SQLAlchemy session factory and an
explicit schema migration. The core jobs protocol never imports SQLAlchemy.

Guarantee: atomic, lease-fenced claims / acknowledgements in PostgreSQL,
with at-least-once execution. This DOES NOT by itself make remote vector
side effects exactly once or safe from a stale processor.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragnor.jobs.outbox import (
    MAX_ATTEMPTS,
    JobPayload,
    LostJobOwnershipError,
)

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
)

CLAIM_SQL = """
WITH picked AS (
    SELECT job_id
      FROM ragnor_embedding_jobs
     WHERE scope = :scope
       AND (
           status = 'pending'
           OR (status = 'failed' AND next_retry_at <= now()
               AND retry_count < max_attempts)
       )
     ORDER BY created_at, job_id
     FOR UPDATE SKIP LOCKED
     LIMIT :limit
)
UPDATE ragnor_embedding_jobs AS job
   SET status='processing', status_detail=NULL, next_retry_at=NULL,
       locked_at=now(),
       lease_expires_at=now() + (:lease_seconds * interval '1 second'),
       locked_by=:worker,
       claim_generation=job.claim_generation + 1
  FROM picked
 WHERE job.job_id=picked.job_id
RETURNING job.*
"""


class PostgresEmbeddingJobStore:
    """Durable scoped PostgreSQL adapter, with no implicit table creation.

    The host is responsible for creating the schema using a versioned
    migration and for closing the engine/session pool. A claim transaction
    commits *before* the embedding processor runs. Completion/failure is
    conditional on an unexpired generation-specific lease.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        scope: str,
        lease_seconds: float = 300.0,
        require_publication: bool = False,
    ) -> None:
        if not scope or not scope.strip():
            raise ValueError("scope must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._sessions = sessions
        self.scope = scope
        self.lease_seconds = lease_seconds
        self.require_publication = require_publication

    async def enqueue(self, payload: dict[str, Any], session: Any = None) -> None:
        if payload.get("scope") != self.scope:
            raise ValueError("job payload scope must match this adapter's scope")
        key = payload.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("durable enqueue requires a non-empty idempotency_key")
        attempts = payload.get("max_attempts", MAX_ATTEMPTS)
        if not isinstance(attempts, int) or attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        params = {
            "id": str(uuid4()),
            "scope": self.scope,
            "key": key,
            "payload": encoded,
            "max_attempts": attempts,
        }
        sql = text(
            """INSERT INTO ragnor_embedding_jobs
               (job_id, scope, idempotency_key, payload, max_attempts)
               VALUES (:id, :scope, :key, CAST(:payload AS jsonb), :max_attempts)
               ON CONFLICT (scope, idempotency_key) DO NOTHING"""
        )
        if session is not None:
            # Caller-owned transaction: SQL executes/flushes, but never commits.
            await session.execute(sql, params)
        else:
            async with self._sessions() as owned, owned.begin():
                await owned.execute(sql, params)

    @staticmethod
    def _from_row(row: Any) -> JobPayload:
        return JobPayload(
            job_id=row["job_id"],
            payload=row["payload"],
            status=row["status"],
            status_detail=row["status_detail"],
            retry_count=row["retry_count"],
            max_attempts=row["max_attempts"],
            next_retry_at=row["next_retry_at"],
            locked_at=row["locked_at"],
            locked_by=row["locked_by"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            claim_generation=row["claim_generation"],
            claim_owner=row["locked_by"],
        )

    async def claim_and_mark_processing(self, limit: int, worker_id: str) -> list[JobPayload]:
        if limit < 0:
            raise ValueError("claim limit must not be negative")
        if not worker_id or not worker_id.strip():
            raise ValueError("worker_id is required for fenced claims")
        if limit == 0:
            return []
        async with self._sessions() as session, session.begin():
            rows = await session.execute(
                text(CLAIM_SQL),
                {
                    "scope": self.scope,
                    "worker": worker_id,
                    "limit": limit,
                    "lease_seconds": self.lease_seconds,
                },
            )
            return [self._from_row(row) for row in rows.mappings().all()]

    async def claim_pending(self, limit: int, worker_id: str) -> list[JobPayload]:
        raise RuntimeError(
            "PostgreSQL claims must use atomic claim_and_mark_processing(), "
            "not a two-step claim_pending()/mark_processing() sequence"
        )

    async def mark_processing(self, jobs: list[JobPayload], worker_id: str) -> list[JobPayload]:
        raise RuntimeError("PostgreSQL claims cannot be transitioned in a separate call")

    def _ownership_params(self, job: JobPayload) -> dict[str, Any]:
        if not job.claim_owner or job.claim_generation < 1:
            raise LostJobOwnershipError("job has no valid claim owner/generation")
        return {
            "scope": self.scope,
            "job_id": job.job_id,
            "owner": job.claim_owner,
            "generation": job.claim_generation,
        }

    async def renew_lease(self, job: JobPayload, *, lease_seconds: float | None = None) -> None:
        """Extend a live lease only for its current scope/owner/generation.

        Uses PostgreSQL server time so host clock skew cannot incorrectly
        renew a reclaimed or expired worker. This method is opt-in: the host
        must invoke it while processing long jobs.
        """
        duration = self.lease_seconds if lease_seconds is None else lease_seconds
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        params = self._ownership_params(job)
        params["lease_seconds"] = duration
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs "
                    "SET lease_expires_at=now() + (:lease_seconds * interval '1 second') "
                    "WHERE scope=:scope AND job_id=:job_id AND status='processing' "
                    "AND locked_by=:owner AND claim_generation=:generation "
                    "AND lease_expires_at>now()"
                ),
                params,
            )
            if result.rowcount != 1:
                raise LostJobOwnershipError("job lease renewal rejected: stale claim")

    async def get_active_generation(self, document_id: str) -> str | None:
        """Read the only visible immutable vector generation for this scope."""
        if not document_id:
            raise ValueError("document_id must be non-empty")
        async with self._sessions() as session:
            result = await session.execute(
                text(
                    "SELECT generation_key FROM ragnor_active_generations "
                    "WHERE scope=:scope AND document_id=:document_id"
                ),
                {"scope": self.scope, "document_id": document_id},
            )
            return result.scalar_one_or_none()

    async def activate_generation(self, job: JobPayload) -> bool:
        """Commit an already-staged vector generation and job ack atomically.

        Staged vectors MUST be keyed by the immutable payload generation_key.
        Readers MUST use get_active_generation() to select a key, never search
        all staged vectors. This method performs no vector writes and requires
        a live, generation-fenced lease; an expired owner cannot activate.
        For a prior source revision, the job completes without rewinding the
        active generation. External side effects are at-least-once.
        """
        params = self._ownership_params(job)
        scope_payload = job.payload.get("scope")
        document_id = job.payload.get("document_id")
        revision = job.payload.get("revision")
        generation_key = job.payload.get("generation_key")
        if scope_payload != self.scope:
            raise LostJobOwnershipError("job scope does not match manifest scope")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be non-empty")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be a non-empty immutable identifier")

        async with self._sessions() as session, session.begin():
            # Lock the authoritative job row before considering publication.
            current = await session.execute(
                text(
                    "SELECT payload FROM ragnor_embedding_jobs WHERE "
                    "scope=:scope AND job_id=:job_id AND status='processing' "
                    "AND locked_by=:owner AND claim_generation=:generation "
                    "AND lease_expires_at>now() FOR UPDATE"
                ),
                params,
            )
            saved = current.scalar_one_or_none()
            if saved is None:
                raise LostJobOwnershipError("publication rejected: expired or stale claim")
            if saved != job.payload:
                raise LostJobOwnershipError(
                    "publication rejected: job payload differs from stored record"
                )
            # Serialize document publication even when it has no manifest row yet.
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": self.scope + ":" + document_id},
            )
            existing = await session.execute(
                text(
                    "SELECT revision, generation_key FROM ragnor_active_generations "
                    "WHERE scope=:scope AND document_id=:document_id"
                ),
                {"scope": self.scope, "document_id": document_id},
            )
            record = existing.mappings().one_or_none()
            activate = record is None or revision > record["revision"]
            same = (
                record is not None
                and revision == record["revision"]
                and generation_key == record["generation_key"]
            )
            if activate:
                await session.execute(
                    text(
                        "INSERT INTO ragnor_active_generations "
                        "(scope, document_id, revision, generation_key) "
                        "VALUES (:scope, :document_id, :revision, :generation_key) "
                        "ON CONFLICT (scope, document_id) DO UPDATE "
                        "SET revision=EXCLUDED.revision, generation_key=EXCLUDED.generation_key, "
                        "activated_at=now() "
                        "WHERE ragnor_active_generations.revision < EXCLUDED.revision"
                    ),
                    {
                        "scope": self.scope,
                        "document_id": document_id,
                        "revision": revision,
                        "generation_key": generation_key,
                    },
                )
            ack = await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs "
                    "SET status='completed', completed_at=now(), "
                    "locked_at=NULL, lease_expires_at=NULL, locked_by=NULL "
                    "WHERE scope=:scope AND job_id=:job_id AND status='processing' "
                    "AND locked_by=:owner AND claim_generation=:generation "
                    "AND lease_expires_at>now()"
                ),
                params,
            )
            if ack.rowcount != 1:
                raise LostJobOwnershipError("publication completion rejected: stale claim")
        job.status = "completed"
        return activate or same

    async def mark_completed(self, job: JobPayload) -> None:
        if self.require_publication:
            raise RuntimeError(
                "strict publication requires activate_generation() instead of direct completion"
            )
        params = self._ownership_params(job)
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs "
                    "SET status='completed', completed_at=now(), "
                    "locked_at=NULL, lease_expires_at=NULL, locked_by=NULL "
                    "WHERE scope=:scope AND job_id=:job_id AND status='processing' "
                    "AND locked_by=:owner AND claim_generation=:generation "
                    "AND lease_expires_at>now()"
                ),
                params,
            )
            if result.rowcount != 1:
                raise LostJobOwnershipError("job completion rejected: stale claim")
        job.status = "completed"

    async def mark_failed(self, job: JobPayload, error: Exception) -> None:
        if job.status not in {"failed", "dead_letter"}:
            raise ValueError("invalid job failure status")
        params = self._ownership_params(job)
        # Exception messages may contain source content, PII or credentials:
        # only the type and fixed retry/terminal status go to the database.
        if job.status == "dead_letter":
            reason = (
                f"Exhausted {job.max_attempts} attempts"
                if job.retry_count >= job.max_attempts
                else "Permanent failure"
            )
            detail = f"{reason} ({type(error).__name__})"
        else:
            detail = f"Retry scheduled ({type(error).__name__})"
        params.update(
            {
                "status": job.status,
                "detail": detail,
                "retries": job.retry_count,
                "next_retry": job.next_retry_at,
            }
        )
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs SET status=:status, "
                    "status_detail=:detail, retry_count=:retries, "
                    "next_retry_at=:next_retry, "
                    "locked_at=NULL, lease_expires_at=NULL, locked_by=NULL "
                    "WHERE scope=:scope AND job_id=:job_id AND status='processing' "
                    "AND locked_by=:owner AND claim_generation=:generation "
                    "AND lease_expires_at>now()"
                ),
                params,
            )
            if result.rowcount != 1:
                raise LostJobOwnershipError("job failure rejected: stale claim")

    async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
        if stale_after_seconds is not None and stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must not be negative")
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs SET status='pending', "
                    "status_detail=NULL, locked_by=NULL, locked_at=NULL, "
                    "lease_expires_at=NULL, claim_generation=claim_generation+1 "
                    "WHERE scope=:scope AND status='processing' "
                    "AND lease_expires_at IS NOT NULL AND lease_expires_at<=now()"
                ),
                {"scope": self.scope},
            )
            return int(result.rowcount)

    async def get_status_counts(self) -> dict[str, int]:
        async with self._sessions() as session:
            rows = await session.execute(
                text(
                    "SELECT status, count(*) AS total FROM ragnor_embedding_jobs "
                    "WHERE scope=:scope GROUP BY status"
                ),
                {"scope": self.scope},
            )
            return {str(row.status): int(row.total) for row in rows}
