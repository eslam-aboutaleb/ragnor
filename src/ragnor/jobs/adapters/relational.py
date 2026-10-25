"""Host-bound, database-portable SQLAlchemy job and publication storage.

The host supplies the async session factory and applies ``metadata`` through
its migration system. This module never constructs an engine or creates tables.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Table,
    UniqueConstraint,
    and_,
    delete,
    func,
    inspect,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragnor.db.schema import ingestion_locks, metadata
from ragnor.jobs.outbox import MAX_ATTEMPTS, JobPayload, LostJobOwnershipError

jobs = Table(
    "ragnor_embedding_jobs",
    metadata,
    Column("job_id", String, primary_key=True),
    Column("scope", String, nullable=False),
    Column("idempotency_key", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("status", String, nullable=False, server_default="pending"),
    Column("retry_count", Integer, nullable=False, server_default="0"),
    Column("max_attempts", Integer, nullable=False, server_default="4"),
    Column("next_retry_at", DateTime(timezone=True)),
    Column("locked_at", DateTime(timezone=True)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("locked_by", String),
    Column("claim_generation", BigInteger, nullable=False, server_default="0"),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
    Column("completed_at", DateTime(timezone=True)),
    Column("status_detail", String),
    UniqueConstraint("scope", "idempotency_key"),
)
Index(
    "ragnor_job_claim_idx",
    jobs.c.scope,
    jobs.c.status,
    jobs.c.next_retry_at,
    jobs.c.created_at,
    jobs.c.job_id,
)
Index("ragnor_job_lease_idx", jobs.c.scope, jobs.c.status, jobs.c.lease_expires_at)

active_generations = Table(
    "ragnor_active_generations",
    metadata,
    Column("scope", String, primary_key=True),
    Column("document_id", String, primary_key=True),
    Column("revision", BigInteger, nullable=False),
    Column("generation_key", String, nullable=False),
    Column(
        "activated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    ),
)

generation_reservations = Table(
    "ragnor_generation_reservations",
    metadata,
    Column("scope", String, primary_key=True),
    Column("generation_key", String, primary_key=True),
    Column("content_digest", String, nullable=False),
)


def _affected(result: Any) -> int:
    count = result.rowcount
    if count is None or count < 0:
        raise RuntimeError("database driver must report conditional UPDATE row counts")
    return int(count)


async def _database_now(session: AsyncSession) -> datetime:
    value = (await session.execute(select(func.current_timestamp()))).scalar_one()
    if not isinstance(value, datetime):
        raise RuntimeError("database must return a timestamp for CURRENT_TIMESTAMP")
    return value


@dataclass(frozen=True)
class RelationalJobStoreFactory:
    """One host-owned database binding shared by SDK components."""

    sessions: async_sessionmaker[AsyncSession]

    def __call__(self) -> AsyncSession:
        return self.sessions()

    def job_store(
        self, *, scope: str, lease_seconds: float = 300.0, require_publication: bool = False
    ) -> RelationalEmbeddingJobStore:
        return RelationalEmbeddingJobStore(
            self.sessions,
            scope=scope,
            lease_seconds=lease_seconds,
            require_publication=require_publication,
        )

    def reservations(self, *, scope: str) -> RelationalGenerationReservations:
        return RelationalGenerationReservations(self.sessions, scope=scope)


class RelationalEmbeddingJobStore:
    """Durable outbox using conditional updates on the host's database."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        scope: str,
        lease_seconds: float = 300.0,
        require_publication: bool = False,
    ) -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be non-empty")
        if not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._sessions = sessions
        self.scope = scope
        self.lease_seconds = lease_seconds
        self.require_publication = require_publication
        self._schema_validated = False

    async def validate_schema(self) -> None:
        """Fail before claiming unless the host has a qualified SQL transaction setup."""
        if self._schema_validated:
            return
        required = (jobs, active_generations, generation_reservations, ingestion_locks)

        def check(connection: Any) -> None:
            dialect = connection.dialect
            if (dialect.name, dialect.driver) not in {
                ("postgresql", "asyncpg"),
                ("sqlite", "aiosqlite"),
            }:
                raise RuntimeError(
                    "distributed outbox is qualified only for PostgreSQL/asyncpg and "
                    "file-backed SQLite/aiosqlite; use a separately qualified host adapter "
                    "for this database"
                )
            isolation = connection.get_isolation_level().upper()
            if dialect.name == "postgresql" and isolation != "READ COMMITTED":
                raise RuntimeError("PostgreSQL outbox requires READ COMMITTED isolation")
            if dialect.name == "sqlite":
                databases = connection.exec_driver_sql("PRAGMA database_list").all()
                if not any(row[1] == "main" and row[2] for row in databases):
                    raise RuntimeError("distributed SQLite outbox requires a file-backed database")
                if isolation != "SERIALIZABLE":
                    raise RuntimeError("SQLite outbox requires SERIALIZABLE isolation")
                if connection.exec_driver_sql("PRAGMA read_uncommitted").scalar() != 0:
                    raise RuntimeError("SQLite outbox requires read_uncommitted=OFF")
                if connection.exec_driver_sql("PRAGMA busy_timeout").scalar() <= 0:
                    raise RuntimeError("SQLite outbox requires a positive busy_timeout")
            if not connection.dialect.supports_sane_rowcount:
                raise RuntimeError("database driver must report conditional UPDATE row counts")
            inspector = inspect(connection)
            missing: list[str] = []
            for table in required:
                if not inspector.has_table(table.name):
                    missing.append(table.name)
                    continue
                actual = {column["name"] for column in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name not in actual:
                        missing.append(f"{table.name}.{column.name}")
                primary = set(
                    inspector.get_pk_constraint(table.name).get("constrained_columns") or ()
                )
                expected_primary = {column.name for column in table.primary_key.columns}
                if primary != expected_primary:
                    missing.append(f"{table.name} primary key")
            if inspector.has_table(jobs.name):
                unique_sets = {
                    frozenset(item.get("column_names") or ())
                    for item in inspector.get_unique_constraints(jobs.name)
                }
                if frozenset(("scope", "idempotency_key")) not in unique_sets:
                    missing.append(
                        "ragnor_embedding_jobs(scope, idempotency_key) unique constraint"
                    )
            if missing:
                raise RuntimeError(
                    "host database schema is missing required outbox objects: " + ", ".join(missing)
                )

        async with self._sessions() as session:
            connection = await session.connection()
            await connection.run_sync(check)
            dialect_name = connection.dialect.name
        if dialect_name == "sqlite":
            await self._validate_savepoint_rollback()
        self._schema_validated = True

    async def _validate_savepoint_rollback(self) -> None:
        """Prove SQLite SAVEPOINT writes are rolled back by the outer transaction."""
        probe_key = f"ragnor-rollback-probe-{uuid4()}"
        async with self._sessions() as session, session.begin():
            async with session.begin_nested():
                await session.execute(insert(ingestion_locks).values(lock_key=probe_key))
            await session.rollback()
        async with self._sessions() as session:
            persisted = (
                await session.execute(
                    select(ingestion_locks.c.lock_key).where(
                        ingestion_locks.c.lock_key == probe_key
                    )
                )
            ).scalar_one_or_none()
        if persisted is not None:
            async with self._sessions() as session, session.begin():
                await session.execute(
                    delete(ingestion_locks).where(ingestion_locks.c.lock_key == probe_key)
                )
            raise RuntimeError(
                "SQLite host transaction control must keep savepoints inside outer rollback"
            )

    @staticmethod
    def _from_row(row: Any) -> JobPayload:
        record = row._mapping
        return JobPayload(
            job_id=record["job_id"],
            payload=record["payload"],
            status=record["status"],
            status_detail=record["status_detail"],
            retry_count=record["retry_count"],
            max_attempts=record["max_attempts"],
            next_retry_at=record["next_retry_at"],
            locked_at=record["locked_at"],
            locked_by=record["locked_by"],
            created_at=record["created_at"],
            completed_at=record["completed_at"],
            claim_generation=record["claim_generation"],
            claim_owner=record["locked_by"],
        )

    async def enqueue(self, payload: dict[str, Any], session: Any = None) -> None:
        if payload.get("scope") != self.scope:
            raise ValueError("job payload scope must match this adapter's scope")
        key = payload.get("idempotency_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("durable enqueue requires a non-empty idempotency_key")
        attempts = payload.get("max_attempts", MAX_ATTEMPTS)
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        encoded = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        values = {
            "job_id": str(uuid4()),
            "scope": self.scope,
            "idempotency_key": key,
            "payload": encoded,
            "max_attempts": attempts,
        }

        async def write(target: AsyncSession) -> None:
            try:
                async with target.begin_nested():
                    await target.execute(insert(jobs).values(**values))
            except IntegrityError:
                existing = await target.execute(
                    select(jobs.c.job_id).where(
                        jobs.c.scope == self.scope, jobs.c.idempotency_key == key
                    )
                )
                if existing.scalar_one_or_none() is None:
                    raise

        if session is not None:
            await write(session)
        else:
            async with self._sessions() as owned, owned.begin():
                await write(owned)

    @staticmethod
    def _due(now: datetime) -> Any:
        return or_(
            jobs.c.status == "pending",
            and_(
                jobs.c.status == "failed",
                jobs.c.next_retry_at <= now,
                jobs.c.retry_count < jobs.c.max_attempts,
            ),
        )

    async def claim_and_mark_processing(self, limit: int, worker_id: str) -> list[JobPayload]:
        if limit < 0:
            raise ValueError("claim limit must not be negative")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id is required for fenced claims")
        if limit == 0:
            return []
        await self.validate_schema()
        for attempt in range(20):
            try:
                async with self._sessions() as session, session.begin():
                    now = await _database_now(session)
                    chosen: list[JobPayload] = []
                    seen: set[str] = set()
                    while len(chosen) < limit:
                        candidates = await session.execute(
                            select(jobs.c.job_id, jobs.c.claim_generation)
                            .where(
                                jobs.c.scope == self.scope,
                                self._due(now),
                                jobs.c.job_id.not_in(seen),
                            )
                            .order_by(jobs.c.created_at, jobs.c.job_id)
                            .limit(limit - len(chosen))
                        )
                        batch = candidates.all()
                        if not batch:
                            break
                        for job_id, generation in batch:
                            job_id = cast(str, job_id)
                            seen.add(job_id)
                            result = await session.execute(
                                update(jobs)
                                .where(
                                    jobs.c.scope == self.scope,
                                    jobs.c.job_id == job_id,
                                    jobs.c.claim_generation == generation,
                                    self._due(now),
                                )
                                .values(
                                    status="processing",
                                    status_detail=None,
                                    next_retry_at=None,
                                    locked_at=now,
                                    lease_expires_at=now + timedelta(seconds=self.lease_seconds),
                                    locked_by=worker_id,
                                    claim_generation=jobs.c.claim_generation + 1,
                                )
                            )
                            if _affected(result) == 1:
                                row = (
                                    await session.execute(
                                        select(jobs).where(jobs.c.job_id == job_id)
                                    )
                                ).one()
                                chosen.append(self._from_row(row))
                    return chosen
            except OperationalError:
                if attempt == 19:
                    raise
                jitter = 0.5 + secrets.randbelow(1000) / 1000
                await asyncio.sleep(min(0.025 * 2**attempt, 0.5) * jitter)
        return []

    async def claim_pending(self, limit: int, worker_id: str) -> list[JobPayload]:
        raise RuntimeError("durable claims require atomic claim_and_mark_processing()")

    async def mark_processing(
        self, jobs_to_mark: list[JobPayload], worker_id: str
    ) -> list[JobPayload]:
        raise RuntimeError("durable claims cannot be transitioned in a separate call")

    def _owned(self, job: JobPayload) -> Any:
        if not job.claim_owner or job.claim_generation < 1:
            raise LostJobOwnershipError("job has no valid claim owner/generation")
        return and_(
            jobs.c.scope == self.scope,
            jobs.c.job_id == job.job_id,
            jobs.c.status == "processing",
            jobs.c.locked_by == job.claim_owner,
            jobs.c.claim_generation == job.claim_generation,
            jobs.c.lease_expires_at > func.current_timestamp(),
        )

    async def renew_lease(self, job: JobPayload, *, lease_seconds: float | None = None) -> None:
        duration = self.lease_seconds if lease_seconds is None else lease_seconds
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        predicate = self._owned(job)
        async with self._sessions() as session, session.begin():
            now = await _database_now(session)
            result = await session.execute(
                update(jobs)
                .where(predicate)
                .values(lease_expires_at=now + timedelta(seconds=duration))
            )
            if _affected(result) != 1:
                raise LostJobOwnershipError("job lease renewal rejected: stale claim")

    async def get_active_generation(self, document_id: str) -> str | None:
        if not document_id:
            raise ValueError("document_id must be non-empty")
        async with self._sessions() as session:
            return (
                await session.execute(
                    select(active_generations.c.generation_key).where(
                        active_generations.c.scope == self.scope,
                        active_generations.c.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()

    async def _complete(self, session: AsyncSession, job: JobPayload) -> None:
        result = await session.execute(
            update(jobs)
            .where(self._owned(job))
            .values(
                status="completed",
                completed_at=func.current_timestamp(),
                locked_at=None,
                lease_expires_at=None,
                locked_by=None,
            )
        )
        if _affected(result) != 1:
            raise LostJobOwnershipError("job completion rejected: stale claim")

    async def activate_generation(self, job: JobPayload) -> bool:
        self._owned(job)
        payload = job.payload
        document_id = payload.get("document_id")
        revision = payload.get("revision")
        generation_key = payload.get("generation_key")
        if payload.get("scope") != self.scope:
            raise LostJobOwnershipError("job scope does not match manifest scope")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be non-empty")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be a non-empty immutable identifier")
        async with self._sessions() as session, session.begin():
            saved = (
                await session.execute(
                    select(jobs.c.payload).where(
                        jobs.c.scope == self.scope, jobs.c.job_id == job.job_id
                    )
                )
            ).scalar_one_or_none()
            if saved != payload:
                raise LostJobOwnershipError(
                    "publication rejected: job payload differs from stored record"
                )
            await self._complete(session, job)
            predicate = and_(
                active_generations.c.scope == self.scope,
                active_generations.c.document_id == document_id,
            )
            record = (
                await session.execute(
                    select(
                        active_generations.c.revision, active_generations.c.generation_key
                    ).where(predicate)
                )
            ).one_or_none()
            if record is None:
                try:
                    async with session.begin_nested():
                        await session.execute(
                            insert(active_generations).values(
                                scope=self.scope,
                                document_id=document_id,
                                revision=revision,
                                generation_key=generation_key,
                            )
                        )
                    activated = True
                except IntegrityError:
                    record = (
                        await session.execute(
                            select(
                                active_generations.c.revision, active_generations.c.generation_key
                            ).where(predicate)
                        )
                    ).one()
                    activated = False
            else:
                activated = False
            if record is not None and record.revision < revision:
                result = await session.execute(
                    update(active_generations)
                    .where(predicate, active_generations.c.revision < revision)
                    .values(
                        revision=revision,
                        generation_key=generation_key,
                        activated_at=func.current_timestamp(),
                    )
                )
                activated = _affected(result) == 1
            if activated:
                visible = True
            else:
                latest = (
                    await session.execute(
                        select(
                            active_generations.c.revision, active_generations.c.generation_key
                        ).where(predicate)
                    )
                ).one()
                visible = latest.revision == revision and latest.generation_key == generation_key
        job.status = "completed"
        return visible

    async def mark_completed(self, job: JobPayload) -> None:
        if self.require_publication:
            raise RuntimeError(
                "strict publication requires activate_generation() instead of direct completion"
            )
        async with self._sessions() as session, session.begin():
            await self._complete(session, job)
        job.status = "completed"

    async def mark_failed(self, job: JobPayload, error: Exception) -> None:
        if job.status not in {"failed", "dead_letter"}:
            raise ValueError("invalid job failure status")
        predicate = self._owned(job)
        if job.status == "dead_letter":
            reason = (
                f"Exhausted {job.max_attempts} attempts"
                if job.retry_count >= job.max_attempts
                else "Permanent failure"
            )
            detail = f"{reason} ({type(error).__name__})"
        else:
            detail = f"Retry scheduled ({type(error).__name__})"
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                update(jobs)
                .where(predicate)
                .values(
                    status=job.status,
                    status_detail=detail,
                    retry_count=job.retry_count,
                    next_retry_at=job.next_retry_at,
                    locked_at=None,
                    lease_expires_at=None,
                    locked_by=None,
                )
            )
            if _affected(result) != 1:
                raise LostJobOwnershipError("job failure rejected: stale claim")

    async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
        if stale_after_seconds is not None and stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must not be negative")
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                update(jobs)
                .where(
                    jobs.c.scope == self.scope,
                    jobs.c.status == "processing",
                    jobs.c.lease_expires_at.is_not(None),
                    jobs.c.lease_expires_at <= func.current_timestamp(),
                )
                .values(
                    status="pending",
                    status_detail=None,
                    locked_by=None,
                    locked_at=None,
                    lease_expires_at=None,
                    claim_generation=jobs.c.claim_generation + 1,
                )
            )
            return _affected(result)

    async def get_status_counts(self) -> dict[str, int]:
        async with self._sessions() as session:
            rows = await session.execute(
                select(jobs.c.status, func.count())
                .where(jobs.c.scope == self.scope)
                .group_by(jobs.c.status)
            )
            return {str(status): cast(int, total) for status, total in rows}


class RelationalGenerationReservations:
    """Committed, scoped immutable-generation content reservations."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, scope: str) -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be nonempty")
        self._sessions = sessions
        self.scope = scope

    async def reserve_generation(self, generation_key: str, content_digest: str) -> None:
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be nonempty")
        if not isinstance(content_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", content_digest):
            raise ValueError("content_digest must be a SHA-256 digest")
        async with self._sessions() as session, session.begin():
            try:
                async with session.begin_nested():
                    await session.execute(
                        insert(generation_reservations).values(
                            scope=self.scope,
                            generation_key=generation_key,
                            content_digest=content_digest,
                        )
                    )
            except IntegrityError:
                existing = (
                    await session.execute(
                        select(generation_reservations.c.content_digest).where(
                            generation_reservations.c.scope == self.scope,
                            generation_reservations.c.generation_key == generation_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing != content_digest:
                    raise ValueError("immutable generation content mismatch on replay") from None
