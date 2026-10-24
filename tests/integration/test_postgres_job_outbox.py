"""PostgreSQL 16 real concurrency contract for the optional job-store adapter."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.postgres import (
    DDL,
    PostgresEmbeddingJobStore,
)
from ragnor.jobs.outbox import LostJobOwnershipError, process_pending_jobs

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="requires GitHub Actions PostgreSQL service"),
]


async def _new_store():
    assert DB_URL is not None
    engine = create_async_engine(DB_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        for statement in DDL:
            await connection.execute(text(statement))
    return engine, sessions


async def test_two_hosts_atomically_claim_disjoint_batches() -> None:
    engine, sessions = await _new_store()
    scope = f"test-{uuid4()}"
    a = PostgresEmbeddingJobStore(sessions, scope=scope)
    b = PostgresEmbeddingJobStore(sessions, scope=scope)
    try:
        for i in range(40):
            await a.enqueue({"scope": scope, "idempotency_key": f"job-{i}", "text": "fake"})
        batches = await asyncio.gather(
            a.claim_and_mark_processing(20, "worker-a"),
            b.claim_and_mark_processing(20, "worker-b"),
        )
        all_jobs = [j for batch in batches for j in batch]
        assert len(all_jobs) == 40
        assert len({j.job_id for j in all_jobs}) == 40
        assert all(j.claim_generation == 1 for j in all_jobs)
        assert all(j.claim_owner in {"worker-a", "worker-b"} for j in all_jobs)
        for j in all_jobs:
            await a.mark_completed(j)
        assert await a.claim_and_mark_processing(1, "third") == []
    finally:
        await engine.dispose()


async def test_expired_owner_cannot_ack_after_reclaim_and_retry() -> None:
    engine, sessions = await _new_store()
    scope = f"test-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope, lease_seconds=1)
    try:
        await store.enqueue({"scope": scope, "idempotency_key": "one"})
        first = (await store.claim_and_mark_processing(1, "owner-a"))[0]
        assert first.claim_generation == 1
        # Force expiration using DB time, not flaky sleep.
        async with sessions() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "UPDATE ragnor_embedding_jobs SET lease_expires_at=now()-interval '1 second' "
                        "WHERE job_id=:id"
                    ),
                    {"id": first.job_id},
                )
        assert await store.reclaim_stale() == 1
        second = (await store.claim_and_mark_processing(1, "owner-b"))[0]
        assert second.claim_generation > first.claim_generation
        with pytest.raises(LostJobOwnershipError):
            await store.mark_completed(first)
        await store.mark_completed(second)
    finally:
        await engine.dispose()


async def test_caller_transaction_enqueues_atomically_and_rolls_back() -> None:
    engine, sessions = await _new_store()
    scope = f"test-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope)
    try:
        async with sessions() as session:
            async with session.begin():
                await store.enqueue(
                    {"scope": scope, "idempotency_key": "rolled-back"}, session=session
                )
                assert (
                    await session.execute(
                        text("SELECT count(*) FROM ragnor_embedding_jobs WHERE scope=:scope"),
                        {"scope": scope},
                    )
                ).scalar_one() == 1
                await session.rollback()
        assert await store.claim_and_mark_processing(10, "owner") == []
        await store.enqueue({"scope": scope, "idempotency_key": "one"})
        await store.enqueue({"scope": scope, "idempotency_key": "one"})
        assert len(await store.claim_and_mark_processing(10, "owner")) == 1
    finally:
        await engine.dispose()


async def test_job_runner_uses_atomic_claim_capability() -> None:
    engine, sessions = await _new_store()
    scope = f"test-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope)

    class Processor:
        def __init__(self) -> None:
            self.calls = 0

        async def process(self, job):
            self.calls += 1

    try:
        await store.enqueue({"scope": scope, "idempotency_key": "atomic"})
        processor = Processor()
        assert await process_pending_jobs(store, processor, worker_id="atomic-worker") == 1
        assert processor.calls == 1
        assert await process_pending_jobs(store, processor, worker_id="atomic-worker") == 0
    finally:
        await engine.dispose()


async def test_fenced_lease_renewal_extends_deadline_and_rejects_expired_owner() -> None:
    engine, sessions = await _new_store()
    scope = f"test-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope, lease_seconds=5)
    try:
        await store.enqueue({"scope": scope, "idempotency_key": "lease"})
        job = (await store.claim_and_mark_processing(1, "owner"))[0]
        await store.renew_lease(job, lease_seconds=35)
        async with sessions() as session:
            result = await session.execute(
                text(
                    "SELECT lease_expires_at > now() + interval '20 seconds' "
                    "FROM ragnor_embedding_jobs WHERE job_id=:id"
                ),
                {"id": job.job_id},
            )
            assert result.scalar_one() is True
        async with sessions() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs SET lease_expires_at=now()-interval '1 second' WHERE job_id=:id"
                ),
                {"id": job.job_id},
            )
        with pytest.raises(LostJobOwnershipError):
            await store.renew_lease(job, lease_seconds=35)
    finally:
        await engine.dispose()


async def test_tenants_with_identical_idempotency_keys_are_isolated() -> None:
    engine, sessions = await _new_store()
    left = PostgresEmbeddingJobStore(sessions, scope=f"left-{uuid4()}")
    right = PostgresEmbeddingJobStore(sessions, scope=f"right-{uuid4()}")
    try:
        await left.enqueue({"scope": left.scope, "idempotency_key": "same"})
        await right.enqueue({"scope": right.scope, "idempotency_key": "same"})
        left_job = (await left.claim_and_mark_processing(1, "owner"))[0]
        right_job = (await right.claim_and_mark_processing(1, "owner"))[0]
        assert left_job.job_id != right_job.job_id
        with pytest.raises(LostJobOwnershipError):
            await right.mark_completed(left_job)
        with pytest.raises(LostJobOwnershipError):
            await right.renew_lease(left_job)
        await left.mark_completed(left_job)
        await right.mark_completed(right_job)
        assert await left.get_status_counts() == {"completed": 1}
        assert await right.get_status_counts() == {"completed": 1}
    finally:
        await engine.dispose()
