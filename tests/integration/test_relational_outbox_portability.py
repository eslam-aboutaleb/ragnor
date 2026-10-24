"""The same host-bound outbox runs on a real non-PostgreSQL database file."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import Column, MetaData, Table, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from ragnor import RagnorBuilder
from ragnor.jobs.adapters.relational import (
    RelationalJobStoreFactory,
    jobs,
    metadata,
)
from ragnor.jobs.outbox import LostJobOwnershipError
from tests.sqlite_host import create_sqlite_engine

pytestmark = pytest.mark.asyncio


class _Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def test_missing_host_schema_fails_before_claim(tmp_path):
    engine = create_sqlite_engine(f"sqlite+aiosqlite:///{tmp_path / 'unmigrated.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = RelationalJobStoreFactory(sessions).job_store(scope="tenant")
    try:
        with pytest.raises(RuntimeError, match="host database schema is missing"):
            await store.claim_and_mark_processing(1, "worker")
    finally:
        await engine.dispose()


async def test_host_schema_without_scoped_unique_constraint_fails_before_claim(tmp_path):
    engine = create_sqlite_engine(f"sqlite+aiosqlite:///{tmp_path / 'unsafe.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = RelationalJobStoreFactory(sessions).job_store(scope="tenant")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
            await connection.run_sync(jobs.drop)
            unsafe_jobs = Table(
                jobs.name,
                MetaData(),
                *(
                    Column(column.name, column.type, primary_key=column.primary_key)
                    for column in jobs.columns
                ),
            )
            await connection.run_sync(unsafe_jobs.create)
        with pytest.raises(RuntimeError, match="unique constraint"):
            await store.claim_and_mark_processing(1, "worker")
    finally:
        await engine.dispose()


async def test_builder_binds_one_host_database_to_ingestion_and_outbox(tmp_path):
    engine = create_sqlite_engine(f"sqlite+aiosqlite:///{tmp_path / 'builder.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    database = RelationalJobStoreFactory(sessions)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        rag = await (
            RagnorBuilder.local(_Embedder(), embedding_dim=2)
            .with_job_persistence(database, scope="tenant")
            .build()
        )
        assert rag.pipeline._session_factory is database
        assert rag.retriever._session_factory is database
        assert rag.job_store is not None
        await rag.job_store.enqueue({"scope": "tenant", "idempotency_key": "builder-job"})
        assert await rag.job_store.get_status_counts() == {"pending": 1}
    finally:
        await engine.dispose()


async def test_host_owned_sqlite_database_runs_outbox_and_publication(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'host.db'}"
    engine = create_sqlite_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    database = RelationalJobStoreFactory(sessions)
    scope = "tenant-a"
    store = database.job_store(scope=scope, lease_seconds=30, require_publication=True)
    reservations = database.reservations(scope=scope)
    try:
        # Schema creation is explicitly performed by the host, never by the SDK.
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        payload = {
            "scope": scope,
            "idempotency_key": "doc-1",
            "document_id": "doc",
            "revision": 1,
            "generation_key": "doc/rev-1",
        }
        with pytest.raises(RuntimeError, match="host rollback"):
            async with sessions() as session, session.begin():
                await store.enqueue(payload, session=session)
                raise RuntimeError("host rollback")
        async with sessions() as session:
            assert (await session.execute(select(jobs.c.job_id))).all() == []
        await store.enqueue(payload)
        digest = "a" * 64
        await reservations.reserve_generation("doc/rev-1", digest)
        await reservations.reserve_generation("doc/rev-1", digest)
        with pytest.raises(ValueError, match="immutable generation content"):
            await reservations.reserve_generation("doc/rev-1", "b" * 64)
        job = (await store.claim_and_mark_processing(1, "worker-a"))[0]
        assert job.claim_generation == 1
        assert await store.get_active_generation("doc") is None
        assert await store.activate_generation(job)
        assert await store.get_active_generation("doc") == "doc/rev-1"
        assert await store.get_status_counts() == {"completed": 1}
    finally:
        await engine.dispose()


async def test_two_host_connections_claim_distinct_sqlite_jobs_and_fence_replay(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}"
    first_engine = create_sqlite_engine(url)
    second_engine = create_sqlite_engine(url)
    first_sessions = async_sessionmaker(first_engine, expire_on_commit=False)
    second_sessions = async_sessionmaker(second_engine, expire_on_commit=False)
    first = RelationalJobStoreFactory(first_sessions).job_store(scope="tenant", lease_seconds=30)
    second = RelationalJobStoreFactory(second_sessions).job_store(scope="tenant", lease_seconds=30)
    try:
        async with first_engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        for number in range(20):
            await first.enqueue({"scope": "tenant", "idempotency_key": str(number)})
        batches = await asyncio.gather(
            first.claim_and_mark_processing(10, "worker-a"),
            second.claim_and_mark_processing(10, "worker-b"),
        )
        claimed = [job for batch in batches for job in batch]
        assert len(claimed) == 20
        assert len({job.job_id for job in claimed}) == 20
        stale = claimed[0]
        async with first_sessions() as session, session.begin():
            await session.execute(
                update(jobs)
                .where(jobs.c.job_id == stale.job_id)
                .values(lease_expires_at=stale.locked_at - timedelta(seconds=1))
            )
        assert await second.reclaim_stale() == 1
        with pytest.raises(LostJobOwnershipError):
            await first.mark_completed(stale)
        replay = (await second.claim_and_mark_processing(1, "worker-c"))[0]
        assert replay.job_id == stale.job_id
        assert replay.claim_generation > stale.claim_generation
        await second.mark_completed(replay)
        async with second_sessions() as session:
            assert (
                await session.execute(select(jobs.c.status).where(jobs.c.job_id == replay.job_id))
            ).scalar_one() == "completed"
    finally:
        await first_engine.dispose()
        await second_engine.dispose()
