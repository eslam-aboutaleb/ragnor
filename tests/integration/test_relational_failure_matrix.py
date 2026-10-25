"""One durable-outbox failure contract against both qualified host databases."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.relational import (
    RelationalJobStoreFactory,
    jobs,
    metadata,
)
from ragnor.jobs.outbox import LostJobOwnershipError
from tests.sqlite_host import create_sqlite_engine

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
async def test_same_host_owned_failure_matrix(dialect: str, tmp_path) -> None:  # noqa: PLR0915
    if dialect == "postgresql":
        url = os.environ.get("RAGNOR_TEST_DATABASE_URL")
        if not url:
            pytest.skip("requires a real PostgreSQL service")
        make_engine = create_async_engine
    else:
        url = f"sqlite+aiosqlite:///{tmp_path / 'contract.db'}"
        make_engine = create_sqlite_engine

    first_engine = make_engine(url)
    second_engine = make_engine(url)
    first_sessions = async_sessionmaker(first_engine, expire_on_commit=False)
    second_sessions = async_sessionmaker(second_engine, expire_on_commit=False)
    scope = f"matrix-{uuid4()}"
    other_scope = f"other-{uuid4()}"
    first_factory = RelationalJobStoreFactory(first_sessions)
    second_factory = RelationalJobStoreFactory(second_sessions)
    first = first_factory.job_store(scope=scope, require_publication=True)
    second = second_factory.job_store(scope=scope, require_publication=True)
    other = second_factory.job_store(scope=other_scope)
    try:
        async with first_engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

        payload = {"scope": scope, "idempotency_key": "rollback"}
        with pytest.raises(RuntimeError, match="abort"):
            async with first_sessions() as session, session.begin():
                await first.enqueue(payload, session=session)
                raise RuntimeError("abort")
        assert await first.get_status_counts() == {}

        for number in range(12):
            await first.enqueue({"scope": scope, "idempotency_key": f"job-{number}"})
        await second.enqueue({"scope": scope, "idempotency_key": "job-0"})
        await other.enqueue({"scope": other_scope, "idempotency_key": "job-0"})
        assert await first.get_status_counts() == {"pending": 12}
        assert await other.get_status_counts() == {"pending": 1}

        batches = await asyncio.gather(
            first.claim_and_mark_processing(6, "worker-a"),
            second.claim_and_mark_processing(6, "worker-b"),
        )
        claimed = [job for batch in batches for job in batch]
        assert len(claimed) == 12
        assert len({job.job_id for job in claimed}) == 12
        with pytest.raises(LostJobOwnershipError):
            await other.renew_lease(claimed[0])

        stale = claimed[0]
        async with first_sessions() as session, session.begin():
            await session.execute(
                update(jobs)
                .where(jobs.c.job_id == stale.job_id)
                .values(lease_expires_at=stale.locked_at - timedelta(days=1))
            )
        assert await second.reclaim_stale() == 1
        with pytest.raises(LostJobOwnershipError):
            await first.renew_lease(stale)
        replay = (await second.claim_and_mark_processing(1, "worker-c"))[0]
        assert replay.job_id == stale.job_id
        assert replay.claim_generation > stale.claim_generation

        # Invalidate a real pooled connection and reconnect through the host.
        async with second_sessions() as session:
            connection = await session.connection()
            await connection.invalidate()
        assert await second.get_status_counts() == {"processing": 12}
        # Rebuilding the host's pool also leaves the durable state intact.
        await second_engine.dispose()
        assert await second.get_status_counts() == {"processing": 12}

        reservations = first_factory.reservations(scope=scope)
        await reservations.reserve_generation("document/r2", "a" * 64)
        await reservations.reserve_generation("document/r2", "a" * 64)
        with pytest.raises(ValueError, match="immutable generation content"):
            await second_factory.reservations(scope=scope).reserve_generation(
                "document/r2", "b" * 64
            )

        for revision in (2, 1):
            key = f"publication-{revision}"
            await first.enqueue(
                {
                    "scope": scope,
                    "idempotency_key": key,
                    "document_id": "document",
                    "revision": revision,
                    "generation_key": f"document/r{revision}",
                }
            )
            publication = (await first.claim_and_mark_processing(1, "publisher"))[0]
            assert publication.payload["idempotency_key"] == key
            assert await first.activate_generation(publication) is (revision == 2)
        assert await second.get_active_generation("document") == "document/r2"
        async with second_sessions() as session:
            stored = (
                await session.execute(select(jobs.c.job_id).where(jobs.c.scope == scope))
            ).all()
            assert len(stored) == 14
    finally:
        await first_engine.dispose()
        await second_engine.dispose()
