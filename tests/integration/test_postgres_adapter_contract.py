"""Real PostgreSQL integration tests for SQL port and advisory-lock semantics.

These tests require a live database and are executed in GitHub Actions.
They do NOT prove job-outbox atomic claiming (#26).
"""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.db.sql_factory import SQLAlchemyFactory
from ragnor.db.schema import ingestion_locks

DATABASE_URL = os.getenv("RAGNOR_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="Real PostgreSQL test URL is required"),
]


async def test_injected_sql_factory_persists_then_reads_transactional_values() -> None:
    assert DATABASE_URL is not None
    factory = SQLAlchemyFactory()
    engine = create_async_engine(DATABASE_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS ragnor_sdk_arch_contract ("
                    "id integer PRIMARY KEY, value text NOT NULL)"
                )
            )
            await connection.execute(text("DELETE FROM ragnor_sdk_arch_contract"))

        async with sessions() as session:
            await factory.execute(
                session,
                "INSERT INTO ragnor_sdk_arch_contract (id, value) VALUES (:id, :value)",
                {"id": 1, "value": "host-storage"},
            )
            await session.commit()

        async with sessions() as session:
            found = await factory.fetch_scalar(
                session, "SELECT value FROM ragnor_sdk_arch_contract WHERE id=:id", {"id": 1}
            )
            assert found == "host-storage"
    finally:
        await engine.dispose()


async def test_postgres_advisory_lock_serializes_two_independent_sessions() -> None:
    assert DATABASE_URL is not None
    engine = create_async_engine(DATABASE_URL)
    factory = SQLAlchemyFactory()
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(ingestion_locks.create, checkfirst=True)
    contender_started = asyncio.Event()
    contender_entered = asyncio.Event()

    async def contender() -> None:
        async with sessions() as session:
            contender_started.set()
            async with factory.advisory_lock(session, "arch-shared-scope"):
                contender_entered.set()

    task: asyncio.Task[None] | None = None
    try:
        async with sessions() as leader:
            async with factory.advisory_lock(leader, "arch-shared-scope"):
                task = asyncio.create_task(contender())
                await asyncio.wait_for(contender_started.wait(), timeout=5)
                # A blocked PostgreSQL advisory lock cannot enter while held.
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(contender_entered.wait(), timeout=0.2)
                assert not contender_entered.is_set()

        assert task is not None
        await asyncio.wait_for(task, timeout=5)
        assert contender_entered.is_set()
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await engine.dispose()


async def test_failed_statement_does_not_poison_injected_session_or_leak_lock() -> None:
    assert DATABASE_URL is not None
    engine = create_async_engine(DATABASE_URL)
    factory = SQLAlchemyFactory()
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(ingestion_locks.create, checkfirst=True)
    try:
        async with sessions() as session:
            with pytest.raises(Exception):  # noqa: B017
                async with factory.advisory_lock(session, "arch-rollback"):
                    await factory.execute(session, "SELECT 1 / 0")

            # The exception path must rollback BEFORE unlocking, and the
            # same session must remain usable afterwards.
            result = await factory.fetch_scalar(session, "SELECT 1")
            assert result == 1

        async with sessions() as second:
            async with factory.advisory_lock(second, "arch-rollback"):
                assert await factory.fetch_scalar(second, "SELECT 2") == 2
    finally:
        await engine.dispose()
