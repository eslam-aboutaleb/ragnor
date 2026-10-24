"""Host-created SQLAlchemy lock table works across independent SQLite engines."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ragnor.db.schema import ingestion_locks
from ragnor.db.sql_factory import SQLAlchemyFactory
from tests.sqlite_host import create_sqlite_engine

pytestmark = pytest.mark.asyncio


async def test_two_host_connections_serialize_source_ingestion(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}"
    engine_a = create_sqlite_engine(url)
    engine_b = create_sqlite_engine(url)
    sessions_a = async_sessionmaker(engine_a, expire_on_commit=False)
    sessions_b = async_sessionmaker(engine_b, expire_on_commit=False)
    factory = SQLAlchemyFactory()
    entered = asyncio.Event()

    async def contender():
        async with sessions_b() as session:
            async with factory.advisory_lock(session, "source-a"):
                entered.set()
                await session.commit()

    try:
        async with engine_a.begin() as connection:
            await connection.run_sync(ingestion_locks.create, checkfirst=True)
        async with sessions_a() as leader:
            async with factory.advisory_lock(leader, "source-a"):
                task = asyncio.create_task(contender())
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(entered.wait(), timeout=0.1)
                assert not entered.is_set()
                await leader.commit()
            await asyncio.wait_for(task, timeout=5)
        assert entered.is_set()
        async with sessions_a() as session:
            assert (await session.execute(select(ingestion_locks.c.generation))).scalar_one() == 1
    finally:
        await engine_a.dispose()
        await engine_b.dispose()
