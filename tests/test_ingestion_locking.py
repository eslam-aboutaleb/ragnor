"""The public ingestion lock helper uses the host's SQLAlchemy transaction."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from ragnor.db.schema import ingestion_locks
from ragnor.ingestion import advisory_lock
from tests.sqlite_host import create_sqlite_engine


@pytest.mark.asyncio
async def test_helper_releases_a_failed_host_transaction(tmp_path) -> None:
    engine = create_sqlite_engine(f"sqlite+aiosqlite:///{tmp_path / 'ingestion.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(ingestion_locks.create, checkfirst=True)
        async with sessions() as session:

            async def abort_ingestion() -> None:
                async with advisory_lock(session, "source-a"):
                    raise RuntimeError("failed ingestion")

            with pytest.raises(RuntimeError, match="failed ingestion"):
                await abort_ingestion()
            async with advisory_lock(session, "source-a"):
                await session.commit()
    finally:
        await engine.dispose()
