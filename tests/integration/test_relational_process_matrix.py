"""Independent worker processes race and recover on each qualified SQL database."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.relational import RelationalJobStoreFactory, jobs, metadata
from tests.sqlite_host import create_sqlite_engine

pytestmark = pytest.mark.asyncio
WORKER = Path(__file__).resolve().parent / "fixtures" / "relational_claim_worker.py"


async def start_worker(url: str, scope: str, worker: str, count: int, mode: str):
    root = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join((str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(WORKER),
        scope,
        worker,
        str(count),
        mode,
        env={**os.environ, "RAGNOR_CONTRACT_DATABASE_URL": url, "PYTHONPATH": python_path},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert await asyncio.wait_for(process.stdout.readline(), 15) == b"READY\n"
    return process


async def release(process) -> list[dict[str, object]]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(b"GO\n")
    await process.stdin.drain()
    return json.loads((await asyncio.wait_for(process.stdout.readline(), 20)).decode())


@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
async def test_process_race_and_killed_worker_replay(dialect: str, tmp_path) -> None:
    if dialect == "postgresql":
        url = os.environ.get("RAGNOR_TEST_DATABASE_URL")
        if not url:
            pytest.skip("requires a real PostgreSQL service")
        engine = create_async_engine(url)
    else:
        url = f"sqlite+aiosqlite:///{tmp_path / 'process.db'}"
        engine = create_sqlite_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    scope = f"process-{uuid4()}"
    store = RelationalJobStoreFactory(sessions).job_store(scope=scope)
    children = []
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        for number in range(20):
            await store.enqueue({"scope": scope, "idempotency_key": str(number)})
        a = await start_worker(url, scope, "a", 10, "hold")
        b = await start_worker(url, scope, "b", 10, "exit")
        children.extend((a, b))
        left, right = await asyncio.gather(release(a), release(b))
        assert len(left) + len(right) == 20
        assert len({item["id"] for item in left + right}) == 20
        assert all(item["epoch"] == 1 for item in left + right)
        await asyncio.wait_for(b.wait(), 10)
        assert b.returncode == 0

        a.kill()
        await asyncio.wait_for(a.wait(), 10)
        assert a.returncode != 0
        stale = left[0]
        async with sessions() as session, session.begin():
            await session.execute(
                update(jobs)
                .where(jobs.c.job_id == stale["id"])
                .values(lease_expires_at=jobs.c.locked_at - timedelta(days=1))
            )
        assert await store.reclaim_stale() == 1
        replay = (await store.claim_and_mark_processing(1, "replacement"))[0]
        assert replay.job_id == stale["id"]
        assert replay.claim_generation > stale["epoch"]
    finally:
        for process in children:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await engine.dispose()
