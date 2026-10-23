"""Real subprocess PostgreSQL claim races and worker-crash recovery."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

from ragnor.jobs.adapters.postgres import PostgresEmbeddingJobStore
from ragnor.jobs.outbox import JobPayload, LostJobOwnershipError
from tests.integration.test_postgres_job_outbox import _new_store

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("RAGNOR_TEST_DATABASE_URL"),
        reason="requires real PostgreSQL CI service",
    ),
]

WORKER = Path(__file__).resolve().parent / "fixtures" / "outbox_claim_worker.py"


async def _start_worker(scope: str, worker: str, count: int, mode: str = "exit"):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(WORKER),
        scope,
        worker,
        str(count),
        mode,
        env=dict(os.environ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    ready = await asyncio.wait_for(process.stdout.readline(), timeout=15)
    assert ready == b"READY\n"
    return process


async def _go(process):
    assert process.stdin is not None
    process.stdin.write(b"GO\n")
    await process.stdin.drain()


@pytest.mark.parametrize("iteration", [0, 1])
async def test_two_independent_processes_claim_one_hundred_jobs_once(iteration: int) -> None:
    engine, sessions = await _new_store()
    scope = f"multi-{iteration}-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope)
    children = []
    try:
        for number in range(100):
            await store.enqueue({"scope": scope, "idempotency_key": str(number)})
        a = await _start_worker(scope, "host-A", 50)
        b = await _start_worker(scope, "host-B", 50)
        children.extend((a, b))
        await asyncio.gather(_go(a), _go(b))
        responses = await asyncio.gather(
            asyncio.wait_for(a.communicate(), timeout=20),
            asyncio.wait_for(b.communicate(), timeout=20),
        )
        batches = []
        for process, (stdout, stderr) in zip(children, responses, strict=True):
            assert process.returncode == 0, stderr.decode()
            batches.append(json.loads(stdout.decode()))
        claims = [job for batch in batches for job in batch]
        assert len(claims) == 100
        assert len({job["id"] for job in claims}) == 100
        assert all(job["epoch"] == 1 for job in claims)
        async with sessions() as session:
            result = await session.execute(
                text(
                    "SELECT count(*) FROM ragnor_embedding_jobs "
                    "WHERE scope=:scope AND status='processing'"
                ),
                {"scope": scope},
            )
            assert result.scalar_one() == 100
    finally:
        for process in children:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await engine.dispose()


async def test_killed_worker_claim_is_replayable_and_stale_ack_is_rejected() -> None:
    engine, sessions = await _new_store()
    scope = f"kill-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope)
    child = None
    try:
        await store.enqueue({"scope": scope, "idempotency_key": "job"})
        child = await _start_worker(scope, "dead-worker", 1, "hold")
        await _go(child)
        assert child.stdout is not None
        claim = json.loads((await asyncio.wait_for(child.stdout.readline(), timeout=15)).decode())[
            0
        ]
        child.kill()
        await asyncio.wait_for(child.wait(), timeout=15)
        assert child.returncode != 0

        async with sessions() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs SET lease_expires_at=now()-interval '1 second' "
                    "WHERE scope=:scope AND job_id=:id"
                ),
                {"scope": scope, "id": claim["id"]},
            )
        assert await store.reclaim_stale() == 1
        replayed = (await store.claim_and_mark_processing(1, "replacement"))[0]
        assert replayed.job_id == claim["id"]
        stale = JobPayload(
            job_id=claim["id"],
            claim_owner=claim["owner"],
            claim_generation=claim["epoch"],
        )
        with pytest.raises(LostJobOwnershipError):
            await store.mark_completed(stale)
        await store.mark_completed(replayed)
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        await engine.dispose()
