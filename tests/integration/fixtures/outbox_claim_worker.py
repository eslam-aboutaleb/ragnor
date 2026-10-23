"""Subprocess worker used by real PostgreSQL multi-process CI tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.postgres import PostgresEmbeddingJobStore


async def main() -> None:
    scope, worker_id, batch, mode = sys.argv[1:5]
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    print("READY", flush=True)
    await asyncio.to_thread(sys.stdin.readline)
    store = PostgresEmbeddingJobStore(sessions, scope=scope)
    jobs = await store.claim_and_mark_processing(int(batch), worker_id)
    payload = [
        {"id": job.job_id, "owner": job.claim_owner, "epoch": job.claim_generation} for job in jobs
    ]
    print(json.dumps(payload), flush=True)
    if mode == "hold":
        await asyncio.Event().wait()
    else:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
