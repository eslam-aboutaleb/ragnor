"""Repeatable real SQL outbox contention benchmark (no asserted SLA).

CI records observed duration/throughput for 1, 2, and 4 workers on
PostgreSQL and file-backed SQLite. Each worker uses host-owned sessions.
No user data, credentials or payload content are printed.
"""

from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from uuid import uuid4

from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.relational import RelationalJobStoreFactory, metadata


async def main() -> None:
    url = os.environ["RAGNOR_CONTRACT_DATABASE_URL"]
    engine = create_async_engine(url)
    if url.startswith("sqlite+"):

        @event.listens_for(engine.sync_engine, "connect")
        def disable_driver_transaction_control(dbapi_connection, _connection_record) -> None:
            dbapi_connection.isolation_level = None

        @event.listens_for(engine.sync_engine, "begin")
        def begin_transaction(connection) -> None:
            connection.exec_driver_sql("BEGIN")

    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        results = []
        for workers in (1, 2, 4):
            scope = f"benchmark-{uuid4()}"
            store = RelationalJobStoreFactory(sessions).job_store(scope=scope)
            for number in range(80):
                await store.enqueue({"scope": scope, "idempotency_key": f"chunk-{number}"})

            async def worker(
                worker_num: int,
                started: asyncio.Event,
                active_store,
            ) -> list[str]:
                await started.wait()
                owned: list[str] = []
                while True:
                    jobs = await active_store.claim_and_mark_processing(5, f"worker-{worker_num}")
                    if not jobs:
                        return owned
                    for job in jobs:
                        await active_store.mark_completed(job)
                        owned.append(job.job_id)

            started = asyncio.Event()
            tasks = [asyncio.create_task(worker(num, started, store)) for num in range(workers)]
            start = perf_counter()
            started.set()
            batches = await asyncio.gather(*tasks)
            seconds = perf_counter() - start
            all_ids = [identifier for batch in batches for identifier in batch]
            assert len(all_ids) == 80
            assert len(set(all_ids)) == 80
            assert await store.get_status_counts() == {"completed": 80}
            results.append(
                {
                    "workers": workers,
                    "total_jobs": 80,
                    "claim_batch_size": 5,
                    "elapsed_seconds": round(seconds, 3),
                    "jobs_per_second": round(80 / max(seconds, 1e-9), 2),
                    "worker_job_counts": [len(batch) for batch in batches],
                }
            )
        print(
            json.dumps(
                {"dialect": engine.dialect.name, "results": results}, sort_keys=True, indent=2
            )
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
