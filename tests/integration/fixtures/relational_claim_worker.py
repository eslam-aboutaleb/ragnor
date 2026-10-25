"""Independent process using only a host-owned SQLAlchemy engine and sessions."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.relational import RelationalJobStoreFactory


async def main() -> None:
    scope, worker_id, batch, mode = sys.argv[1:5]
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
    print("READY", flush=True)
    await asyncio.to_thread(sys.stdin.readline)
    store = RelationalJobStoreFactory(sessions).job_store(scope=scope)
    claimed = await store.claim_and_mark_processing(int(batch), worker_id)
    print(
        json.dumps([{"id": job.job_id, "epoch": job.claim_generation} for job in claimed]),
        flush=True,
    )
    if mode == "hold":
        await asyncio.Event().wait()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
