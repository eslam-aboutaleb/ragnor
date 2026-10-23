"""Real PostgreSQL durable retry-classification and sanitized event accounting."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from ragnor.jobs.adapters.postgres import PostgresEmbeddingJobStore
from ragnor.jobs.outbox import process_pending_jobs
from tests.integration.test_postgres_job_outbox import _new_store

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("RAGNOR_TEST_DATABASE_URL"), reason="requires PostgreSQL"
    ),
]


class Permanent:
    def delay(self, error: Exception, attempt: int, max_attempts: int):
        return None


class PoisonedProcessor:
    async def process(self, job):
        raise ValueError("SECRET raw customer data must not reach status")


class Events:
    def __init__(self) -> None:
        self.observed = []

    async def emit(self, event) -> None:
        self.observed.append(event)


async def test_durable_permanent_job_is_dead_lettered_once_and_redacted() -> None:
    engine, sessions = await _new_store()
    scope = f"policy-{uuid4()}"
    store = PostgresEmbeddingJobStore(sessions, scope=scope)
    observer = Events()
    try:
        await store.enqueue({"scope": scope, "idempotency_key": "bad"})
        assert (
            await process_pending_jobs(
                store,
                PoisonedProcessor(),
                worker_id="worker",
                require_atomic_claims=True,
                retry_policy=Permanent(),
                event_sink=observer,
            )
            == 0
        )
        async with sessions() as s:
            result = await s.execute(
                text(
                    "SELECT status,status_detail,retry_count FROM ragnor_embedding_jobs "
                    "WHERE scope=:scope"
                ),
                {"scope": scope},
            )
            row = result.one()
            assert row.status == "dead_letter"
            assert row.retry_count == 1
            assert "Permanent failure" in row.status_detail
            assert "SECRET" not in row.status_detail
        assert [x.kind for x in observer.observed] == ["claimed", "dead_letter"]
        assert all("SECRET" not in repr(x) for x in observer.observed)
        assert await store.claim_and_mark_processing(1, "replayer") == []
    finally:
        await engine.dispose()
