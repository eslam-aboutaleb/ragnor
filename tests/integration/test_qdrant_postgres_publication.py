"""Real PostgreSQL + real qdrant-client local engine publication integration.

Not a live Qdrant HTTP deployment: the same Qdrant client implementation in
local mode verifies genuine adapter API and payload filters, while PostgreSQL
provides real transactional manifest/lease fencing.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

qdrant = pytest.importorskip("qdrant_client")
from qdrant_client import AsyncQdrantClient  # noqa: E402

from ragnor.jobs.adapters.generation_vectors import ImmutableGenerationVectorAdapter  # noqa: E402
from ragnor.jobs.adapters.postgres import DDL, PostgresEmbeddingJobStore  # noqa: E402
from ragnor.jobs.outbox import LostJobOwnershipError  # noqa: E402
from ragnor.jobs.publication import ActiveGenerationReader  # noqa: E402
from ragnor.stores.qdrant import QdrantStore  # noqa: E402

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("RAGNOR_TEST_DATABASE_URL"), reason="real PostgreSQL is required"
    ),
]


async def test_crash_after_real_qdrant_write_before_db_activation_is_safe() -> None:
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = AsyncQdrantClient(location=":memory:")
    scope = f"tenant-{uuid4()}"
    job_store = PostgresEmbeddingJobStore(sessions, scope=scope, require_publication=True)
    backend = QdrantStore(
        table_name="postgres_qdrant_generation", embedding_dim=2, client=client
    )
    vectors = ImmutableGenerationVectorAdapter(store=backend, scope=scope)
    reader = ActiveGenerationReader(job_store, vectors)
    try:
        async with engine.begin() as connection:
            for statement in DDL:
                await connection.execute(text(statement))
        await job_store.enqueue({
            "scope": scope, "idempotency_key": "policy-rev1",
            "document_id": "policy", "revision": 1, "generation_key": "gen1",
        })
        first = (await job_store.claim_and_mark_processing(1, "worker-1"))[0]
        await vectors.stage("gen1", [
            {"id": "chunk", "text": "original revision", "embedding": [0.5, 0.5]}
        ])
        assert await reader.read(scope, "policy") == ()
        assert await job_store.activate_generation(first)
        assert await reader.read(scope, "policy") == ("original revision",)

        await job_store.enqueue({
            "scope": scope, "idempotency_key": "policy-rev2",
            "document_id": "policy", "revision": 2, "generation_key": "gen2",
        })
        crashed = (await job_store.claim_and_mark_processing(1, "worker-crashed"))[0]
        await vectors.stage("gen2", [
            {"id": "chunk", "text": "replacement revision", "embedding": [0.2, 0.8]}
        ])
        assert await reader.read(scope, "policy") == ("original revision",)
        # Simulate SIGKILL and lease timeout after Qdrant ACK but before manifest activation.
        async with sessions() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs "
                    "SET lease_expires_at = now() - interval '1 second' "
                    "WHERE job_id=:id"
                ),
                {"id": crashed.job_id},
            )
        assert await job_store.reclaim_stale() == 1
        replacement = (await job_store.claim_and_mark_processing(1, "worker-replay"))[0]
        await vectors.stage("gen2", [
            {"id": "chunk", "text": "replacement revision", "embedding": [0.2, 0.8]}
        ])
        assert await job_store.activate_generation(replacement)
        assert await reader.read(scope, "policy") == ("replacement revision",)
        with pytest.raises(LostJobOwnershipError):
            await job_store.activate_generation(crashed)
        assert await reader.read(scope, "policy") == ("replacement revision",)
    finally:
        await client.close()
        await engine.dispose()
