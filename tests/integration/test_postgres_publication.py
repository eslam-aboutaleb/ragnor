"""PostgreSQL manifest publication: staging is invisible until fenced activation."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.jobs.adapters.postgres import DDL, PostgresEmbeddingJobStore
from ragnor.jobs.outbox import LostJobOwnershipError
from ragnor.jobs.publication import ActiveGenerationReader

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="real PostgreSQL required"),
]


class ImmutableVectorBackend:
    def __init__(self) -> None:
        self.generations: dict[str, tuple[str, ...]] = {}

    async def stage(self, generation_key: str, vectors: tuple[str, ...]) -> None:
        if generation_key in self.generations and self.generations[generation_key] != vectors:
            raise ValueError("immutable vector generation cannot be overwritten")
        self.generations[generation_key] = vectors

    async def read_generation(self, generation_key: str) -> tuple[str, ...]:
        return self.generations[generation_key]


async def _configured():
    assert DB_URL
    engine = create_async_engine(DB_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        for sql in DDL:
            await conn.execute(text(sql))
    return engine, sessions


async def test_uncommitted_vectors_are_invisible_and_crash_replay_activates_once() -> None:
    engine, sessions = await _configured()
    scope = f"tenant-{uuid4()}"
    writer = PostgresEmbeddingJobStore(sessions, scope=scope)
    vectors = ImmutableVectorBackend()
    reader = ActiveGenerationReader(writer, vectors)
    document = "policy-a"
    try:
        await writer.enqueue(
            {
                "scope": scope,
                "idempotency_key": "revision-one",
                "document_id": document,
                "revision": 1,
                "generation_key": f"{scope}/policy-a/1",
            }
        )
        first = (await writer.claim_and_mark_processing(1, "host-one"))[0]
        await vectors.stage(first.payload["generation_key"], ("old-policy",))
        assert await reader.read(scope, document) == ()
        assert await writer.activate_generation(first) is True
        assert await reader.read(scope, document) == ("old-policy",)

        await writer.enqueue(
            {
                "scope": scope,
                "idempotency_key": "revision-two",
                "document_id": document,
                "revision": 2,
                "generation_key": f"{scope}/policy-a/2",
            }
        )
        dying = (await writer.claim_and_mark_processing(1, "host-dying"))[0]
        await vectors.stage(dying.payload["generation_key"], ("new-policy",))
        # External upsert completes, but the worker dies before activation.
        assert await reader.read(scope, document) == ("old-policy",)

        async with sessions() as session, session.begin():
            await session.execute(
                text(
                    "UPDATE ragnor_embedding_jobs SET lease_expires_at=now()-interval '1 second' WHERE job_id=:id"
                ),
                {"id": dying.job_id},
            )
        assert await writer.reclaim_stale() == 1
        replay = (await writer.claim_and_mark_processing(1, "host-replay"))[0]
        await vectors.stage(replay.payload["generation_key"], ("new-policy",))
        assert await writer.activate_generation(replay) is True
        assert await reader.read(scope, document) == ("new-policy",)
        with pytest.raises(LostJobOwnershipError):
            await writer.activate_generation(dying)
        assert await reader.read(scope, document) == ("new-policy",)
    finally:
        await engine.dispose()


async def test_old_revision_cannot_roll_back_new_active_version() -> None:
    engine, sessions = await _configured()
    scope = f"tenant-{uuid4()}"
    writer = PostgresEmbeddingJobStore(sessions, scope=scope)
    vectors = ImmutableVectorBackend()
    reader = ActiveGenerationReader(writer, vectors)
    document = "shared-id"
    try:
        for rev in (1, 2):
            await writer.enqueue(
                {
                    "scope": scope,
                    "idempotency_key": str(rev),
                    "document_id": document,
                    "revision": rev,
                    "generation_key": f"{scope}/{document}/{rev}",
                }
            )
        old, new = await writer.claim_and_mark_processing(2, "host")
        await vectors.stage(old.payload["generation_key"], ("old",))
        await vectors.stage(new.payload["generation_key"], ("new",))
        assert await writer.activate_generation(new) is True
        assert await writer.activate_generation(old) is False
        assert await reader.read(scope, document) == ("new",)
    finally:
        await engine.dispose()


async def test_foreign_scope_and_bad_revision_rejected_before_publication() -> None:
    engine, sessions = await _configured()
    a = PostgresEmbeddingJobStore(sessions, scope=f"left-{uuid4()}")
    b = PostgresEmbeddingJobStore(sessions, scope=f"right-{uuid4()}")
    try:
        await a.enqueue(
            {
                "scope": a.scope,
                "idempotency_key": "a",
                "document_id": "d",
                "revision": 1,
                "generation_key": "gen-a",
            }
        )
        job = (await a.claim_and_mark_processing(1, "host"))[0]
        with pytest.raises(LostJobOwnershipError):
            await b.activate_generation(job)
        assert await b.get_active_generation("d") is None
        assert await a.get_active_generation("d") is None
        await a.activate_generation(job)
        assert await a.get_active_generation("d") == "gen-a"
    finally:
        await engine.dispose()


async def test_invalid_manifest_coordinates_fail_closed() -> None:
    engine, sessions = await _configured()
    s = PostgresEmbeddingJobStore(sessions, scope=f"tenant-{uuid4()}")
    try:
        for payload in [
            {
                "scope": s.scope,
                "idempotency_key": "x",
                "document_id": "",
                "revision": 1,
                "generation_key": "a",
            },
            {
                "scope": s.scope,
                "idempotency_key": "y",
                "document_id": "a",
                "revision": -1,
                "generation_key": "b",
            },
        ]:
            await s.enqueue(payload)
        jobs = await s.claim_and_mark_processing(2, "host")
        for job in jobs:
            with pytest.raises(ValueError, match="document_id|revision"):
                await s.activate_generation(job)
    finally:
        await engine.dispose()

async def test_strict_publication_mode_disallows_direct_ack_bypass() -> None:
    engine, sessions = await _configured()
    scope = f"strict-{uuid4()}"
    publisher = PostgresEmbeddingJobStore(sessions, scope=scope, require_publication=True)
    try:
        await publisher.enqueue(
            {"scope": scope, "idempotency_key": "first", "document_id": "d",
             "revision": 1, "generation_key": "immutable-key"}
        )
        job = (await publisher.claim_and_mark_processing(1, "host"))[0]
        with pytest.raises(RuntimeError, match="activate_generation"):
            await publisher.mark_completed(job)
        assert await publisher.get_active_generation("d") is None
        assert await publisher.activate_generation(job) is True
        assert await publisher.get_active_generation("d") == "immutable-key"
    finally:
        await engine.dispose()
