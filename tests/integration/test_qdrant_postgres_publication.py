"""Real PostgreSQL generation CAS with Qdrant HTTP in CI and local engine fallback.

RAGNOR_TEST_QDRANT_URL selects the actual service provisioned by CI. Local
runs without it qualify local-client behavior; the TCP fault case then skips.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

qdrant = pytest.importorskip("qdrant_client")
from qdrant_client import AsyncQdrantClient  # noqa: E402

from ragnor.jobs.adapters.generation_vectors import ImmutableGenerationVectorAdapter  # noqa: E402
from ragnor.jobs.adapters.generation_reservations import PostgresGenerationReservations  # noqa: E402
from ragnor.jobs.adapters.postgres import DDL, PostgresEmbeddingJobStore  # noqa: E402
from ragnor.jobs.outbox import LostJobOwnershipError, process_pending_jobs  # noqa: E402
from ragnor.jobs.publication import ActiveGenerationReader  # noqa: E402
from ragnor.stores.qdrant import QdrantStore  # noqa: E402

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("RAGNOR_TEST_DATABASE_URL"), reason="real PostgreSQL is required"
    ),
]


def _qdrant_client():
    url = os.environ.get("RAGNOR_TEST_QDRANT_URL")
    return AsyncQdrantClient(url=url, timeout=5) if url else AsyncQdrantClient(location=":memory:")


async def test_crash_after_real_qdrant_write_before_db_activation_is_safe() -> None:
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = _qdrant_client()
    scope = f"tenant-{uuid4()}"
    job_store = PostgresEmbeddingJobStore(sessions, scope=scope, require_publication=True)
    backend = QdrantStore(table_name="postgres_qdrant_generation", embedding_dim=2, client=client)
    vectors = ImmutableGenerationVectorAdapter(
        store=backend,
        scope=scope,
        reservations=PostgresGenerationReservations(sessions, scope=scope),
    )
    reader = ActiveGenerationReader(job_store, vectors)
    try:
        async with engine.begin() as connection:
            for statement in DDL:
                await connection.execute(text(statement))
        await job_store.enqueue(
            {
                "scope": scope,
                "idempotency_key": "policy-rev1",
                "document_id": "policy",
                "revision": 1,
                "generation_key": "gen1",
            }
        )
        first = (await job_store.claim_and_mark_processing(1, "worker-1"))[0]
        await vectors.stage(
            "gen1", [{"id": "chunk", "text": "original revision", "embedding": [0.5, 0.5]}]
        )
        assert await reader.read(scope, "policy") == ()
        assert await job_store.activate_generation(first)
        assert await reader.read(scope, "policy") == ("original revision",)

        await job_store.enqueue(
            {
                "scope": scope,
                "idempotency_key": "policy-rev2",
                "document_id": "policy",
                "revision": 2,
                "generation_key": "gen2",
            }
        )
        crashed = (await job_store.claim_and_mark_processing(1, "worker-crashed"))[0]
        await vectors.stage(
            "gen2", [{"id": "chunk", "text": "replacement revision", "embedding": [0.2, 0.8]}]
        )
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
        await vectors.stage(
            "gen2", [{"id": "chunk", "text": "replacement revision", "embedding": [0.2, 0.8]}]
        )
        assert await job_store.activate_generation(replacement)
        assert await reader.read(scope, "policy") == ("replacement revision",)
        with pytest.raises(LostJobOwnershipError):
            await job_store.activate_generation(crashed)
        assert await reader.read(scope, "policy") == ("replacement revision",)
    finally:
        await client.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_worker_orchestrator_stages_then_publishes_without_direct_ack() -> None:
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = _qdrant_client()
    scope = f"integrated-{uuid4()}"
    outbox = PostgresEmbeddingJobStore(sessions, scope=scope, require_publication=True)
    vectors = ImmutableGenerationVectorAdapter(
        store=QdrantStore(table_name="integrated_worker", embedding_dim=2, client=client),
        scope=scope,
        reservations=PostgresGenerationReservations(sessions, scope=scope),
    )

    class StagingProcessor:
        async def process(self, job):
            await vectors.stage(
                job.payload["generation_key"],
                [{"id": "chunk-1", "text": "active grounded policy", "embedding": [0.3, 0.7]}],
            )

    try:
        async with engine.begin() as connection:
            for statement in DDL:
                await connection.execute(text(statement))
        await outbox.enqueue(
            {
                "scope": scope,
                "idempotency_key": "revision",
                "document_id": "policy",
                "revision": 1,
                "generation_key": "gen-policy",
            }
        )
        reader = ActiveGenerationReader(outbox, vectors)
        assert await reader.read(scope, "policy") == ()
        assert (
            await process_pending_jobs(
                outbox,
                StagingProcessor(),
                worker_id="production-worker",
                require_atomic_claims=True,
            )
            == 1
        )
        assert await reader.read(scope, "policy") == ("active grounded policy",)
        assert await outbox.get_status_counts() == {"completed": 1}
    finally:
        await client.close()
        await engine.dispose()


async def test_concurrent_conflicting_generation_reservations_cannot_overwrite_vectors():
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    scope = "racing-" + uuid4().hex
    client = _qdrant_client()
    backend = QdrantStore(table_name="reservation_races", embedding_dim=2, client=client)
    try:
        async with engine.begin() as conn:
            for statement in DDL:
                await conn.execute(text(statement))
        # Prime the collection so the race is about generation CAS, not schema creation.
        await backend.get(ids=["not-present"])
        left = ImmutableGenerationVectorAdapter(
            store=backend,
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        right = ImmutableGenerationVectorAdapter(
            store=backend,
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        documents = [
            [{"id": "chunk", "text": value, "embedding": vector}]
            for value, vector in (("left", [1.0, 0.0]), ("right", [0.0, 1.0]))
        ]
        results = await asyncio.gather(
            left.stage("same-generation", documents[0]),
            right.stage("same-generation", documents[1]),
            return_exceptions=True,
        )
        assert sum(result is None for result in results) == 1
        assert sum(isinstance(result, ValueError) for result in results) == 1
        winner = 0 if results[0] is None else 1
        assert await left.read_generation("same-generation") == (documents[winner][0]["text"],)
        # A subset/superset is also a different complete generation, even when
        # it does not overwrite an existing per-document id.
        with pytest.raises(ValueError, match="immutable"):
            await right.stage(
                "same-generation",
                documents[winner] + [{"id": "another", "text": "extra", "embedding": [0.5, 0.5]}],
            )
        await right.stage("same-generation", documents[winner])
        assert await left.read_generation("same-generation") == (documents[winner][0]["text"],)
    finally:
        await client.close()
        await engine.dispose()


async def test_partial_vector_failure_keeps_the_durable_complete_generation_reservation():
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    scope = "partial-" + uuid4().hex
    client = _qdrant_client()

    class InterruptedStore(QdrantStore):
        async def upsert(self, documents, **kwargs):
            await super().upsert(documents[:1], **kwargs)
            raise ConnectionError("connection lost after one persisted point")

    backend = QdrantStore(table_name="partial_generations", embedding_dim=2, client=client)
    interrupted = InterruptedStore(table_name="partial_generations", embedding_dim=2, client=client)
    documents = [
        {"id": "a", "text": "first", "embedding": [1.0, 0.0]},
        {"id": "b", "text": "second", "embedding": [0.0, 1.0]},
    ]
    try:
        async with engine.begin() as conn:
            for statement in DDL:
                await conn.execute(text(statement))
        first = ImmutableGenerationVectorAdapter(
            store=interrupted,
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        with pytest.raises(ConnectionError):
            await first.stage("revision", documents)
        # A new reservation adapter/session simulates a restarted worker.
        replay = ImmutableGenerationVectorAdapter(
            store=backend,
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        with pytest.raises(ValueError, match="immutable"):
            await replay.stage("revision", documents[:1])
        await replay.stage("revision", list(reversed(documents)))
        assert await replay.read_generation("revision") == ("first", "second")
    finally:
        await client.close()
        await engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("RAGNOR_TEST_QDRANT_URL"), reason="real Qdrant HTTP service required"
)
async def test_tcp_disconnect_and_real_server_reconnect_preserve_generation_identity():
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    scope = "network-" + uuid4().hex

    async def disconnect(reader, writer):
        writer.transport.abort()

    broken_endpoint = await asyncio.start_server(disconnect, "127.0.0.1", 0)
    port = broken_endpoint.sockets[0].getsockname()[1]
    failed_client = AsyncQdrantClient(
        url=f"http://127.0.0.1:{port}", timeout=1, check_compatibility=False
    )
    live_client = _qdrant_client()
    documents = [{"id": "a", "text": "persist after reconnect", "embedding": [1.0, 0.0]}]
    try:
        async with engine.begin() as conn:
            for statement in DDL:
                await conn.execute(text(statement))
        unreachable = ImmutableGenerationVectorAdapter(
            store=QdrantStore(table_name="network_recovery", embedding_dim=2, client=failed_client),
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        from qdrant_client.http.exceptions import ResponseHandlingException

        with pytest.raises(ResponseHandlingException):
            await unreachable.stage("retry", documents)
        recovered = ImmutableGenerationVectorAdapter(
            store=QdrantStore(table_name="network_recovery", embedding_dim=2, client=live_client),
            scope=scope,
            reservations=PostgresGenerationReservations(sessions, scope=scope),
        )
        with pytest.raises(ValueError, match="immutable"):
            await recovered.stage(
                "retry", [{"id": "a", "text": "changed", "embedding": [1.0, 0.0]}]
            )
        await recovered.stage("retry", documents)
        assert await recovered.read_generation("retry") == ("persist after reconnect",)
    finally:
        broken_endpoint.close()
        await broken_endpoint.wait_closed()
        await failed_client.close()
        await live_client.close()
        await engine.dispose()
