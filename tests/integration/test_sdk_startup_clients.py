"""Startup and cleanup against real PostgreSQL/pgvector and Qdrant HTTP clients.

The embedder is a deterministic unit fixture; these tests qualify database
and vector-client lifecycle, not a deployed embedding model or its credentials.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor import Document, RagnorBuilder
from ragnor.db.sql_factory import SQLAlchemyFactory
from ragnor.exceptions import BackendUnavailableError, ConfigurationError, OperationTimeoutError
from ragnor.stores.pgvector import PgVectorStore
from tests.integration.test_postgres_sdk_host import DeterministicEmbedder, SQLVersionStore

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="real PostgreSQL required"),
]


class EngineResource:
    def __init__(self, engine):
        self.engine = engine
        self.closed = 0

    async def aclose(self):
        await self.engine.dispose()
        self.closed += 1


@pytest.fixture
async def database():
    admin = create_async_engine(DB_URL)
    schema = "ragnor_startup_" + uuid4().hex
    role = schema + "_reader"
    try:
        async with admin.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(text(f"CREATE SCHEMA {schema}"))  # noqa: S608
            await connection.execute(text(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS"))  # noqa: S608
            await connection.execute(text(f"SET LOCAL search_path TO {schema}, public"))  # noqa: S608
            await connection.execute(
                text("""CREATE TABLE chunks (
                    id text PRIMARY KEY, text text NOT NULL, embedding vector(2),
                    source text, external_id text, version_id text, source_id text,
                    scope text NOT NULL DEFAULT current_setting('ragnor.scope'),
                    tsvector tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', text)) STORED
                )""")
            )
            await connection.execute(
                text("""CREATE TABLE ragnor_sdk_host_versions (
                    scope text NOT NULL, version_id text PRIMARY KEY,
                    snapshot jsonb NOT NULL, active boolean NOT NULL DEFAULT true
                )""")
            )
            for table in ("chunks", "ragnor_sdk_host_versions"):
                await connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))  # noqa: S608
                await connection.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))  # noqa: S608
                await connection.execute(
                    text(  # noqa: S608 - fixed table and generated role
                        f"CREATE POLICY tenant_scope ON {table} FOR ALL TO {role} "
                        "USING (scope=current_setting('ragnor.scope')) "
                        "WITH CHECK (scope=current_setting('ragnor.scope'))"
                    )
                )
                await connection.execute(
                    text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {role}")
                )  # noqa: S608
            await connection.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {role}"))  # noqa: S608
        yield admin, schema, role
    finally:
        async with admin.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))  # noqa: S608
            await connection.execute(text(f"DROP ROLE IF EXISTS {role}"))  # noqa: S608
        await admin.dispose()


def production_builder(engine, schema, role, scope):
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def scoped_sessions():
        async with sessions() as session:
            await session.execute(text(f"SET LOCAL ROLE {role}"))  # noqa: S608
            await session.execute(text(f"SET LOCAL search_path TO {schema}, public"))  # noqa: S608
            await session.execute(
                text("SELECT set_config('ragnor.scope', :scope, true)"), {"scope": scope}
            )
            yield session

    store = PgVectorStore(table_name="chunks", embedding_dim=2, session_factory=scoped_sessions)

    class Versions(SQLVersionStore):
        async def delete_chunks(self, session, version_id):
            await store.delete(session=session, filters={"version_id": version_id})

    builder = RagnorBuilder.production(
        DeterministicEmbedder(),
        embedding_model="startup-client-fixture",
        embedding_dim=2,
        store=store,
        version_store=Versions(scope),
        session_factory=scoped_sessions,
        sql_factory=SQLAlchemyFactory(),
    )
    return builder, store, scoped_sessions


async def test_real_pgvector_startup_and_sdk_workflow_use_restricted_tenant_sessions(database):
    _, schema, role = database
    engine = create_async_engine(DB_URL)
    first, store, sessions = production_builder(engine, schema, role, "left")

    async def ready():
        async with sessions() as session:
            flags = await session.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user")
            )
            assert flags.one() == (False, False)
        assert await store.count(strict=True) == 0

    try:
        async with await first.with_startup_check(ready).build() as left:
            other, _, _ = production_builder(engine, schema, role, "right")
            async with await other.build() as right:
                await left.ingest([Document(id="policy", text="left policy")], lock_key="left")
                await right.ingest([Document(id="policy", text="right policy")], lock_key="right")
                assert left.store is store
                assert await left.count() == await right.count() == 1
                assert await left.count(scope="right") == 0
                hits = await left.search("left policy")
                assert [hit.document for hit in hits] == ["left policy"]
                # RLS also protects low-level paths with no application WHERE.
                assert [r["document"] for r in await store.get()] == ["left policy"]
        # Borrowed pool survives SDK shutdown; scope is re-applied per transaction.
        assert await store.count(strict=True) == 1
        assert engine.pool.checkedout() == 0
    finally:
        await engine.dispose()


@pytest.mark.parametrize("failure", ["timeout", "cancel", "disconnect"])
async def test_real_postgres_startup_failure_releases_connections_and_owned_pool(database, failure):
    admin, schema, role = database
    engine = create_async_engine(DB_URL)
    resource = EngineResource(engine)
    builder, _, sessions = production_builder(engine, schema, role, "tenant")
    entered = asyncio.Event()

    async def probe():
        async with sessions() as session:
            pid = await session.scalar(text("SELECT pg_backend_pid()"))
            if failure == "disconnect":
                async with admin.connect() as killer:
                    assert await killer.scalar(
                        text("SELECT pg_terminate_backend(:pid)"), {"pid": pid}
                    )
                await session.execute(text("SELECT 1"))
            else:
                entered.set()
                await session.execute(text("SELECT pg_sleep(30)"))

    builder.with_owned_resource(resource).with_startup_check(
        probe, timeout_seconds=0.3 if failure == "timeout" else 5
    )
    task = asyncio.create_task(builder.build())
    if failure == "cancel":
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
    expected = {
        "timeout": OperationTimeoutError,
        "cancel": asyncio.CancelledError,
        "disconnect": BackendUnavailableError,
    }[failure]
    with pytest.raises(expected):
        await asyncio.wait_for(task, 8)
    assert resource.closed == 1
    assert engine.pool.checkedout() == 0
    # New clients can recover; failed readiness never wrote index/version data.
    recovery = create_async_engine(DB_URL)
    try:
        retry, store, _ = production_builder(recovery, schema, role, "tenant")

        async def ready():
            assert await store.count(strict=True) == 0

        async with await retry.with_startup_check(ready).build() as rag:
            assert await rag.count() == 0
    finally:
        await recovery.dispose()


@pytest.mark.skipif(
    not os.environ.get("RAGNOR_TEST_QDRANT_URL"), reason="real Qdrant HTTP required"
)
async def test_real_qdrant_readiness_failure_preserves_borrowed_client_and_allows_retry(database):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    _, schema, role = database
    engine = create_async_engine(DB_URL)
    client = AsyncQdrantClient(url=os.environ["RAGNOR_TEST_QDRANT_URL"], timeout=2)
    collection = "startup_" + uuid4().hex
    builder, _, _ = production_builder(engine, schema, role, "tenant")

    async def ready():
        info = await client.get_collection(collection)
        if info.config.params.vectors.size != 2:
            raise ConfigurationError("vector collection has the wrong dimension")

    try:
        with pytest.raises(BackendUnavailableError):
            await builder.with_startup_check(ready).build()
        # Host-owned real network client is still usable after failed startup.
        await client.create_collection(
            collection, vectors_config=VectorParams(size=3, distance=Distance.EUCLID)
        )
        with pytest.raises(ConfigurationError, match="dimension"):
            await builder.build()
        await client.delete_collection(collection)
        await client.create_collection(
            collection, vectors_config=VectorParams(size=2, distance=Distance.EUCLID)
        )
        async with await builder.build() as rag:
            assert await rag.count() == 0
        assert (await client.get_collection(collection)).points_count == 0
    finally:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
        await client.close()
        await engine.dispose()


@pytest.mark.parametrize("failure", ["timeout", "cancel", "disconnect"])
async def test_qdrant_http_client_startup_fault_closes_owned_transport(database, failure):
    """Real Qdrant client sockets against controlled stalled/reset TCP peers.

    This qualifies transport failure/cleanup, not Qdrant service semantics;
    the separate HTTP-service test checks actual collections and readiness.
    """
    from qdrant_client import AsyncQdrantClient

    _, schema, role = database
    entered, disconnected = asyncio.Event(), asyncio.Event()

    async def endpoint(reader, writer):
        entered.set()
        try:
            if failure == "disconnect":
                writer.transport.abort()
            else:
                await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()

    server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = AsyncQdrantClient(url=f"http://127.0.0.1:{port}", timeout=5, check_compatibility=False)
    engine = create_async_engine(DB_URL)
    closed = []

    class ClientResource:
        async def aclose(self):
            await client.close()
            closed.append("qdrant")

    async def ready():
        await client.get_collections()

    builder, _, _ = production_builder(engine, schema, role, "tenant")
    builder.with_owned_resource(ClientResource()).with_startup_check(
        ready, timeout_seconds=0.3 if failure == "timeout" else 5
    )
    try:
        task = asyncio.create_task(builder.build())
        if failure == "cancel":
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
        expected = {
            "timeout": OperationTimeoutError,
            "cancel": asyncio.CancelledError,
            "disconnect": BackendUnavailableError,
        }[failure]
        with pytest.raises(expected):
            await asyncio.wait_for(task, 8)
        assert closed == ["qdrant"]
        await asyncio.wait_for(disconnected.wait(), 3)
        assert engine.pool.checkedout() == 0
    finally:
        server.close()
        await server.wait_closed()
        await client.close()
        await engine.dispose()
