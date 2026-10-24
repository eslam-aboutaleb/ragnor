"""Real PostgreSQL RLS and synchronous-facade connection cancellation contracts."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.exceptions import OperationTimeoutError
from ragnor.sync_client import SyncRagnor
from ragnor.types import Document
from tests.integration.test_postgres_sdk_host import TABLE_DDL, _client

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="real PostgreSQL required")


@pytest.mark.asyncio
async def test_injected_production_sessions_enforce_rls_for_unfiltered_and_spoofed_sql():
    engine = create_async_engine(DB_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    identifier = "ragnor_rls_" + uuid4().hex
    role = identifier + "_reader"
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f"CREATE SCHEMA {identifier}"))  # noqa: S608
            await connection.execute(text(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS"))  # noqa: S608
            await connection.execute(text(f"GRANT USAGE ON SCHEMA {identifier} TO {role}"))  # noqa: S608
            await connection.execute(text(f"SET LOCAL search_path TO {identifier}, public"))  # noqa: S608
            for ddl in TABLE_DDL:
                await connection.execute(text(ddl))
            for table in ("ragnor_sdk_host_versions", "ragnor_sdk_host_vectors"):
                await connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))  # noqa: S608
                await connection.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))  # noqa: S608
                await connection.execute(
                    text(  # noqa: S608 - constant table and generated role
                        f"CREATE POLICY tenant_scope ON {table} FOR ALL TO {role} "
                        "USING (scope=current_setting('ragnor.scope')) "
                        "WITH CHECK (scope=current_setting('ragnor.scope'))"
                    )
                )
                await connection.execute(
                    text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {role}")
                )  # noqa: S608

        def scoped_sessions(scope):
            @asynccontextmanager
            async def configured():
                async with sessions() as session:
                    await session.execute(text(f"SET LOCAL ROLE {role}"))  # noqa: S608
                    await session.execute(text(f"SET LOCAL search_path TO {identifier}, public"))  # noqa: S608
                    await session.execute(
                        text("SELECT set_config('ragnor.scope', :scope, true)"), {"scope": scope}
                    )
                    yield session

            return configured

        left, _, _ = await _client("left", scoped_sessions("left"))
        right, _, _ = await _client("right", scoped_sessions("right"))
        try:
            await left.ingest([Document(id="same", text="left policy")])
            await right.ingest([Document(id="same", text="right policy")])
            assert await left.count() == await right.count() == 1
            async with scoped_sessions("left")() as session:
                # No application WHERE clause: PostgreSQL itself hides right's rows.
                rows = await session.execute(
                    text("SELECT scope, content FROM ragnor_sdk_host_vectors")
                )
                assert rows.all() == [("left", "left policy")]
            async with scoped_sessions("left")() as session:
                with pytest.raises(DBAPIError):
                    await session.execute(
                        text(
                            "INSERT INTO ragnor_sdk_host_vectors(scope,doc_id,version_id,content) "
                            "VALUES ('right','spoof','spoof','forbidden')"
                        )
                    )
        finally:
            await left.aclose()
            await right.aclose()
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA IF EXISTS {identifier} CASCADE"))  # noqa: S608
            await connection.execute(text(f"DROP ROLE IF EXISTS {role}"))  # noqa: S608
        await engine.dispose()


@pytest.mark.parametrize("_repeat", range(2))
def test_sync_timeout_cancels_real_postgres_query_and_closes_loop_thread_and_pool(_repeat):
    reached_database = threading.Event()
    cancelled = threading.Event()
    state = {}

    class SQLClient:
        async def ingest(self, *args, **kwargs):
            raise NotImplementedError

        async def search(self, *args, **kwargs):
            raise NotImplementedError

        async def count(self, **kwargs):
            async with state["engine"].connect() as connection:
                await connection.execute(text("SELECT pg_backend_pid()"))
                reached_database.set()
                try:
                    await connection.execute(text("SELECT pg_sleep(30)"))
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return 0

        async def aclose(self):
            await state["engine"].dispose()
            state["disposed"] = True

    async def construct():
        state["engine"] = create_async_engine(DB_URL)
        return SQLClient()

    with SyncRagnor.from_async_factory(construct, timeout=5) as client:
        with pytest.raises(OperationTimeoutError):
            client.count(timeout=0.5)
    assert reached_database.is_set()
    assert cancelled.is_set()
    assert state["disposed"]
    assert state["engine"].pool.checkedout() == 0
    assert not client._thread.is_alive()
    assert client._loop.is_closed()
