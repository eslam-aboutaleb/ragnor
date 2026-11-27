"""Regression tests for store bugs found by the LlamaIndex backend tests.

Real PostgreSQL/pgvector and real Qdrant; skipped when the service URLs are unset.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
QDRANT_URL = os.environ.get("RAGNOR_TEST_QDRANT_URL")


@pytest.mark.skipif(not DB_URL, reason="real PostgreSQL required")
async def test_pgvector_upsert_with_extra_fields_and_metadata_keeps_columns_aligned() -> None:
    """extra_fields (e.g. tenant_id) and metadata must land in their own columns.

    Before the fix the INSERT column list was (extra, metadata) but the VALUES
    list was (metadata, extra), so ``tenant_id`` silently received a metadata value.
    """
    from sqlalchemy import text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from ragnor.stores.pgvector import PgVectorStore  # noqa: PLC0415

    schema = "reg_" + uuid.uuid4().hex
    admin = create_async_engine(DB_URL)
    async with admin.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text(f"CREATE SCHEMA {schema}"))  # noqa: S608
        await conn.execute(text(f"SET LOCAL search_path TO {schema}, public"))  # noqa: S608
        await conn.execute(
            text(
                "CREATE TABLE chunks (id text PRIMARY KEY, text text NOT NULL, "
                "embedding vector(2), tenant_id text NOT NULL, kind text, source text, "
                "tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)"
            )
        )
    engine = create_async_engine(
        DB_URL, connect_args={"server_settings": {"search_path": f"{schema},public"}}
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = PgVectorStore(table_name="chunks", embedding_dim=2, session_factory=sessions)
    try:
        await store.upsert(
            [
                {
                    "id": "x",
                    "text": "hello",
                    "metadata": {"kind": "note", "source": "doc"},
                    "embedding": [1.0, 0.0],
                }
            ],
            extra_fields={"tenant_id": "alice"},
        )
        async with sessions() as s:
            row = (await s.execute(text("SELECT tenant_id, kind, source FROM chunks"))).one()
        assert tuple(row) == ("alice", "note", "doc")
        assert await store.count(tenant_id="alice") == 1
        assert await store.count(tenant_id="note") == 0
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))  # noqa: S608
        await admin.dispose()


@pytest.mark.skipif(not QDRANT_URL, reason="real Qdrant required")
async def test_qdrant_concurrent_first_use_creates_the_collection_once() -> None:
    """Concurrent first writes used to race on create_collection (409 Conflict)."""
    from qdrant_client import AsyncQdrantClient  # noqa: PLC0415

    from ragnor.stores.qdrant import QdrantStore  # noqa: PLC0415

    client = AsyncQdrantClient(url=QDRANT_URL, check_compatibility=False)
    name = "reg_" + uuid.uuid4().hex
    store = QdrantStore(table_name=name, embedding_dim=2, client=client)
    try:
        await asyncio.gather(
            *(
                store.upsert(
                    [{"id": f"d{i}", "text": f"doc {i}", "metadata": {}, "embedding": [1.0, 0.0]}]
                )
                for i in range(8)
            )
        )
        assert await store.count() == 8
    finally:
        await client.delete_collection(name)
        await client.close()
