"""Production host adapters: real PostgreSQL transactions, tenant scope and rollback.

Unlike pure mocks, the session provider and version/vector adapters execute
against the GitHub Actions PostgreSQL service. Custom adapters prove the
core builder does not require a specific host app or vector SDK.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.builder import RagnorBuilder
from ragnor.db.sql_factory import SQLAlchemyFactory
from ragnor.db.schema import ingestion_locks
from ragnor.ingestion.pipeline import VersionRecord
from ragnor.ingestion.versioning import IngestionSnapshot
from ragnor.retrieval.capabilities import BackendFeature, FilterOperator, StoreCapabilities
from ragnor.types import Document

DB_URL = os.environ.get("RAGNOR_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DB_URL, reason="requires real PostgreSQL service"),
]

TABLE_DDL = (
    """CREATE TABLE IF NOT EXISTS ragnor_sdk_host_versions (
        scope text NOT NULL, version_id text PRIMARY KEY, snapshot jsonb NOT NULL,
        active boolean NOT NULL DEFAULT true
    )""",
    """CREATE TABLE IF NOT EXISTS ragnor_sdk_host_vectors (
        scope text NOT NULL, doc_id text NOT NULL, version_id text NOT NULL,
        content text NOT NULL, PRIMARY KEY(scope, doc_id)
    )""",
)


class DeterministicEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]

    async def embed_query(self, text_value: str) -> list[float]:
        return [float(len(text_value)), 1.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class SQLVectorStore:
    capabilities = StoreCapabilities(
        features=frozenset({BackendFeature.ISOLATION_SCOPE}),
        filter_operators=frozenset({FilterOperator.EQ}),
    )
    embedding_dim = 2

    def __init__(self, scope: str, sessions) -> None:
        self.scope = scope
        self.sessions = sessions
        self.fail_after_insert = False

    async def upsert(self, documents, session=None) -> None:
        if session is None:
            raise RuntimeError("production requires caller-owned ingestion transaction")
        for doc in documents:
            await session.execute(
                text(
                    "INSERT INTO ragnor_sdk_host_vectors "
                    "(scope, doc_id, version_id, content) "
                    "VALUES (:scope,:doc,:version,:content)"
                ),
                {
                    "scope": self.scope,
                    "doc": doc["id"],
                    "version": doc["metadata"]["version_id"],
                    "content": doc["text"],
                },
            )
        if self.fail_after_insert:
            raise RuntimeError("upsert failed after SQL write")

    async def count(self, session=None, **filters) -> int:
        async def _query(s):
            result = await s.execute(
                text("SELECT count(*) FROM ragnor_sdk_host_vectors WHERE scope=:scope"),
                {"scope": self.scope},
            )
            return int(result.scalar_one())

        if session is not None:
            return await _query(session)
        async with self.sessions() as owned:
            return await _query(owned)

    async def hybrid_search(self, *args, **kwargs):
        return []


class SQLVersionStore:
    def __init__(self, scope: str) -> None:
        self.scope = scope

    async def find_active(self, session) -> VersionRecord | None:
        row = await session.execute(
            text(
                "SELECT version_id, snapshot FROM ragnor_sdk_host_versions "
                "WHERE scope=:scope AND active=true"
            ),
            {"scope": self.scope},
        )
        rec = row.mappings().one_or_none()
        if rec is None:
            return None
        return VersionRecord(
            version_id=rec["version_id"],
            snapshot=IngestionSnapshot(**rec["snapshot"]),
            source_id=self.scope,
        )

    async def close_active(self, session) -> None:
        await session.execute(
            text("UPDATE ragnor_sdk_host_versions SET active=false WHERE scope=:scope"),
            {"scope": self.scope},
        )

    async def create_version(self, session, snapshot: IngestionSnapshot) -> VersionRecord:
        version_id = str(uuid4())
        await session.execute(
            text(
                "INSERT INTO ragnor_sdk_host_versions (scope,version_id,snapshot) "
                "VALUES (:scope,:version,CAST(:snapshot AS jsonb))"
            ),
            {
                "scope": self.scope,
                "version": version_id,
                "snapshot": json.dumps(asdict(snapshot)),
            },
        )
        return VersionRecord(version_id=version_id, snapshot=snapshot, source_id=self.scope)

    async def delete_chunks(self, session, version_id: str) -> None:
        await session.execute(
            text("DELETE FROM ragnor_sdk_host_vectors WHERE scope=:scope AND version_id=:version"),
            {"scope": self.scope, "version": version_id},
        )


async def _fixture():
    assert DB_URL is not None
    engine = create_async_engine(DB_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as session:
        await session.run_sync(ingestion_locks.create, checkfirst=True)
        for schema in TABLE_DDL:
            await session.execute(text(schema))
    return engine, sessions


async def _client(scope: str, sessions):
    store = SQLVectorStore(scope, sessions)
    versions = SQLVersionStore(scope)
    rag = await RagnorBuilder(
        embedder=DeterministicEmbedder(),
        embedding_model="sdk-host-embedding",
        embedding_dim=2,
        store=store,
        version_store=versions,
        session_factory=sessions,
        sql_factory=SQLAlchemyFactory(),
        production_mode=True,
    ).build()
    return rag, store, versions


async def test_real_host_injection_is_tenant_isolated_and_idempotent() -> None:
    engine, sessions = await _fixture()
    scopes = (f"t1-{uuid4()}", f"t2-{uuid4()}")
    try:
        first, _, _ = await _client(scopes[0], sessions)
        second, _, _ = await _client(scopes[1], sessions)
        assert (
            await first.ingest(
                [Document(id="doc", text="Tenant A refund policy")], lock_key=scopes[0]
            )
            == 1
        )
        assert (
            await second.ingest(
                [Document(id="doc", text="Tenant B separate policy")], lock_key=scopes[1]
            )
            == 1
        )
        assert await first.count() == 1
        assert await second.count() == 1
        assert (
            await first.ingest(
                [Document(id="doc", text="Tenant A refund policy")], lock_key=scopes[0]
            )
            == 1
        )
        async with sessions() as session:
            rows = await session.execute(
                text(
                    "SELECT scope, count(*) AS total FROM ragnor_sdk_host_vectors "
                    "WHERE scope IN (:a,:b) GROUP BY scope"
                ),
                {"a": scopes[0], "b": scopes[1]},
            )
            assert {row.scope: row.total for row in rows} == {
                scopes[0]: 1,
                scopes[1]: 1,
            }
    finally:
        await engine.dispose()


async def test_mid_upsert_sql_exception_rolls_back_active_version_and_vectors() -> None:
    engine, sessions = await _fixture()
    scope = f"rollback-{uuid4()}"
    try:
        rag, store, versions = await _client(scope, sessions)
        assert (
            await rag.ingest(
                [Document(id="policy", text="Original coverage terms")], lock_key=scope
            )
            == 1
        )
        async with sessions() as session:
            original = await versions.find_active(session)
        assert original is not None
        store.fail_after_insert = True
        with pytest.raises(RuntimeError, match="upsert failed"):
            await rag.ingest(
                [Document(id="policy", text="Changed policy that must rollback")],
                lock_key=scope,
            )
        async with sessions() as session:
            after = await versions.find_active(session)
            assert after is not None
            assert after.version_id == original.version_id
            rows = await session.execute(
                text("SELECT content FROM ragnor_sdk_host_vectors WHERE scope=:scope"),
                {"scope": scope},
            )
            assert rows.scalars().all() == ["Original coverage terms"]
    finally:
        await engine.dispose()
