"""SDK-02 production startup validation avoids implicit dimensions or leaked config."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.capabilities import BackendFeature, FilterOperator, StoreCapabilities

from tests.production_doubles import StoragePort, VersionPort


class Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class HostStore(StoragePort):
    capabilities = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
        features=frozenset({BackendFeature.ISOLATION_SCOPE}),
    )
    embedding_dim = 2


@asynccontextmanager
async def sessions():
    yield object()


def builder(model: str, dimension: int | None):
    return RagnorBuilder(
        embedder=Embedder(),
        embedding_model=model,
        embedding_dim=dimension,
        store=HostStore(),
        version_store=VersionPort(),
        session_factory=sessions,
        production_mode=True,
    )


@pytest.mark.asyncio
async def test_production_unknown_dimension_requires_explicit_registration() -> None:
    with pytest.raises(ValueError, match="dimension"):
        await builder("new-unknown-embedding-model", None).build()


@pytest.mark.asyncio
async def test_production_rejects_zero_or_negative_dimensions() -> None:
    for dimension in (0, -5):
        with pytest.raises(ValueError, match="embedding_dim"):
            await builder("known-model", dimension).build()


@pytest.mark.asyncio
async def test_production_rejects_store_dimension_mismatch_before_write() -> None:
    with pytest.raises(ValueError, match="dimension"):
        await builder("host-model", 3).build()


@pytest.mark.asyncio
async def test_production_explicit_dimension_and_matching_host_store_build() -> None:
    rag = await builder("host-model", 2).build()
    assert rag.store.embedding_dim == 2


@pytest.mark.asyncio
async def test_invalid_model_identifier_does_not_enter_registry() -> None:
    with pytest.raises(ValueError, match="embedding_model"):
        await builder("", 2).build()


def test_ragnor_config_and_builder_settings_validation_and_redaction() -> None:
    from ragnor.builder import BuilderSettings
    from ragnor.config import DeploymentMode, RagnorConfig
    from ragnor.exceptions import ConfigurationError

    cfg = RagnorConfig(
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        openai_api_key="sk-secret-key-99887766",
        deployment_mode=DeploymentMode.PRODUCTION,
    )
    cfg.validate()
    assert "sk-secret-key-99887766" not in repr(cfg)

    bs = BuilderSettings(
        openai_api_key="sk-secret-key-99887766",
    )
    assert "sk-secret-key-99887766" not in repr(bs)

    with pytest.raises(ConfigurationError):
        RagnorConfig(embedding_model="").validate()
    with pytest.raises(ConfigurationError):
        RagnorConfig(embedding_dim=0).validate()
    with pytest.raises(ConfigurationError):
        RagnorConfig(rag_distance_threshold=-1.0).validate()
    with pytest.raises(ConfigurationError):
        RagnorConfig(embedding_drain_batch_size=0).validate()


@pytest.mark.asyncio
async def test_production_factory_lock_key_and_health_check() -> None:
    from ragnor.exceptions import ConfigurationError
    from ragnor.sync.manifest import InMemoryManifestStore
    from ragnor.sync.state import InMemorySyncStateStore
    from ragnor.telemetry.health import HealthState
    from ragnor.types import Document

    # Invalid sql_factory without advisory_lock is rejected in production mode
    with pytest.raises(ConfigurationError, match="advisory_lock"):
        await (
            RagnorBuilder.production(
                Embedder(),
                store=HostStore(),
                version_store=VersionPort(),
                session_factory=sessions,
                embedding_dim=2,
            )
            .with_sql_factory(object())  # type: ignore[arg-type]
            .build()
        )

    # Local client rejects lock_key when no real session_factory is wired
    local_rag = await RagnorBuilder.local(Embedder(), embedding_dim=2).build()
    with pytest.raises(ConfigurationError, match="lock_key"):
        await local_rag.ingest([Document(id="d1", text="hello")], lock_key="scope-a")
    with pytest.raises(ConfigurationError, match="lock_key"):
        await local_rag.ingest([Document(id="d1", text="hello")], lock_key="   ")

    # Health check integration on Ragnor facade
    manifest = InMemoryManifestStore()
    sync_state = InMemorySyncStateStore()
    health_rag = await (
        RagnorBuilder.local(Embedder(), embedding_dim=2)
        .with_manifest_store(manifest)
        .with_sync_state_store(sync_state)
        .build()
    )
    report = await health_rag.check_health(scope="tenant-1", allow_empty=True)
    assert report.state == HealthState.READY
    assert report.is_ready


@pytest.mark.asyncio
async def test_real_sqlite_production_host_ingest_idempotency_and_rollback() -> None:
    """Real SQLite (aiosqlite) transactional production host lifecycle & rollback test."""
    import json
    from dataclasses import asdict
    from uuid import uuid4

    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from ragnor.ingestion.pipeline import VersionRecord
    from ragnor.ingestion.versioning import IngestionSnapshot
    from ragnor.types import Document

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE host_versions ("
                "scope TEXT NOT NULL, version_id TEXT PRIMARY KEY, "
                "snapshot TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE host_vectors ("
                "scope TEXT NOT NULL, doc_id TEXT NOT NULL, "
                "version_id TEXT NOT NULL, content TEXT NOT NULL, "
                "PRIMARY KEY(scope, doc_id))"
            )
        )

    class SqliteVectorStore:
        capabilities = StoreCapabilities(
            filter_operators=frozenset({FilterOperator.EQ}),
            features=frozenset({BackendFeature.ISOLATION_SCOPE}),
        )
        embedding_dim = 2

        def __init__(self, scope: str) -> None:
            self.scope = scope
            self.fail_after_insert = False

        async def upsert(self, documents, session=None, **kwargs) -> None:
            assert session is not None
            for doc in documents:
                await session.execute(
                    text(
                        "INSERT INTO host_vectors (scope, doc_id, version_id, content) "
                        "VALUES (:scope, :doc_id, :version_id, :content)"
                    ),
                    {
                        "scope": self.scope,
                        "doc_id": doc["id"],
                        "version_id": doc["metadata"]["version_id"],
                        "content": doc["text"],
                    },
                )
            if self.fail_after_insert:
                raise RuntimeError("simulated mid-transaction vector write failure")

        async def count(self, session=None, **filters) -> int:
            async def _q(s) -> int:
                res = await s.execute(
                    text("SELECT count(*) FROM host_vectors WHERE scope = :scope"),
                    {"scope": self.scope},
                )
                return int(res.scalar_one())

            if session is not None:
                return await _q(session)
            async with session_maker() as owned:
                return await _q(owned)

        async def vector_search(self, **kwargs):
            async with session_maker() as s:
                res = await s.execute(
                    text(
                        "SELECT doc_id, content, version_id FROM host_vectors WHERE scope = :scope"
                    ),
                    {"scope": self.scope},
                )
                return [
                    {
                        "id": row.doc_id,
                        "document": row.content,
                        "metadata": {"version_id": row.version_id, "tenant_id": self.scope},
                        "distance": 0.1,
                        "keyword_score": None,
                    }
                    for row in res
                ]

        async def keyword_search(self, **kwargs):
            return []

    class SqliteVersionStore:
        def __init__(self, scope: str) -> None:
            self.scope = scope

        async def find_active(self, session) -> VersionRecord | None:
            res = await session.execute(
                text(
                    "SELECT version_id, snapshot FROM host_versions "
                    "WHERE scope = :scope AND active = 1"
                ),
                {"scope": self.scope},
            )
            row = res.mappings().one_or_none()
            if row is None:
                return None
            return VersionRecord(
                version_id=row["version_id"],
                snapshot=IngestionSnapshot(**json.loads(row["snapshot"])),
                source_id=self.scope,
            )

        async def close_active(self, session) -> None:
            await session.execute(
                text("UPDATE host_versions SET active = 0 WHERE scope = :scope"),
                {"scope": self.scope},
            )

        async def create_version(self, session, snapshot: IngestionSnapshot) -> VersionRecord:
            vid = str(uuid4())
            await session.execute(
                text(
                    "INSERT INTO host_versions (scope, version_id, snapshot, active) "
                    "VALUES (:scope, :vid, :snap, 1)"
                ),
                {"scope": self.scope, "vid": vid, "snap": json.dumps(asdict(snapshot))},
            )
            return VersionRecord(version_id=vid, snapshot=snapshot, source_id=self.scope)

        async def delete_chunks(self, session, version_id: str) -> None:
            await session.execute(
                text("DELETE FROM host_vectors WHERE scope = :scope AND version_id = :vid"),
                {"scope": self.scope, "vid": version_id},
            )

    @asynccontextmanager
    async def transactional_sessions():
        async with session_maker() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    try:
        store = SqliteVectorStore("tenant-sqlite")
        versions = SqliteVersionStore("tenant-sqlite")
        rag = await RagnorBuilder.production(
            Embedder(),
            store=store,  # type: ignore[arg-type]
            version_store=versions,
            session_factory=transactional_sessions,
            embedding_model="sqlite-prod-model",
            embedding_dim=2,
        ).build()

        assert await rag.ingest([Document(id="p1", text="Original SQLite policy")]) == 1
        assert await rag.count() == 1
        # Idempotent second ingest
        assert await rag.ingest([Document(id="p1", text="Original SQLite policy")]) == 1

        async with session_maker() as s:
            first_active = await versions.find_active(s)
        assert first_active is not None

        # Simulate mid-transaction failure during re-ingest: must rollback version retirement & delete!
        store.fail_after_insert = True
        with pytest.raises(RuntimeError, match="mid-transaction"):
            await rag.ingest([Document(id="p1", text="Corrupted update that must rollback")])

        async with session_maker() as s:
            after_active = await versions.find_active(s)
            vec_rows = (
                (
                    await s.execute(
                        text("SELECT content FROM host_vectors WHERE scope = 'tenant-sqlite'")
                    )
                )
                .scalars()
                .all()
            )
        assert after_active is not None
        assert after_active.version_id == first_active.version_id
        assert vec_rows == ["Original SQLite policy"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["version_store", "store", "session_factory"])
async def test_production_rejects_nonfunctional_injected_ports_before_writes(component):
    from ragnor.exceptions import ConfigurationError

    candidate = builder("validated-model", 2)
    if component == "version_store":
        candidate._version_store = object()
    elif component == "store":
        candidate._store.upsert = None
    else:
        candidate._session_factory = object()
    with pytest.raises(ConfigurationError, match=component):
        await candidate.build()


@pytest.mark.asyncio
async def test_production_settings_cannot_silently_select_local_fallbacks():
    from ragnor.config import DeploymentMode, RagnorConfig
    from ragnor.exceptions import ConfigurationError

    candidate = RagnorBuilder.local(Embedder()).with_settings(
        RagnorConfig(deployment_mode=DeploymentMode.PRODUCTION)
    )
    with pytest.raises(ConfigurationError, match="explicit"):
        await candidate.build()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["not_callable", "embed_query", "embed_documents"])
async def test_generic_production_provider_rejects_nonexecutable_embedding_ports(invalid):
    from ragnor.exceptions import ConfigurationError

    candidate = builder("host-provider-model", 2)
    if invalid == "not_callable":
        candidate._embedder = object()
    else:
        setattr(candidate._embedder, invalid, None)
    with pytest.raises(ConfigurationError, match="embedder"):
        await candidate.build()


@pytest.mark.asyncio
async def test_generic_batch_callable_can_run_readiness_without_a_vendor_sdk():
    calls = []

    async def host_provider(texts):
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    async def ready():
        assert await host_provider(["readiness"]) == [[1.0, 0.0]]

    candidate = builder("host-provider-model", 2).with_embedder(host_provider)
    async with await candidate.with_startup_check(ready).build():
        assert calls == [["readiness"]]
