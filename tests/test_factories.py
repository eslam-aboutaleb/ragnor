"""Tests for the abstract factory layer.

Covers the two seams a host can override: ``SQLFactory``
(sessions, queries, mutations, advisory locking -- built on
``SQLAlchemyFactory``) and ``VectorStoreFactory`` (store
creation -- built on ``RegistryVectorStoreFactory``), the
``get_vector_store`` compatibility path, factory injection
into the ingestion pipeline and the pgvector store, the
convention-based backend loading that keeps every vector
database driver optional, and the lazy-import decoupling
that keeps sqlalchemy optional.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from ragnor.db.sql_factory import SQLAlchemyFactory, SQLFactory
from ragnor.embeddings.dimensions import register_dimension
from ragnor.ingestion import IngestionPipeline, IngestionSnapshot, VersionRecord
from ragnor.ingestion.sources import RawDocument
from ragnor.registry import Registry
from ragnor.stores import (
    InMemoryVectorStore,
    PgVectorStore,
    VectorStore,
    VectorStoreFactory,
    get_vector_store,
)
from ragnor.stores.store_factory import RegistryVectorStoreFactory
from ragnor.types import Chunk

register_dimension("ragnor-factories-test-model", 4)

SQLITE_URL = "sqlite+aiosqlite:///:memory:"
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# SQLAlchemyFactory: real SQL over aiosqlite
# ---------------------------------------------------------------------------


async def test_create_session_factory_returns_callable() -> None:
    session_factory = SQLAlchemyFactory().create_session_factory(SQLITE_URL)

    assert callable(session_factory)
    async with session_factory() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


async def test_execute_fetch_all_and_fetch_scalar() -> None:
    factory = SQLAlchemyFactory()
    session_factory = factory.create_session_factory(SQLITE_URL)

    async with session_factory() as session:
        await factory.execute(session, "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        await factory.execute(session, "INSERT INTO items (name) VALUES (:name)", {"name": "alpha"})
        await factory.execute(session, "INSERT INTO items (name) VALUES (:name)", {"name": "beta"})

        rows = await factory.fetch_all(session, "SELECT id, name FROM items ORDER BY id")
        assert rows == [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]

        count = await factory.fetch_scalar(session, "SELECT COUNT(*) FROM items")
        assert count == 2


async def test_fetch_all_binds_named_params() -> None:
    factory = SQLAlchemyFactory()
    session_factory = factory.create_session_factory(SQLITE_URL)

    async with session_factory() as session:
        await factory.execute(session, "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        await factory.execute(session, "INSERT INTO items (name) VALUES (:name)", {"name": "alpha"})

        rows = await factory.fetch_all(
            session, "SELECT name FROM items WHERE name = :name", {"name": "alpha"}
        )
        assert rows == [{"name": "alpha"}]


# ---------------------------------------------------------------------------
# Abstract factories cannot be instantiated
# ---------------------------------------------------------------------------


def test_sql_factory_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        SQLFactory()


def test_vector_store_factory_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        VectorStoreFactory()


# ---------------------------------------------------------------------------
# RegistryVectorStoreFactory
# ---------------------------------------------------------------------------


class _DummyStore(VectorStore):
    """Minimal concrete store for registry tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        return 0

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        pass


def test_create_defaults_to_memory() -> None:
    store = RegistryVectorStoreFactory().create(table_name="chunks", embedding_dim=1536)

    assert isinstance(store, InMemoryVectorStore)


def test_create_pgvector_provider() -> None:
    settings = SimpleNamespace(vector_store_provider="pgvector")

    store = RegistryVectorStoreFactory().create(
        settings=settings, table_name="chunks", embedding_dim=1536
    )

    assert isinstance(store, PgVectorStore)


def test_create_custom_registered_provider() -> None:
    registry = Registry[VectorStore]()

    @registry.register("custom-backend")
    class CustomBackend(_DummyStore):
        pass

    factory = RegistryVectorStoreFactory(registry=registry)

    store = factory.create(
        settings=SimpleNamespace(vector_store_provider="custom-backend"),
        table_name="chunks",
        embedding_dim=4,
    )

    assert isinstance(store, CustomBackend)
    assert store.kwargs == {"table_name": "chunks", "embedding_dim": 4}


def test_create_forwards_settings_to_builtin_backend() -> None:
    settings = SimpleNamespace(vector_store_provider="memory", rag_distance_metric="cosine")

    store = RegistryVectorStoreFactory().create(
        settings=settings, table_name="chunks", embedding_dim=4
    )

    assert isinstance(store, InMemoryVectorStore)
    assert store.metric == "cosine"


def test_create_forwards_settings_only_when_constructor_declares_it() -> None:
    registry = Registry[VectorStore]()

    @registry.register("settings-aware")
    class _SettingsAwareStore(_DummyStore):
        def __init__(self, settings: Any = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.settings_received = settings

    factory = RegistryVectorStoreFactory(registry=registry)

    store = factory.create(
        settings=SimpleNamespace(vector_store_provider="settings-aware"),
        table_name="chunks",
        embedding_dim=4,
    )

    assert isinstance(store, _SettingsAwareStore)
    assert store.settings_received is not None
    assert "settings" not in store.kwargs


def test_host_registered_pgvector_override_is_respected() -> None:
    registry = Registry[VectorStore]()

    @registry.register("pgvector")
    class DummyPgVector(_DummyStore):
        pass

    # The host's registration wins: the name is already in the
    # registry, so the builtin pgvector module is not imported.
    factory = RegistryVectorStoreFactory(registry=registry)

    store = factory.create(
        settings=SimpleNamespace(vector_store_provider="pgvector"),
        table_name="chunks",
        embedding_dim=1536,
    )

    assert isinstance(store, DummyPgVector)
    assert store.kwargs == {"table_name": "chunks", "embedding_dim": 1536}


def test_create_unknown_provider_raises_key_error() -> None:
    factory = RegistryVectorStoreFactory()
    settings = SimpleNamespace(vector_store_provider="nope")

    with pytest.raises(KeyError, match="Unknown provider 'nope'"):
        factory.create(settings=settings, table_name="chunks", embedding_dim=1536)


def test_create_non_identifier_provider_raises_key_error() -> None:
    factory = RegistryVectorStoreFactory()
    settings = SimpleNamespace(vector_store_provider="not-a-module")

    with pytest.raises(KeyError, match="Unknown provider 'not-a-module'"):
        factory.create(settings=settings, table_name="chunks", embedding_dim=1536)


def test_create_backend_with_missing_dependency_raises_import_error() -> None:
    result = _run_with_sqlalchemy_blocked(
        "import sys\n"
        "sys.modules['sqlalchemy'] = None\n"
        "from ragnor.stores.store_factory import RegistryVectorStoreFactory\n"
        "from types import SimpleNamespace\n"
        "try:\n"
        "    RegistryVectorStoreFactory().create(\n"
        "        settings=SimpleNamespace(vector_store_provider='pgvector'),\n"
        "        table_name='chunks', embedding_dim=1536,\n"
        "    )\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )

    assert result.returncode == 0, result.stderr
    assert "pgvector" in result.stdout
    assert "extra" in result.stdout


# ---------------------------------------------------------------------------
# get_vector_store compatibility
# ---------------------------------------------------------------------------


def test_get_vector_store_defaults_to_memory() -> None:
    store = get_vector_store(table_name="chunks", embedding_dim=1536)

    assert isinstance(store, InMemoryVectorStore)


def test_get_vector_store_memory_provider() -> None:
    settings = SimpleNamespace(vector_store_provider="memory")

    store = get_vector_store(table_name="chunks", embedding_dim=1536, settings=settings)

    assert isinstance(store, InMemoryVectorStore)


# ---------------------------------------------------------------------------
# Custom SQLFactory injected into the ingestion pipeline
# ---------------------------------------------------------------------------


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self) -> None:
        self.embedding_model = "ragnor-factories-test-model"
        self.rag_distance_threshold = 1.3
        self.vector_store_provider = "memory"
        self.openai_api_key = "test-key"
        self.openai_api_base = ""
        self.embedding_drain_interval_seconds = 5.0
        self.embedding_drain_batch_size = 10
        self.job_stale_after_seconds = 300.0
        self.worker_id = "test-worker"
        self.ingest_on_startup = True
        self.database_url = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"


class _FakeEmbedder:
    """Deterministic 4-dimension embedder."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)) + i for i in range(4)] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class _FakeChunker:
    """Chunker producing a single chunk."""

    config = {"chunk_size": 600, "overlap": 100}

    @property
    def version(self) -> str:
        return "test-chunker-v1"

    def chunk(self, text: str) -> list[Chunk]:
        return [Chunk(id="chunk-0", text=text, metadata={})]


class _FakeStore:
    """Vector store recording upsert calls."""

    def __init__(self) -> None:
        self.upserted: list[list[dict[str, Any]]] = []

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        return 0

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        self.upserted.append(documents)


class _FakeVersionStore:
    """Version store with no active version."""

    async def find_active(self, session: Any) -> VersionRecord | None:
        return None

    async def close_active(self, session: Any) -> None:
        pass

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        return VersionRecord(version_id="version-1", snapshot=snapshot, source_id="source-1")

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        pass


class _StubSource:
    """Document source yielding one document."""

    async def load(self) -> list[RawDocument]:
        return [RawDocument(source_name="src", content="content")]


class _FakeSession:
    """Async session recording executed SQL."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, statement: Any, params: Any = None) -> MagicMock:
        self.executed.append(str(statement))
        return MagicMock()

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


class _SessionContext:
    """Context manager yielding a fake session."""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class _FakeSessionFactory:
    """Session factory recording every session it opens."""

    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []

    def __call__(self) -> _SessionContext:
        session = _FakeSession()
        self.sessions.append(session)
        return _SessionContext(session)


class _RecordingLockFactory(SQLFactory):
    """SQL factory recording advisory_lock usage."""

    def __init__(self) -> None:
        self.lock_keys: list[str] = []

    def create_session_factory(self, database_url: str) -> Any:
        raise NotImplementedError

    async def fetch_all(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return []

    async def fetch_scalar(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> Any:
        return None

    async def execute(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> None:
        pass

    @asynccontextmanager
    async def advisory_lock(self, session: Any, key: str) -> AsyncIterator[None]:
        self.lock_keys.append(key)
        yield


async def test_pipeline_uses_injected_sql_factory_for_advisory_lock() -> None:
    sql_factory = _RecordingLockFactory()
    session_factory = _FakeSessionFactory()
    pipeline = IngestionPipeline(
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        chunker=_FakeChunker(),
        session_factory=session_factory,
        settings=_FakeSettings(),
        sql_factory=sql_factory,
    )

    count = await pipeline.run(_StubSource(), _FakeVersionStore(), lock_key="src")

    assert count == 1
    assert sql_factory.lock_keys == ["src"]
    # The SQLAlchemy factory was not used: no advisory-lock SQL
    # reached the session the pipeline owns.
    assert not any("pg_advisory_lock" in sql for sql in session_factory.sessions[0].executed)


# ---------------------------------------------------------------------------
# Custom SQLFactory injected into PgVectorStore
# ---------------------------------------------------------------------------


class _RecordingSQLFactory(SQLFactory):
    """SQL factory recording every statement routed through it."""

    def __init__(self) -> None:
        self.fetched_all: list[tuple[str, dict[str, Any] | None]] = []
        self.fetched_scalars: list[tuple[str, dict[str, Any] | None]] = []
        self.executed: list[tuple[str, dict[str, Any] | None]] = []

    def create_session_factory(self, database_url: str) -> Any:
        raise NotImplementedError

    async def fetch_all(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.fetched_all.append((statement, params))
        return []

    async def fetch_scalar(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> Any:
        self.fetched_scalars.append((statement, params))
        return 0

    async def execute(
        self, session: Any, statement: str, params: dict[str, Any] | None = None
    ) -> None:
        self.executed.append((statement, params))

    @asynccontextmanager
    async def advisory_lock(self, session: Any, key: str) -> AsyncIterator[None]:
        yield


async def test_pgvector_store_routes_sql_through_injected_factory() -> None:
    sql_factory = _RecordingSQLFactory()
    store = PgVectorStore(
        table_name="policy_chunks",
        embedding_dim=1536,
        session_factory=MagicMock(),
        sql_factory=sql_factory,
    )
    session = MagicMock()

    count = await store.count(session=session)

    assert count == 0
    assert len(sql_factory.fetched_scalars) == 1
    statement, params = sql_factory.fetched_scalars[0]
    assert "SELECT COUNT(*) FROM policy_chunks" in statement
    assert params == {}

    await store.upsert(
        documents=[{"id": "1", "text": "test", "embedding": [0.0] * 1536}],
        session=session,
    )

    assert len(sql_factory.executed) == 1
    assert "INSERT INTO policy_chunks" in sql_factory.executed[0][0]


# ---------------------------------------------------------------------------
# Custom VectorStoreFactory
# ---------------------------------------------------------------------------


class _FixedStoreFactory(VectorStoreFactory):
    """Vector store factory returning a fixed store."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def create(self, settings: Any = None, **kwargs: Any) -> VectorStore:
        return self._store


def test_custom_vector_store_factory_returns_fixed_store() -> None:
    store = InMemoryVectorStore(table_name="chunks", embedding_dim=4)
    factory = _FixedStoreFactory(store)

    assert factory.create() is store
    assert factory.create(settings=SimpleNamespace(vector_store_provider="memory")) is store


# ---------------------------------------------------------------------------
# Lazy import decoupling: core imports without sqlalchemy
# ---------------------------------------------------------------------------


def _run_with_sqlalchemy_blocked(code: str) -> subprocess.CompletedProcess[str]:
    """Run python code in a subprocess with sqlalchemy blocked."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else SRC_DIR
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run python code in a subprocess with src on the path."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else SRC_DIR
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_factory_loads_backend_module_by_convention() -> None:
    for provider, class_name in (
        ("qdrant", "QdrantStore"),
        ("chroma", "ChromaStore"),
        ("faiss", "FaissStore"),
    ):
        result = _run_python(
            "from types import SimpleNamespace\n"
            "from ragnor.stores.store_factory import RegistryVectorStoreFactory\n"
            "from ragnor.stores.registry import vector_store_registry\n"
            f"assert {provider!r} not in vector_store_registry.list()\n"
            "store = RegistryVectorStoreFactory().create(\n"
            f"    settings=SimpleNamespace(vector_store_provider={provider!r}),\n"
            "    table_name='chunks', embedding_dim=4,\n"
            ")\n"
            "print(type(store).__name__)\n"
        )

        assert result.returncode == 0, result.stderr
        assert class_name in result.stdout


def test_core_modules_import_without_sqlalchemy() -> None:
    result = _run_with_sqlalchemy_blocked(
        "import sys; sys.modules['sqlalchemy'] = None; "
        "import ragnor.stores, ragnor.ingestion; print('ok')"
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_pgvector_store_access_requires_pgvector_extra() -> None:
    result = _run_with_sqlalchemy_blocked(
        "import sys\n"
        "sys.modules['sqlalchemy'] = None\n"
        "import ragnor.stores\n"
        "try:\n"
        "    ragnor.stores.PgVectorStore\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )

    assert result.returncode == 0, result.stderr
    assert "pgvector" in result.stdout
    assert "extra" in result.stdout
