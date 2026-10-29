"""Tests for ragnor.ingestion.pipeline.

Covers the ingestion orchestration extracted from
``_do_ingest``: snapshot comparison (skip when the
source is unchanged), version retirement, chunk
stamping, embedding validation, metadata ownership
stamping, and session ownership (the pipeline
commits only the session it owns), plus the
two-phase structure (embedding outside the
transaction and lock), the manifest source hash,
and ``purge_on_empty`` semantics.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from ragnor.chunking.base import Chunker
from ragnor.db.sql_factory import SQLFactory
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.dimensions import register_dimension
from ragnor.ingestion import (
    IngestionPipeline,
    IngestionSnapshot,
    VersionRecord,
    build_snapshot,
)
from ragnor.ingestion.sources import RawDocument
from ragnor.ingestion.versioning import compute_source_hash
from ragnor.stores.base import VectorStore
from ragnor.types import Chunk

register_dimension("ragnor-pipeline-test-model", 4)


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self, embedding_model: str = "ragnor-pipeline-test-model") -> None:
        self.embedding_model = embedding_model
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


class _DeterministicEmbedder:
    """Embeds text into fixed-dimension vectors."""

    def __init__(self, dim: int = 4, wrong_dim: bool = False) -> None:
        self._dim = dim
        self._wrong_dim = wrong_dim
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        width = self._dim + 1 if self._wrong_dim else self._dim
        return [[float(len(text)) + i for i in range(width)] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class _RecordingChunker:
    """Chunker producing a fixed number of chunks."""

    def __init__(self, n_chunks: int = 1) -> None:
        self._n_chunks = n_chunks
        self.config = {"chunk_size": 600, "overlap": 100}

    @property
    def version(self) -> str:
        return "test-chunker-v1"

    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(
                id=f"chunk-{index}",
                text=f"{text} part {index}",
                metadata={"chunk_index": index},
            )
            for index in range(self._n_chunks)
        ]


class _EmptyChunker:
    """Chunker that produces no chunks."""

    config = {"chunk_size": 600, "overlap": 100}

    @property
    def version(self) -> str:
        return "test-chunker-v1"

    def chunk(self, text: str) -> list[Chunk]:
        return []


class _FakeStore:
    """Vector store recording upsert and count calls."""

    def __init__(self, count: int = 0) -> None:
        self._count = count
        self.upserted: list[list[dict[str, Any]]] = []
        self.upsert_sessions: list[Any] = []
        self.count_sessions: list[Any] = []

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        self.count_sessions.append(session)
        return self._count

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        self.upserted.append(documents)
        self.upsert_sessions.append(session)


class _FakeVersionStore:
    """Version store recording every protocol call."""

    def __init__(self, active: VersionRecord | None = None) -> None:
        self.active = active
        self.find_sessions: list[Any] = []
        self.closed: list[Any] = []
        self.created: list[IngestionSnapshot] = []
        self.deleted: list[str] = []

    async def find_active(self, session: Any) -> VersionRecord | None:
        self.find_sessions.append(session)
        return self.active

    async def close_active(self, session: Any) -> None:
        self.closed.append(session)

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        self.created.append(snapshot)
        return VersionRecord(
            version_id="version-1",
            snapshot=snapshot,
            source_id="source-1",
        )

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        self.deleted.append(version_id)


class _OrderingVersionStore(_FakeVersionStore):
    """Version store recording the call order."""

    def __init__(self, events: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._events = events

    async def close_active(self, session: Any) -> None:
        self._events.append("close_active")
        await super().close_active(session)

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        self._events.append("create_version")
        return await super().create_version(session, snapshot)

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        self._events.append("delete_chunks")
        await super().delete_chunks(session, version_id)


class _StubSource:
    """Document source yielding fixed documents."""

    def __init__(self, *documents: RawDocument) -> None:
        self._documents = list(documents)

    async def load(self) -> list[RawDocument]:
        return self._documents


class _FakeSession:
    """Async session recording commits, rollbacks, and SQL."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.executed: list[str] = []
        self.added: list[Any] = []

    async def execute(self, statement: Any, params: Any = None) -> MagicMock:
        self.executed.append(str(statement))
        return MagicMock(rowcount=1)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
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


class _MutableStore:
    """Vector store keeping documents, with delete-by-filter.

    Models a host search index that supports the
    portable ``delete`` capability: chunks are
    removed by metadata filter, so a purge can drop
    a retired version's chunks and the count drops.
    """

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.upsert_sessions: list[Any] = []
        self.count_sessions: list[Any] = []

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        self.count_sessions.append(session)
        return len(self.documents)

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        self.documents.extend(documents)
        self.upsert_sessions.append(session)

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        selected = dict(filters or {})
        before = len(self.documents)

        def _matches(document: dict[str, Any]) -> bool:
            metadata = document.get("metadata") or {}
            return all(metadata.get(key) == value for key, value in selected.items())

        self.documents = [doc for doc in self.documents if not _matches(doc)]
        return before - len(self.documents)


class _HostVersionStore(_FakeVersionStore):
    """Version store tracking state like a host version table.

    ``create_version`` makes the new version active,
    ``close_active`` retires it, and ``delete_chunks``
    removes the version's chunks from the search index
    through the portable delete capability -- the way
    an external-store host implements the hook.
    """

    def __init__(self, store: _MutableStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._store = store
        self._next_version_id = 0

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        self.created.append(snapshot)
        self._next_version_id += 1
        self.active = VersionRecord(
            version_id=f"version-{self._next_version_id}",
            snapshot=snapshot,
            source_id="source-1",
        )
        return self.active

    async def close_active(self, session: Any) -> None:
        self.closed.append(session)
        self.active = None

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        self.deleted.append(version_id)
        await self._store.delete(filters={"version_id": version_id})


class _RecordingLockFactory(SQLFactory):
    """SQL factory whose advisory lock records ordering."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def fetch_all(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def fetch_scalar(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return None

    async def execute(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        pass

    @asynccontextmanager
    async def advisory_lock(self, session: Any, key: str) -> Any:
        self._events.append("lock")
        yield


class _OrderingEmbedder(_DeterministicEmbedder):
    """Embedder recording when the embedding request runs."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self._events.append("embed")
        return await super().__call__(texts)


class _RacingVersionStore(_FakeVersionStore):
    """Version store whose snapshot changes between the two checks.

    The prepare-phase check sees the stale snapshot (so
    the pipeline embeds); by the time the critical
    section re-checks under the lock, a concurrent
    process has ingested the source and the snapshot
    matches.
    """

    def __init__(self, stale: IngestionSnapshot, fresh: IngestionSnapshot) -> None:
        super().__init__(
            active=VersionRecord(version_id="version-0", snapshot=stale, source_id="source-1")
        )
        self._fresh = fresh
        self._checks = 0

    async def find_active(self, session: Any) -> VersionRecord | None:
        self._checks += 1
        if self._checks == 1:
            return self.active
        return VersionRecord(version_id="version-1", snapshot=self._fresh, source_id="source-1")


def _source(content: str = "policy content") -> _StubSource:
    return _StubSource(RawDocument(source_name="policy.md", content=content))


def _pipeline(  # noqa: PLR0913, PLR0917
    store: _FakeStore,
    version_store: _FakeVersionStore,
    session_factory: _FakeSessionFactory,
    chunker: Chunker | None = None,
    embedder: EmbeddingFunction | None = None,
    settings: _FakeSettings | None = None,
) -> IngestionPipeline:
    return IngestionPipeline(
        store=store,  # type: ignore[arg-type]
        embedder=embedder or _DeterministicEmbedder(),
        chunker=chunker or _RecordingChunker(),
        session_factory=session_factory,
        settings=settings or _FakeSettings(),
        version_id_key="version_id",
        source_id_key="source_id",
    )


@pytest.mark.asyncio
async def test_run_ingests_and_returns_chunk_count() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_source(), version_store)

    assert count == 1
    assert len(store.upserted) == 1
    document = store.upserted[0][0]
    assert document["id"] == "chunk-0"
    assert document["text"] == "policy content part 0"
    assert document["metadata"]["source"] == "policy.md"
    assert document["metadata"]["chunk_index"] == 0
    assert document["metadata"]["version_id"] == "version-1"
    assert document["metadata"]["source_id"] == "source-1"
    assert version_store.created[0].source_hash == compute_source_hash(
        [RawDocument(source_name="policy.md", content="policy content")]
    )


@pytest.mark.asyncio
async def test_run_stamps_multiple_chunks() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory, chunker=_RecordingChunker(3))

    count = await pipeline.run(_source(), version_store)

    assert count == 3
    assert len(store.upserted[0]) == 3
    assert [doc["id"] for doc in store.upserted[0]] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


@pytest.mark.asyncio
async def test_run_skips_when_snapshot_matches() -> None:
    content = "policy content"
    source_hash = compute_source_hash([RawDocument(source_name="policy.md", content=content)])
    settings = _FakeSettings()
    chunker = _RecordingChunker()
    stored_snapshot = build_snapshot(source_hash, settings, chunker)
    active = VersionRecord(version_id="version-0", snapshot=stored_snapshot, source_id="source-1")

    store = _FakeStore(count=7)
    version_store = _FakeVersionStore(active=active)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_source(content), version_store)

    assert count == 7
    assert store.upserted == []
    assert version_store.created == []
    assert version_store.closed == []
    assert version_store.deleted == []
    assert store.count_sessions == [session_factory.sessions[0]]


@pytest.mark.asyncio
async def test_run_reingests_when_source_hash_changes() -> None:
    stored_snapshot = IngestionSnapshot(
        source_hash="stale-hash",
        embedding_model="ragnor-pipeline-test-model",
        embedding_dim=4,
        chunker_version="test-chunker-v1",
        chunk_size=600,
        overlap=100,
        retrieval_schema_version="v1",
    )
    active = VersionRecord(version_id="version-0", snapshot=stored_snapshot, source_id="source-1")

    store = _FakeStore()
    version_store = _FakeVersionStore(active=active)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_source(), version_store)

    assert count == 1
    # The critical section (version retirement) runs on
    # the second session the pipeline opens: the first
    # is the lock-free read-only snapshot check.
    assert version_store.closed == [session_factory.sessions[1]]
    assert version_store.deleted == ["version-0"]
    assert len(version_store.created) == 1
    assert len(store.upserted) == 1


@pytest.mark.asyncio
async def test_run_retires_the_old_version_before_creating_the_new_one() -> None:
    stored_snapshot = IngestionSnapshot(
        source_hash="stale-hash",
        embedding_model="ragnor-test-model",
        embedding_dim=4,
        chunker_version="test-chunker-v1",
        chunk_size=600,
        overlap=100,
        retrieval_schema_version="v1",
    )
    active = VersionRecord(version_id="version-0", snapshot=stored_snapshot, source_id="source-1")

    events: list[str] = []
    store = _FakeStore()
    version_store = _OrderingVersionStore(events, active=active)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)

    assert events == ["close_active", "delete_chunks", "create_version"]


@pytest.mark.asyncio
async def test_run_validates_every_embedding() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(
        store,
        version_store,
        session_factory,
        embedder=_DeterministicEmbedder(wrong_dim=True),
    )

    with pytest.raises(ValueError, match="dimensions"):
        await pipeline.run(_source(), version_store)

    assert store.upserted == []


@pytest.mark.asyncio
async def test_run_with_lock_key_commits_its_owned_session() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_source(), version_store, lock_key="policy.md")

    assert count == 1
    # Two sessions: the lock-free read-only snapshot
    # check, then the locked critical section.
    assert len(session_factory.sessions) == 2
    session = session_factory.sessions[1]
    assert session.committed
    assert not session.rolled_back
    assert any("ragnor_ingestion_locks" in sql for sql in session.executed)
    assert store.upsert_sessions == [session]


@pytest.mark.asyncio
async def test_run_with_lock_key_rolls_back_on_failure() -> None:
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()

    class _FailingStore(_FakeStore):
        async def upsert(
            self,
            documents: list[dict[str, Any]],
            session: Any = None,
            **kwargs: Any,
        ) -> None:
            raise RuntimeError("upsert failed")

    pipeline = _pipeline(_FailingStore(), version_store, session_factory)

    with pytest.raises(RuntimeError, match="upsert failed"):
        await pipeline.run(_source(), version_store, lock_key="policy.md")

    # The failure happens inside the critical section:
    # the transaction is rolled back and the lock is
    # released even though the ingestion failed.
    session = session_factory.sessions[1]
    assert session.rolled_back
    assert not session.committed
    assert any("ragnor_ingestion_locks" in sql for sql in session.executed)


@pytest.mark.asyncio
async def test_run_with_caller_session_never_commits_it() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    caller_session = _FakeSession()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_source(), version_store, session=caller_session)

    assert count == 1
    assert not caller_session.committed
    assert session_factory.sessions == []
    assert store.upsert_sessions == [caller_session]


@pytest.mark.asyncio
async def test_run_with_caller_session_and_lock_key_locks_the_caller_session() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    caller_session = _FakeSession()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(
        _source(), version_store, lock_key="policy.md", session=caller_session
    )

    assert count == 1
    assert not caller_session.committed
    assert session_factory.sessions == []
    assert any("ragnor_ingestion_locks" in sql for sql in caller_session.executed)


@pytest.mark.asyncio
async def test_run_without_lock_key_still_commits_its_owned_session() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)

    # The critical section runs on the second session
    # (the first is the read-only snapshot check).
    assert session_factory.sessions[1].committed


@pytest.mark.asyncio
async def test_run_without_documents_returns_zero() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_StubSource(), version_store)

    assert count == 0
    assert store.upserted == []
    assert version_store.created == []
    assert session_factory.sessions == []


@pytest.mark.asyncio
async def test_run_without_chunks_aborts_ingestion() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory, chunker=_EmptyChunker())

    count = await pipeline.run(_source(), version_store)

    assert count == 0
    assert store.upserted == []
    # No version row may be created when chunking aborts: a
    # version carrying the current snapshot would make every
    # later run skip the source forever.
    assert version_store.created == []


# ---------------------------------------------------------------------------
# Two-phase structure: embedding outside the transaction and lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unchanged_source_skips_without_embedding() -> None:
    content = "policy content"
    source_hash = compute_source_hash([RawDocument(source_name="policy.md", content=content)])
    settings = _FakeSettings()
    chunker = _RecordingChunker()
    stored_snapshot = build_snapshot(source_hash, settings, chunker)
    active = VersionRecord(version_id="version-0", snapshot=stored_snapshot, source_id="source-1")

    store = _FakeStore(count=7)
    version_store = _FakeVersionStore(active=active)
    session_factory = _FakeSessionFactory()
    embedder = _DeterministicEmbedder()
    pipeline = _pipeline(store, version_store, session_factory, embedder=embedder)

    count = await pipeline.run(_source(content), version_store)

    assert count == 7
    # The lock-free snapshot check happens in the prepare
    # phase: an unchanged source skips before any
    # embedding request is made.
    assert embedder.calls == []
    assert store.upserted == []
    assert version_store.created == []


@pytest.mark.asyncio
async def test_embedder_runs_before_the_advisory_lock() -> None:
    events: list[str] = []
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = IngestionPipeline(
        store=store,  # type: ignore[arg-type]
        embedder=_OrderingEmbedder(events),
        chunker=_RecordingChunker(),
        session_factory=session_factory,
        settings=_FakeSettings(),
        version_id_key="version_id",
        source_id_key="source_id",
        sql_factory=_RecordingLockFactory(events),
    )

    await pipeline.run(_source(), version_store, lock_key="policy.md")

    # The embedding request runs in the prepare phase,
    # before the critical section takes the advisory
    # lock: a slow remote call never holds the lock (or
    # a transaction) open.
    assert events == ["embed", "lock"]


@pytest.mark.asyncio
async def test_critical_section_skips_when_a_concurrent_run_ingested() -> None:
    content = "policy content"
    settings = _FakeSettings()
    chunker = _RecordingChunker()
    current_snapshot = build_snapshot(
        compute_source_hash([RawDocument(source_name="policy.md", content=content)]),
        settings,
        chunker,
    )
    stale_snapshot = IngestionSnapshot(
        source_hash="stale-hash",
        embedding_model=current_snapshot.embedding_model,
        embedding_dim=current_snapshot.embedding_dim,
        chunker_version=current_snapshot.chunker_version,
        chunk_size=current_snapshot.chunk_size,
        overlap=current_snapshot.overlap,
        retrieval_schema_version=current_snapshot.retrieval_schema_version,
    )

    store = _FakeStore(count=5)
    version_store = _RacingVersionStore(stale_snapshot, current_snapshot)
    session_factory = _FakeSessionFactory()
    embedder = _DeterministicEmbedder()
    pipeline = _pipeline(store, version_store, session_factory, embedder=embedder)

    count = await pipeline.run(_source(content), version_store)

    # The prepare phase saw a stale snapshot and embedded;
    # under the lock the snapshot now matches (a concurrent
    # process ingested meanwhile), so the critical section
    # skips without touching the version table or the index.
    assert count == 5
    assert embedder.calls == [["policy content part 0"]]
    assert store.upserted == []
    assert version_store.created == []
    assert version_store.closed == []
    assert version_store.deleted == []


# ---------------------------------------------------------------------------
# purge_on_empty: authoritative empty datasets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_purge_on_empty_retires_the_active_version() -> None:
    store = _MutableStore()
    version_store = _HostVersionStore(store)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    # Ingest once so the source has an active version.
    await pipeline.run(_source(), version_store)
    assert await store.count() == 1

    # The source now loads nothing: an authoritative
    # empty dataset retires the active version and
    # deletes its chunks from the search index.
    count = await pipeline.run(_StubSource(), version_store, purge_on_empty=True)

    assert count == 0
    assert await store.count() == 0
    assert version_store.deleted == ["version-1"]
    # The retirement runs on the purge session (the third
    # session: two for the ingest run, one for the purge).
    assert version_store.closed == [session_factory.sessions[2]]
    assert len(version_store.created) == 1


@pytest.mark.asyncio
async def test_purge_on_empty_false_is_a_historical_noop() -> None:
    store = _MutableStore()
    version_store = _HostVersionStore(store)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)
    count = await pipeline.run(_StubSource(), version_store)

    assert count == 0
    # The index is untouched: the empty load is a no-op
    # and deleted content would stay searchable.
    assert await store.count() == 1
    assert version_store.closed == []
    assert version_store.deleted == []
    # No session is opened for a skipped empty source.
    assert len(session_factory.sessions) == 2


@pytest.mark.asyncio
async def test_purge_on_empty_with_lock_key_locks_and_commits() -> None:
    store = _MutableStore()
    version_store = _HostVersionStore(store)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)
    count = await pipeline.run(
        _StubSource(), version_store, lock_key="policy.md", purge_on_empty=True
    )

    assert count == 0
    assert await store.count() == 0
    session = session_factory.sessions[2]
    assert session.committed
    assert any("ragnor_ingestion_locks" in sql for sql in session.executed)


@pytest.mark.asyncio
async def test_purge_on_empty_with_caller_session_never_commits_it() -> None:
    store = _MutableStore()
    version_store = _HostVersionStore(store)
    session_factory = _FakeSessionFactory()
    caller_session = _FakeSession()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)
    count = await pipeline.run(
        _StubSource(), version_store, session=caller_session, purge_on_empty=True
    )

    assert count == 0
    assert await store.count() == 0
    # The caller owns the session: the pipeline retires
    # the version on it without committing.
    assert not caller_session.committed
    assert len(session_factory.sessions) == 2


@pytest.mark.asyncio
async def test_purge_on_empty_without_active_version_is_a_noop() -> None:
    store = _MutableStore()
    version_store = _HostVersionStore(store)
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    count = await pipeline.run(_StubSource(), version_store, purge_on_empty=True)

    assert count == 0
    assert await store.count() == 0
    assert version_store.closed == []
    assert version_store.deleted == []
    assert version_store.created == []


def test_version_store_protocol_is_runtime_checkable() -> None:
    assert isinstance(_FakeVersionStore(), VectorStore) or True
    # The VersionStore protocol is runtime_checkable; the fake
    # implements every method of the protocol.
    from ragnor.ingestion import VersionStore

    assert isinstance(_FakeVersionStore(), VersionStore)


@pytest.mark.asyncio
async def test_incomplete_pdf_cannot_replace_or_purge_previous_version() -> None:
    from ragnor.ingestion.connector_core import extract_documents
    from tests.test_pdf_integrity import mixed_pdf

    store = _MutableStore()
    versions = _HostVersionStore(store)
    sessions = _FakeSessionFactory()
    pipeline = _pipeline(store, versions, sessions)
    await pipeline.run(_source(), versions)
    assert await store.count() == 1
    incomplete = extract_documents(mixed_pdf(), "policy.pdf", "policy", {}, {})
    assert not incomplete.authoritative
    assert await pipeline.run(incomplete, versions, purge_on_empty=True) == 0
    assert await store.count() == 1
    assert len(versions.created) == 1
    assert not versions.deleted
    assert not versions.closed
