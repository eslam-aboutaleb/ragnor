"""Tests for ragit.ingestion.pipeline.

Covers the ingestion orchestration extracted from
``_do_ingest``: snapshot comparison (skip when the
source is unchanged), version retirement, chunk
stamping, embedding validation, metadata ownership
stamping, and session ownership (the pipeline
commits only the session it owns).
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from ragit.chunking.base import Chunker
from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.dimensions import register_dimension
from ragit.ingestion import (
    IngestionPipeline,
    IngestionSnapshot,
    VersionRecord,
    build_snapshot,
)
from ragit.ingestion.sources import RawDocument
from ragit.stores.base import VectorStore
from ragit.types import Chunk

register_dimension("ragit-pipeline-test-model", 4)


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self, embedding_model: str = "ragit-pipeline-test-model") -> None:
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
        return MagicMock()

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
    assert version_store.created[0].source_hash == hashlib.sha256(b"policy content").hexdigest()


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
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
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
        embedding_model="ragit-pipeline-test-model",
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
    assert version_store.closed == [session_factory.sessions[0]]
    assert version_store.deleted == ["version-0"]
    assert len(version_store.created) == 1
    assert len(store.upserted) == 1


@pytest.mark.asyncio
async def test_run_retires_the_old_version_before_creating_the_new_one() -> None:
    stored_snapshot = IngestionSnapshot(
        source_hash="stale-hash",
        embedding_model="ragit-test-model",
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
    assert len(session_factory.sessions) == 1
    session = session_factory.sessions[0]
    assert session.committed
    assert not session.rolled_back
    assert any("pg_advisory_lock" in sql for sql in session.executed)
    assert any("pg_advisory_unlock" in sql for sql in session.executed)
    assert store.upsert_sessions == [session]


@pytest.mark.asyncio
async def test_run_with_lock_key_rolls_back_on_failure() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()

    class _FailingEmbedder(_DeterministicEmbedder):
        async def __call__(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embed failed")

    pipeline = _pipeline(store, version_store, session_factory, embedder=_FailingEmbedder())

    with pytest.raises(RuntimeError, match="embed failed"):
        await pipeline.run(_source(), version_store, lock_key="policy.md")

    session = session_factory.sessions[0]
    assert session.rolled_back
    assert not session.committed
    # The lock is released even though the ingestion failed.
    assert any("pg_advisory_unlock" in sql for sql in session.executed)


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
    assert any("pg_advisory_lock" in sql for sql in caller_session.executed)
    assert any("pg_advisory_unlock" in sql for sql in caller_session.executed)


@pytest.mark.asyncio
async def test_run_without_lock_key_still_commits_its_owned_session() -> None:
    store = _FakeStore()
    version_store = _FakeVersionStore()
    session_factory = _FakeSessionFactory()
    pipeline = _pipeline(store, version_store, session_factory)

    await pipeline.run(_source(), version_store)

    assert session_factory.sessions[0].committed


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


def test_version_store_protocol_is_runtime_checkable() -> None:
    assert isinstance(_FakeVersionStore(), VectorStore) or True
    # The VersionStore protocol is runtime_checkable; the fake
    # implements every method of the protocol.
    from ragit.ingestion import VersionStore

    assert isinstance(_FakeVersionStore(), VersionStore)
