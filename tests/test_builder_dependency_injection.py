"""Regression tests for host-owned builder dependencies and SQL adapter isolation."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.strategies import KeywordRetrieval
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.telemetry.events import MemoryEventSink
from ragnor.types import Document


class _Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _LockFactory:
    def __init__(self) -> None:
        self.claimed: list[str] = []

    @asynccontextmanager
    async def advisory_lock(self, session: object, lock_key: str):
        self.claimed.append(lock_key)
        yield


@pytest.mark.asyncio
async def test_builder_uses_injected_session_factory_during_ingestion() -> None:
    session = _Session()

    @asynccontextmanager
    async def session_factory():
        yield session

    builder = RagnorBuilder.local(_Embedder(), embedding_dim=2)
    rag = await builder.with_session_factory(session_factory).build()
    await rag.ingest([Document(id="p1", text="policy refund")])

    assert session.commits == 1
    assert rag.pipeline._session_factory is session_factory
    assert rag.retriever._session_factory is session_factory


@pytest.mark.asyncio
async def test_builder_injected_sql_lock_factory_without_optional_sqlalchemy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()

    @asynccontextmanager
    async def session_factory():
        yield session

    locks = _LockFactory()
    builder = (
        RagnorBuilder.local(_Embedder(), embedding_dim=2)
        .with_session_factory(session_factory)
        .with_sql_factory(locks)
    )
    rag = await builder.build()

    # This is the reported bug: selecting an injected SQL factory must NOT
    # try importing the optional SQLAlchemy implementation.
    monkeypatch.setitem(sys.modules, "ragnor.db.sql_factory", None)
    await rag.ingest([Document(id="p2", text="retention policy")], lock_key="tenant:collection")

    assert locks.claimed == ["tenant:collection"]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_builder_injects_retrieval_strategy_and_event_sink() -> None:
    embedder = _Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    strategy = KeywordRetrieval(lexical_index=store)
    sink = MemoryEventSink()
    rag = await (
        RagnorBuilder.local(embedder, embedding_dim=2, store=store)
        .with_retrieval_strategy(strategy)
        .with_event_sink(sink)
        .build()
    )

    assert rag.retriever._strategy is strategy
    assert rag.retriever._event_sink is sink
    await rag.ingest([Document(id="p3", text="refund policy covered")])
    result = await rag.search("refund")
    assert result
    assert result[0].document == "refund policy covered"
    assert sink.get_by_stage("retrieval")


@pytest.mark.asyncio
async def test_production_builder_never_uses_implicit_local_dependencies() -> None:
    with pytest.raises(ValueError, match="explicit"):
        await RagnorBuilder.local(_Embedder(), embedding_dim=2).with_production_mode().build()


@pytest.mark.asyncio
async def test_production_builder_rejects_known_in_memory_adapters() -> None:
    session = _Session()

    @asynccontextmanager
    async def session_factory():
        yield session

    embedder = _Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    from ragnor.builder import InMemoryVersionStore

    builder = (
        RagnorBuilder.local(embedder, embedding_dim=2, store=store)
        .with_version_store(InMemoryVersionStore(store))
        .with_session_factory(session_factory)
        .with_production_mode()
    )
    with pytest.raises(ValueError, match="InMemory"):
        await builder.build()
