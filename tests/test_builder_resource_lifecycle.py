"""Opt-in SDK resource ownership and async context-manager regression tests."""

from __future__ import annotations

import asyncio

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.types import Document


class HostEmbedder:
    def __init__(self) -> None:
        self.close_calls = 0

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)

    async def aclose(self) -> None:
        self.close_calls += 1


class OwnedResource:
    def __init__(self, name: str, events: list[str], error: Exception | None = None) -> None:
        self.name = name
        self.events = events
        self.error = error

    async def aclose(self) -> None:
        self.events.append(self.name)
        if self.error is not None:
            raise self.error


class BorrowedStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__(embedding_dim=2)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_owned_resources_close_in_reverse_order_exactly_once() -> None:
    events: list[str] = []
    first = OwnedResource("first", events)
    second = OwnedResource("second", events)
    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(first)
        .with_owned_resource(second)
        .with_owned_resource(first)
        .build()
    )
    async with rag:
        assert await rag.ingest([Document(id="doc", text="knowledge")]) == 1
    await rag.aclose()
    assert events == ["second", "first"]
    with pytest.raises(RuntimeError, match="closed"):
        await rag.count()


@pytest.mark.asyncio
async def test_injected_host_resources_are_borrowed_unless_explicitly_owned() -> None:
    embedder = HostEmbedder()
    store = BorrowedStore()
    rag = await RagnorBuilder.local(embedder, embedding_dim=2, store=store).build()
    async with rag:
        assert await rag.count() == 0
    assert embedder.close_calls == 0
    assert store.close_calls == 0


@pytest.mark.asyncio
async def test_failing_owned_close_still_closes_other_owned_resources() -> None:
    events: list[str] = []
    first = OwnedResource("first", events)
    failing = OwnedResource("failing", events, ValueError("close failed"))
    last = OwnedResource("last", events)
    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(first)
        .with_owned_resource(failing)
        .with_owned_resource(last)
        .build()
    )
    with pytest.raises(ValueError, match="close failed"):
        await rag.aclose()
    assert events == ["last", "failing", "first"]
    await rag.aclose()
    assert events == ["last", "failing", "first"]


@pytest.mark.asyncio
async def test_invalid_owned_resource_rejected_before_build() -> None:
    builder = RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
    with pytest.raises(TypeError, match="aclose"):
        builder.with_owned_resource(object())


async def fail_in_context() -> None:
    """A runtime failure exercises context-manager exit without unreachable code."""
    raise ValueError("body")


@pytest.mark.asyncio
async def test_async_context_cleans_up_after_body_exception() -> None:
    events: list[str] = []
    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("owned", events))
        .build()
    )
    with pytest.raises(ValueError, match="body"):
        async with rag:
            await fail_in_context()
    assert events == ["owned"]
    with pytest.raises(RuntimeError, match="closed"):
        await rag.__aenter__()


@pytest.mark.asyncio
async def test_builder_transfers_owned_resources_to_only_one_facade() -> None:
    events: list[str] = []
    owned = OwnedResource("once", events)
    builder = RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
    builder.with_owned_resource(owned)
    first = await builder.build()
    second = await builder.build()
    await first.aclose()
    await second.aclose()
    assert events == ["once"]


@pytest.mark.asyncio
async def test_cancelled_close_waits_for_owned_cleanup_before_propagating() -> None:
    events: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowResource:
        async def aclose(self) -> None:
            started.set()
            await release.wait()
            events.append("slow")

    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("early", events))
        .with_owned_resource(SlowResource())
        .build()
    )
    closing = asyncio.create_task(rag.aclose())
    await asyncio.wait_for(started.wait(), timeout=3)
    closing.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=3)
    assert events == ["slow", "early"]


@pytest.mark.asyncio
async def test_simultaneous_close_callers_share_one_cleanup() -> None:
    events: list[str] = []
    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("once", events))
        .build()
    )
    await asyncio.gather(rag.aclose(), rag.aclose(), rag.aclose())
    assert events == ["once"]


@pytest.mark.asyncio
async def test_close_waits_for_inflight_operation_before_releasing_owned_clients() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    events: list[str] = []

    class BlockingStore(BorrowedStore):
        async def count(self, **filters: object) -> int:
            started.set()
            await release.wait()
            return await super().count(**filters)

    store = BlockingStore()
    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2, store=store)
        .with_owned_resource(OwnedResource("owned", events))
        .build()
    )
    operation = asyncio.create_task(rag.count())
    await asyncio.wait_for(started.wait(), timeout=3)
    closing = asyncio.create_task(rag.aclose())
    await asyncio.sleep(0)
    assert events == []
    with pytest.raises(RuntimeError, match="closed"):
        await rag.count()
    release.set()
    assert await asyncio.wait_for(operation, timeout=3) == 0
    await asyncio.wait_for(closing, timeout=3)
    assert events == ["owned"]
