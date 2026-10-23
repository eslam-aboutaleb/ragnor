"""SDK-05 (#25): Pluggable synchronous Python facade with safe event loop and lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.exceptions import ConfigurationError, OperationTimeoutError
from ragnor.sync_client import SyncRagnor
from ragnor.types import Chunk, Document, SearchResult


class SyncDemoEmbedder:
    """Deterministic synchronous-compatible test embedder."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10)] * self.dim for t in texts]


def _make_builder() -> RagnorBuilder:
    return RagnorBuilder.local(
        embedder=SyncDemoEmbedder(8),
        embedding_dim=8,
    )


class TestSyncRagnorFacade:
    """Test blocking SDK workflow, loop management, and resource lifecycle."""

    def test_sync_workflow_ingest_search_count(self) -> None:
        client = SyncRagnor.from_builder(_make_builder)
        try:
            docs = [
                Document(id="doc1", text="Refund policy allows 30 days return."),
                Document(id="doc2", text="Shipping guidelines and transit times."),
            ]
            count = client.ingest(docs)
            assert count >= 2

            results = client.search("Refund policy", n_results=5)
            assert isinstance(results, list)
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert any("Refund" in r.document for r in results)

            total = client.count()
            assert total >= 2
        finally:
            client.close()

    def test_context_manager_lifecycle(self) -> None:
        with SyncRagnor.from_builder(_make_builder) as client:
            docs = [Document(id="doc-ctx", text="Context manager text")]
            count = client.ingest(docs)
            assert count >= 1

        # Calling after context exit raises ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            client.search("test")
        assert exc_info.value.code == "ERR_CLIENT_CLOSED"

    def test_idempotent_close(self) -> None:
        client = SyncRagnor.from_builder(_make_builder)
        client.close()
        # Second and third close must be safe no-ops
        client.close()
        client.close()

    def test_event_loop_reuse_on_single_worker_thread(self) -> None:
        client = SyncRagnor.from_builder(_make_builder)
        try:
            worker_thread = client._thread
            worker_loop = client._loop
            assert worker_thread is not None
            assert worker_thread.is_alive()
            assert worker_loop.is_running()

            client.count()
            client.count()

            # Thread and loop remain the same persistent instances
            assert client._thread is worker_thread
            assert client._loop is worker_loop
        finally:
            client.close()
            assert not worker_thread.is_alive()
            assert worker_loop.is_closed()

    def test_loop_safety_rejects_running_event_loop(self) -> None:
        async def try_running_sync_inside_async() -> None:
            SyncRagnor.from_builder(_make_builder)

        with pytest.raises(ConfigurationError) as exc_info:
            asyncio.run(try_running_sync_inside_async())

        assert exc_info.value.code == "ERR_LOOP_CONFLICT"
        assert "SyncRagnor cannot be used inside an active asyncio event loop" in str(
            exc_info.value
        )

    def test_operation_timeout_raises_typed_error(self) -> None:
        class SlowEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                await asyncio.sleep(0.5)
                return [[0.1] * 8 for _ in texts]

        builder = RagnorBuilder.local(embedder=SlowEmbedder(), embedding_dim=8)
        client = SyncRagnor.from_builder(builder, timeout=0.05)
        try:
            with pytest.raises(OperationTimeoutError):
                client.ingest([Document(id="slow", text="slow text")], timeout=0.05)
        finally:
            client.close()

    def test_pluggable_components_injection(self) -> None:
        records: list[str] = []

        class CustomChunker:
            config: dict[str, Any] = {"size": 100}
            version: str = "1.0"

            def chunk(self, text: str) -> list[Chunk]:
                records.append(text)
                return [Chunk(id="custom-1", text=text)]

        builder = RagnorBuilder.local(
            embedder=SyncDemoEmbedder(8),
            embedding_dim=8,
            chunker=CustomChunker(),
        )

        with SyncRagnor.from_builder(builder) as client:
            client.ingest([Document(id="custom", text="custom chunker text")])
            assert "custom chunker text" in records
