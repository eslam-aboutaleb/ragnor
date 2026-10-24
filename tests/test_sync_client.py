"""SDK-05 (#25): Pluggable synchronous Python facade with safe event loop and lifecycle."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from ragnor.builder import RagnorBuilder
from ragnor.exceptions import ConfigurationError, OperationTimeoutError
from ragnor.sync_client import AsyncRagnorOperations, SyncRagnor
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
        with SyncRagnor.from_builder(_make_builder) as client:
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
        with SyncRagnor.from_builder(builder, timeout=0.05) as client:
            with pytest.raises(OperationTimeoutError):
                client.ingest([Document(id="slow", text="slow text")], timeout=0.05)

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

    def test_builder_build_sync_and_protocol_conformance(self) -> None:
        with _make_builder().build_sync() as client:
            assert isinstance(client._ragnor, AsyncRagnorOperations)
            client.ingest([Document(id="p1", text="protocol check")])
            assert client.count() >= 1

    def test_construction_failure_cleans_up_worker_thread_and_loop(self) -> None:
        threads_before = {
            t.ident for t in threading.enumerate() if t.name.startswith("ragnor-sync-")
        }

        async def failing_factory() -> AsyncRagnorOperations:
            raise ConfigurationError("factory initialization failed")

        with pytest.raises(ConfigurationError, match="factory initialization failed"):
            SyncRagnor.from_async_factory(failing_factory)

        threads_after = {
            t.ident for t in threading.enumerate() if t.name.startswith("ragnor-sync-")
        }
        assert threads_after == threads_before

    def test_loop_affinity_across_build_ingest_search_count_aclose(self) -> None:
        observed_loops: list[int] = []
        observed_threads: list[int] = []

        class LoopTrackingEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                observed_loops.append(id(asyncio.get_running_loop()))
                observed_threads.append(threading.get_ident())
                return [[0.5] * 8 for _ in texts]

        async def factory() -> AsyncRagnorOperations:
            observed_loops.append(id(asyncio.get_running_loop()))
            observed_threads.append(threading.get_ident())
            return await RagnorBuilder.local(
                embedder=LoopTrackingEmbedder(),
                embedding_dim=8,
            ).build()

        with SyncRagnor.from_async_factory(factory) as client:
            client.ingest([Document(id="loop-1", text="loop affinity test")])
            client.search("loop affinity")
            client.count()

        assert len(observed_loops) >= 3
        assert len(set(observed_loops)) == 1
        assert len(set(observed_threads)) == 1
        assert observed_threads[0] != threading.get_ident()

    def test_worker_thread_reentrant_call_rejected_without_deadlock(self) -> None:
        holder: dict[str, SyncRagnor] = {}

        class ReentrantEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                # Calling the sync client from inside the worker thread must fail immediately
                holder["client"].count()
                return [[0.1] * 8 for _ in texts]

        builder = RagnorBuilder.local(embedder=ReentrantEmbedder(), embedding_dim=8)
        with SyncRagnor.from_builder(builder, timeout=2.0) as client:
            holder["client"] = client
            with pytest.raises(ConfigurationError) as exc_info:
                client.ingest([Document(id="reentrant", text="trigger self call")])
            assert exc_info.value.code == "ERR_LOOP_CONFLICT"

    def test_concurrent_external_caller_threads_synchronized_with_barrier(self) -> None:
        with SyncRagnor.from_builder(_make_builder) as client:
            client.ingest(
                [
                    Document(id="c1", text="Concurrent search document alpha"),
                    Document(id="c2", text="Concurrent search document beta"),
                ]
            )
            barrier = threading.Barrier(2)
            results_by_thread: list[int] = []
            errors: list[Exception] = []

            def worker(query: str) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    hits = client.search(query, n_results=2)
                    total = client.count()
                    results_by_thread.append(len(hits) + total)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=worker, args=("alpha",))
            t2 = threading.Thread(target=worker, args=("beta",))
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            assert not errors
            assert len(results_by_thread) == 2
            assert all(val >= 3 for val in results_by_thread)

    def test_real_aiosqlite_connection_bound_to_worker_loop(self, tmp_path: Path) -> None:
        db_file = tmp_path / "sync_loop_affinity.sqlite"
        conn_holder: dict[str, aiosqlite.Connection] = {}

        class SqliteBackedEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                conn = conn_holder["conn"]
                await conn.execute(
                    "INSERT INTO embed_log (text_len) VALUES (?)",
                    (len(texts[0]),),
                )
                await conn.commit()
                async with conn.execute("SELECT COUNT(*) FROM embed_log") as cursor:
                    row = await cursor.fetchone()
                    assert row is not None and row[0] >= 1
                return [[0.25] * 8 for _ in texts]

        class _CloseableSqliteConn:
            def __init__(self, conn: aiosqlite.Connection) -> None:
                self._conn = conn

            async def aclose(self) -> None:
                await self._conn.close()

        async def factory() -> AsyncRagnorOperations:
            conn = await aiosqlite.connect(str(db_file))
            await conn.execute("CREATE TABLE embed_log (id INTEGER PRIMARY KEY, text_len INTEGER)")
            await conn.commit()
            conn_holder["conn"] = conn
            ragnor = await (
                RagnorBuilder.local(
                    embedder=SqliteBackedEmbedder(),
                    embedding_dim=8,
                )
                .with_owned_resource(_CloseableSqliteConn(conn))
                .build()
            )
            return ragnor

        with SyncRagnor.from_async_factory(factory) as client:
            client.ingest([Document(id="sql-1", text="aiosqlite loop affinity")])
            hits = client.search("aiosqlite")
            assert len(hits) == 1

    def test_standalone_subprocess_cli_exits_cleanly(self) -> None:
        script = (
            "from ragnor import Document, RagnorBuilder\n"
            "async def embed(texts: list[str]) -> list[list[float]]:\n"
            "    return [[0.5, 0.5, 0.5, 0.5] for _ in texts]\n"
            "with RagnorBuilder.local(embedder=embed, embedding_dim=4).build_sync() as client:\n"
            "    client.ingest([Document(id='cli-1', text='CLI sync execution')])\n"
            "    hits = client.search('CLI sync')\n"
            "    assert len(hits) == 1\n"
        )
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
