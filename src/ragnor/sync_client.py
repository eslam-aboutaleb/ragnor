"""Pluggable synchronous Python facade with safe event-loop and resource lifecycle."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ragnor.exceptions import ConfigurationError, OperationTimeoutError, RagnorError

if TYPE_CHECKING:
    from ragnor.builder import Ragnor, RagnorBuilder
    from ragnor.types import Document, SearchResult


class SyncRagnor:
    """Synchronous facade over the asynchronous Ragnor instance.

    Manages a dedicated background thread running an event loop so blocking
    scripts and CLI applications can interact with Ragnor without invoking
    asyncio.run() on each call.

    Pluggability and Clean Architecture are preserved: all custom stores,
    chunkers, embedders, and authorizers configured in the builder are
    injected seamlessly.
    """

    def __init__(
        self,
        ragnor: Ragnor,
        *,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread | None = None,
        default_timeout: float | None = 60.0,
    ) -> None:
        self._ragnor = ragnor
        self._loop = loop
        self._thread = thread
        self._default_timeout = default_timeout
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def from_builder(
        cls,
        builder: RagnorBuilder | Callable[[], RagnorBuilder],
        *,
        timeout: float | None = 60.0,
    ) -> SyncRagnor:
        """Construct a SyncRagnor instance on a managed background event loop.

        Args:
            builder: A configured RagnorBuilder instance or a zero-arg factory
                returning one.
            timeout: Default timeout in seconds for operations.

        Returns:
            A ready-to-use SyncRagnor instance.

        Raises:
            ConfigurationError: If called from within an already running event loop.
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            raise ConfigurationError(
                "SyncRagnor cannot be used inside an active asyncio event loop. "
                "Use the async Ragnor API directly with 'await ragnor.search(...)'.",
                code="ERR_LOOP_CONFLICT",
            )

        loop = asyncio.new_event_loop()
        loop_ready = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_runner, daemon=True, name="SyncRagnorWorker")
        thread.start()
        loop_ready.wait()

        async def _init_ragnor() -> Ragnor:
            resolved_builder = builder() if callable(builder) else builder
            return await resolved_builder.build()

        future = asyncio.run_coroutine_threadsafe(_init_ragnor(), loop)
        try:
            ragnor_instance = future.result(timeout=timeout)
        except Exception as exc:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()
            if isinstance(exc, (RagnorError, TimeoutError)):
                raise
            raise ConfigurationError(f"Failed to build Ragnor instance: {exc}") from exc

        return cls(ragnor_instance, loop=loop, thread=thread, default_timeout=timeout)

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise ConfigurationError("SyncRagnor client is closed", code="ERR_CLIENT_CLOSED")

    def _run_coroutine(self, coro: Any, timeout: float | None = None) -> Any:
        with self._lock:
            if self._closed:
                raise ConfigurationError("SyncRagnor client is closed", code="ERR_CLIENT_CLOSED")

            effective_timeout = self._default_timeout if timeout is None else timeout
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            try:
                return future.result(timeout=effective_timeout)
            except TimeoutError as exc:
                future.cancel()
                raise OperationTimeoutError(
                    f"SyncRagnor operation timed out after {effective_timeout}s"
                ) from exc
            except (RagnorError, RuntimeError, ValueError):
                raise
            except Exception as exc:
                raise RagnorError(f"Sync operation failed: {exc}") from exc

    def ingest(
        self,
        documents: list[Document],
        *,
        purge_on_empty: bool = False,
        lock_key: str | None = None,
        timeout: float | None = None,
    ) -> int:
        """Synchronously ingest documents and return the stored chunk count."""
        self._ensure_open()
        return self._run_coroutine(
            self._ragnor.ingest(documents, purge_on_empty=purge_on_empty, lock_key=lock_key),
            timeout=timeout,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        *,
        timeout: float | None = None,
        **filters: Any,
    ) -> list[SearchResult]:
        """Synchronously search for documents matching query."""
        self._ensure_open()
        return self._run_coroutine(
            self._ragnor.search(query, n_results=n_results, **filters),
            timeout=timeout,
        )

    def count(self, *, timeout: float | None = None, **filters: Any) -> int:
        """Synchronously count documents in the store."""
        self._ensure_open()
        return self._run_coroutine(
            self._ragnor.count(**filters),
            timeout=timeout,
        )

    def close(self, *, timeout: float | None = None) -> None:
        """Close the client and shut down the underlying event loop and thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        effective_timeout = self._default_timeout if timeout is None else timeout

        try:
            with suppress(Exception):
                future = asyncio.run_coroutine_threadsafe(self._ragnor.aclose(), self._loop)
                future.result(timeout=effective_timeout)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=effective_timeout)
            if not self._loop.is_closed():
                self._loop.close()

    def __enter__(self) -> SyncRagnor:
        if self._closed:
            raise ConfigurationError("SyncRagnor client is closed", code="ERR_CLIENT_CLOSED")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
