"""Pluggable synchronous Python facade with safe event-loop and resource lifecycle."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from ragnor.core.exceptions import ConfigurationError, OperationTimeoutError, RagnorError

if TYPE_CHECKING:
    from ragnor.core.types import Document, SearchResult

T = TypeVar("T")


@runtime_checkable
class AsyncRagnorOperations(Protocol):
    """Minimal asynchronous facade port wrapped by :class:`SyncRagnor`."""

    async def ingest(
        self,
        documents: list[Document],
        *,
        purge_on_empty: bool = False,
        lock_key: str | None = None,
    ) -> int:
        """Ingest documents and return the stored chunk count."""
        raise NotImplementedError

    async def search(
        self,
        query: str,
        n_results: int = 5,
        **filters: Any,
    ) -> list[SearchResult]:
        """Search for documents matching ``query``."""
        raise NotImplementedError

    async def count(self, **filters: Any) -> int:
        """Count documents in the store."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Close owned async resources."""
        raise NotImplementedError


class AsyncRagnorBuilder(Protocol):
    """Construction port for host builders without coupling to the SDK builder."""

    async def build(self) -> AsyncRagnorOperations:
        """Construct the asynchronous facade on the managed event loop."""
        raise NotImplementedError


def _reject_running_event_loop() -> None:
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


class SyncRagnor:
    """Synchronous facade over an :class:`AsyncRagnorOperations` implementation.

    Manages a dedicated background thread running a persistent event loop so
    blocking scripts and CLI applications can interact with Ragnor without
    invoking ``asyncio.run()`` on each call.
    """

    def __init__(
        self,
        ragnor: AsyncRagnorOperations,
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
    def from_async_factory(
        cls,
        factory: Callable[[], Awaitable[AsyncRagnorOperations]],
        *,
        timeout: float | None = 60.0,
    ) -> SyncRagnor:
        """Construct a SyncRagnor instance from an async factory on a managed loop."""
        _reject_running_event_loop()

        loop = asyncio.new_event_loop()
        loop_ready = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_runner, daemon=True, name="SyncRagnorWorker")
        thread.start()
        loop_ready.wait()

        async def _init_client() -> AsyncRagnorOperations:
            return await factory()

        future = asyncio.run_coroutine_threadsafe(_init_client(), loop)
        try:
            ragnor_instance = future.result(timeout=timeout)
        except Exception as exc:
            future.cancel()
            with suppress(Exception):
                clean_fut = asyncio.run_coroutine_threadsafe(
                    cls._drain_loop_tasks(loop),
                    loop,
                )
                clean_fut.result(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            if not loop.is_closed():
                loop.close()
            if isinstance(exc, TimeoutError) and not isinstance(exc, OperationTimeoutError):
                raise OperationTimeoutError(
                    f"SyncRagnor initialization timed out after {timeout}s"
                ) from exc
            if isinstance(exc, RagnorError):
                raise
            raise ConfigurationError(
                f"Failed to build Ragnor instance ({type(exc).__name__})"
            ) from exc

        return cls(ragnor_instance, loop=loop, thread=thread, default_timeout=timeout)

    @classmethod
    def from_builder(
        cls,
        builder: AsyncRagnorBuilder | Callable[[], AsyncRagnorBuilder],
        *,
        timeout: float | None = 60.0,
    ) -> SyncRagnor:
        """Construct a SyncRagnor instance from a RagnorBuilder or builder factory."""

        async def _build() -> AsyncRagnorOperations:
            resolved_builder = builder() if callable(builder) else builder
            return await resolved_builder.build()

        return cls.from_async_factory(_build, timeout=timeout)

    @staticmethod
    async def _drain_loop_tasks(loop: asyncio.AbstractEventLoop) -> None:
        current = asyncio.current_task(loop)
        pending = [t for t in asyncio.all_tasks(loop) if t is not current and not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await loop.shutdown_asyncgens()
        await loop.shutdown_default_executor()

    def _ensure_caller_safe(self, *, allow_closed: bool = False) -> None:
        if self._thread is not None and threading.current_thread() is self._thread:
            raise ConfigurationError(
                "SyncRagnor methods cannot be called reentrantly from its internal "
                "event-loop worker thread.",
                code="ERR_LOOP_CONFLICT",
            )
        _reject_running_event_loop()
        if not allow_closed:
            with self._lock:
                if self._closed:
                    raise ConfigurationError(
                        "SyncRagnor client is closed", code="ERR_CLIENT_CLOSED"
                    )

    def _run_coroutine(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: float | None = None,
    ) -> T:
        with self._lock:
            if self._closed or self._loop.is_closed():
                coro.close()
                raise ConfigurationError("SyncRagnor client is closed", code="ERR_CLIENT_CLOSED")
            effective_timeout = self._default_timeout if timeout is None else timeout
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            except RuntimeError as exc:
                coro.close()
                raise ConfigurationError(
                    "SyncRagnor event loop is not running", code="ERR_CLIENT_CLOSED"
                ) from exc

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
            raise RagnorError(f"Sync operation failed ({type(exc).__name__})") from exc

    def ingest(
        self,
        documents: list[Document],
        *,
        purge_on_empty: bool = False,
        lock_key: str | None = None,
        timeout: float | None = None,
    ) -> int:
        """Synchronously ingest documents and return the stored chunk count."""
        self._ensure_caller_safe()
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
        self._ensure_caller_safe()
        return self._run_coroutine(
            self._ragnor.search(query, n_results=n_results, **filters),
            timeout=timeout,
        )

    def count(self, *, timeout: float | None = None, **filters: Any) -> int:
        """Synchronously count documents in the store."""
        self._ensure_caller_safe()
        return self._run_coroutine(
            self._ragnor.count(**filters),
            timeout=timeout,
        )

    def close(self, *, timeout: float | None = None) -> None:
        """Close the client and shut down the underlying event loop and thread."""
        self._ensure_caller_safe(allow_closed=True)
        with self._lock:
            if self._closed:
                return
            self._closed = True

        effective_timeout = self._default_timeout if timeout is None else timeout
        close_error: Exception | None = None

        async def _shutdown() -> None:
            nonlocal close_error
            aclose = getattr(self._ragnor, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception as exc:  # noqa: BLE001
                    close_error = exc
            await self._drain_loop_tasks(self._loop)

        try:
            if self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
                try:
                    future.result(timeout=effective_timeout)
                except TimeoutError as exc:
                    future.cancel()
                    close_error = OperationTimeoutError(
                        f"SyncRagnor shutdown timed out after {effective_timeout}s"
                    )
                    close_error.__cause__ = exc
        finally:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=effective_timeout)
            if not self._loop.is_closed():
                with suppress(RuntimeError):
                    self._loop.close()

        if close_error is not None:
            raise close_error

    def __enter__(self) -> SyncRagnor:
        self._ensure_caller_safe()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
