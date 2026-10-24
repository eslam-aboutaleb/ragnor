"""
Tests for the ragnor job drainer loop.

The drainer is parameterized by a store and a processor,
so these tests drive ``drain_forever`` with in-memory
doubles instead of monkeypatching module attributes.
Host wiring (settings, store, processor) belongs to the
host application's own test suite.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from unittest.mock import patch

import pytest

from ragnor.jobs import drainer as drainer_module
from ragnor.jobs.drainer import (
    _install_signal_handlers,
    drain_forever,
    main,
)
from tests.job_doubles import FakeJobProcessor, FakeJobStore


@pytest.mark.asyncio
async def test_drain_stops_when_event_is_set() -> None:
    """The loop exits promptly once the stop event is set."""
    store = FakeJobStore()
    store.add()
    stop = asyncio.Event()
    processor = FakeJobProcessor()
    original_process = processor.process

    async def _stop_after_first(job) -> None:
        await original_process(job)
        stop.set()

    processor.process = _stop_after_first  # type: ignore[method-assign]
    await drain_forever(store, processor, interval=0.05, batch_size=2, worker_id="w", stop=stop)

    assert len(processor.processed) == 1
    assert store.claim_limits == [2]


@pytest.mark.asyncio
async def test_drain_passes_the_batch_size_through() -> None:
    """The configured batch size reaches the store on every pass."""
    store = FakeJobStore()
    for _ in range(6):
        store.add()
    stop = asyncio.Event()
    seen: list[int] = []

    class _CountingProcessor:
        async def process(self, job) -> None:
            seen.append(store.claim_limits[-1])
            if len(seen) >= 6:
                stop.set()

    await drain_forever(
        store,
        _CountingProcessor(),
        interval=0.05,
        batch_size=2,
        worker_id="w",
        stop=stop,
    )

    assert store.claim_limits == [2, 2, 2]


@pytest.mark.asyncio
async def test_drain_passes_its_worker_id_to_the_processor() -> None:
    """The configured worker identity is stamped onto every claimed batch."""
    store = FakeJobStore()
    job = store.add()
    stop = asyncio.Event()

    class _StopProcessor:
        async def process(self, job) -> None:
            stop.set()

    await drain_forever(
        store,
        _StopProcessor(),
        interval=0.05,
        batch_size=2,
        worker_id="worker-1",
        stop=stop,
    )

    assert store.claim_worker_ids == ["worker-1"]
    assert job.locked_by == "worker-1"
    assert job.locked_at is not None


@pytest.mark.asyncio
async def test_drain_reclaims_stale_jobs_before_claiming() -> None:
    """Every pass reclaims abandoned locks before claiming new work."""
    stop = asyncio.Event()

    class _StopAfterTwoClaimsStore(FakeJobStore):
        async def claim_pending(self, limit: int, worker_id: str):
            result = await super().claim_pending(limit, worker_id)
            if len(self.claim_limits) >= 2:
                stop.set()
            return result

    flaky = _StopAfterTwoClaimsStore()
    await drain_forever(
        flaky,
        FakeJobProcessor(),
        interval=0.05,
        batch_size=2,
        worker_id="w",
        stop=stop,
    )

    # One reclaim per pass, each before the pass's claim.
    assert len(flaky.reclaim_calls) == 2
    assert len(flaky.claim_limits) == 2


@pytest.mark.asyncio
async def test_drain_survives_a_failing_pass() -> None:
    """A raising pass is logged and retried; it does not terminate the worker."""

    class _FlakyStore(FakeJobStore):
        """First claim raises; the worker must survive and retry."""

        def __init__(self, stop: asyncio.Event) -> None:
            super().__init__()
            self._stop = stop
            self.calls = 0

        async def claim_pending(self, limit: int, worker_id: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient embedding API failure")
            if self.calls >= 2:
                self._stop.set()
            return await super().claim_pending(limit, worker_id)

    stop = asyncio.Event()
    store = _FlakyStore(stop)

    await drain_forever(
        store,
        FakeJobProcessor(),
        interval=0.05,
        batch_size=2,
        worker_id="w",
        stop=stop,
    )

    # The first pass raised, the worker survived and retried.
    assert store.calls == 2


@pytest.mark.asyncio
async def test_drain_keeps_draining_while_work_remains() -> None:
    """A pass that processes jobs is followed immediately by another, without sleeping."""

    class _RefillingStore(FakeJobStore):
        """New work arrives on every pass, so the drainer never idles."""

        async def claim_pending(self, limit: int, worker_id: str):
            self.add()
            return await super().claim_pending(limit, worker_id)

    store = _RefillingStore()
    stop = asyncio.Event()

    class _StopAfterFourProcessor:
        def __init__(self) -> None:
            self.calls = 0

        async def process(self, job) -> None:
            self.calls += 1
            if self.calls >= 4:
                stop.set()

    processor = _StopAfterFourProcessor()
    await drain_forever(
        store,
        processor,
        interval=5.0,  # would time out if the loop slept between passes
        batch_size=10,
        worker_id="w",
        stop=stop,
    )

    assert processor.calls == 4


@pytest.mark.asyncio
async def test_drain_logs_when_stale_jobs_are_reclaimed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pass that reclaims abandoned locks logs the reclaimed count."""

    class _TwoStaleJobsStore(FakeJobStore):
        async def reclaim_stale(self, stale_after_seconds: float | None = None) -> int:
            await super().reclaim_stale(stale_after_seconds)
            return 2

    store = _TwoStaleJobsStore()
    store.add()
    stop = asyncio.Event()

    class _StopProcessor:
        async def process(self, job) -> None:
            stop.set()

    with caplog.at_level(logging.INFO, logger="ragnor.jobs.drainer"):
        await drain_forever(
            store,
            _StopProcessor(),
            interval=0.05,
            batch_size=2,
            worker_id="w",
            stop=stop,
        )

    assert "Reclaimed 2 stale job(s)" in caplog.text


@pytest.mark.asyncio
async def test_drain_propagates_cancellation() -> None:
    """A cancelled pass re-raises CancelledError instead of being swallowed."""

    class _CancellingStore(FakeJobStore):
        async def claim_pending(self, limit: int, worker_id: str):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await drain_forever(
            _CancellingStore(),
            FakeJobProcessor(),
            interval=0.05,
            batch_size=2,
            worker_id="w",
        )


def test_install_signal_handlers_registers_both_signals() -> None:
    """SIGINT and SIGTERM are wired to the stop event on the running loop."""
    stop = asyncio.Event()
    registered: list[object] = []

    class _FakeLoop:
        def add_signal_handler(self, sig, callback):  # noqa: ANN001, ANN202
            registered.append(sig)

    with patch.object(
        drainer_module.asyncio,
        "get_running_loop",
        return_value=_FakeLoop(),
    ):
        _install_signal_handlers(stop)

    assert set(registered) == {signal.SIGINT, signal.SIGTERM}


def test_install_signal_handlers_tolerates_unsupported_loops() -> None:
    """Loops without signal support (e.g. Windows) do not crash the worker."""

    class _UnsupportedLoop:
        def add_signal_handler(self, sig, callback):  # noqa: ANN001, ANN202
            raise NotImplementedError

    with patch.object(
        drainer_module.asyncio,
        "get_running_loop",
        return_value=_UnsupportedLoop(),
    ):
        _install_signal_handlers(asyncio.Event())


def test_main_returns_zero_after_clean_shutdown() -> None:
    """The console entry point installs logging and returns 0 on exit."""
    store = FakeJobStore()
    processor = FakeJobProcessor()
    with patch("ragnor.jobs.drainer.asyncio.run", return_value=None) as mock_run:
        assert main(store, processor, interval=1.0, batch_size=2, worker_id="w") == 0
    mock_run.assert_called_once()
    mock_run.call_args.args[0].close()  # Patched asyncio.run never consumes the coroutine.


def test_main_swallows_keyboard_interrupt() -> None:
    """Ctrl-C during shutdown is expected and still exits 0."""
    store = FakeJobStore()
    processor = FakeJobProcessor()
    with patch(
        "ragnor.jobs.drainer.asyncio.run",
        side_effect=KeyboardInterrupt,
    ) as mock_run:
        assert main(store, processor, interval=1.0, batch_size=2, worker_id="w") == 0
    mock_run.call_args.args[0].close()  # The interrupting mock never awaits it.
