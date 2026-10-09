"""Coverage tests for the ragit job outbox enqueue path.

Includes the outbox invariant test: a job enqueued with a
caller session is not committed by the enqueue call.
"""

from __future__ import annotations

import asyncio

import pytest

from ragit.jobs import drainer
from ragit.jobs.outbox import enqueue_job
from tests.job_doubles import FakeJobStore, RecordingSession


class TestEnqueueJob:
    @pytest.mark.asyncio
    async def test_enqueue_with_caller_session_does_not_commit(self):
        """Passing a session must only flush, never commit."""
        store = FakeJobStore()
        session = RecordingSession()
        payload = {"claim_id": "CLM-9999", "description": "Test"}

        await enqueue_job(store, payload, session=session)

        assert session.commits == 0, "enqueue committed a caller-owned session"
        assert session.flushes == 1
        assert store.enqueued_payloads == [payload]

    @pytest.mark.asyncio
    async def test_enqueue_without_session_commits(self):
        """With no session the enqueue is self-contained and durable."""
        store = FakeJobStore()
        payload = {"claim_id": "CLM-9999", "description": "Test"}

        await enqueue_job(store, payload)

        assert store.enqueued_payloads == [payload]
        assert store.committed_payloads == [payload]

    @pytest.mark.asyncio
    async def test_enqueue_passes_the_session_through_to_the_store(self):
        """The caller's session reaches the store's enqueue verbatim."""
        store = FakeJobStore()
        session = RecordingSession()

        await enqueue_job(store, {"claim_id": "CLM-1"}, session=session)

        assert session.flushes == 1
        assert session.commits == 0


class TestDrainerRun:
    @pytest.mark.asyncio
    async def test_run_installs_signal_handlers_and_drains(self, monkeypatch):
        installed: list[asyncio.Event] = []
        drained: dict[str, object] = {}

        def _record_stop(stop: asyncio.Event) -> None:
            installed.append(stop)

        async def _drain_stub(*args, **kwargs):
            drained["stop"] = args[-1] if args else kwargs.get("stop")

        monkeypatch.setattr(drainer, "_install_signal_handlers", _record_stop)
        monkeypatch.setattr(drainer, "drain_forever", _drain_stub)

        await drainer._run(
            store=object(),
            processor=object(),
            interval=0.0,
            batch_size=1,
            worker_id="test-worker",
        )

        assert len(installed) == 1
        assert isinstance(installed[0], asyncio.Event)
        assert drained["stop"] is installed[0]
