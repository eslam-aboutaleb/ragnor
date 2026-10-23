"""Outbox worker must not persist or log arbitrary provider exception text."""

from __future__ import annotations

import logging

import pytest

from ragnor.jobs.outbox import InMemoryEmbeddingJobStore, process_pending_jobs


class FailingProcessor:
    async def process(self, job):
        raise RuntimeError("private-doc-body API_KEY=secret-value")


@pytest.mark.asyncio
async def test_untrusted_processor_errors_are_redacted(caplog: pytest.LogCaptureFixture) -> None:
    store = InMemoryEmbeddingJobStore()
    await store.enqueue({"idempotency_key": "redaction", "scope": "demo"})
    with caplog.at_level(logging.ERROR):
        assert await process_pending_jobs(store, FailingProcessor(), worker_id="worker") == 0
    job = store.jobs[0]
    assert job.status == "failed"
    assert "RuntimeError" in (job.status_detail or "")
    assert "secret-value" not in (job.status_detail or "")
    assert "secret-value" not in caplog.text
