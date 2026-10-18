"""Unit tests for HealthReport, HealthState, ComponentHealth, and HealthChecker."""

from __future__ import annotations

import pytest

from ragnor.sync.manifest import InMemoryManifestStore, ManifestRecord, ManifestStatus
from ragnor.sync.state import InMemorySyncStateStore
from ragnor.telemetry.health import (
    ComponentHealth,
    HealthChecker,
    HealthReport,
    HealthState,
)


def test_health_state_rollup_precedence() -> None:
    """Unavailable beats degraded, degraded beats unknown, unknown beats ready."""
    report_ready = HealthReport(
        components=[
            ComponentHealth(name="c1", state=HealthState.READY, reason="ok"),
            ComponentHealth(name="c2", state=HealthState.READY, reason="ok"),
        ],
    )
    assert report_ready.state == HealthState.READY
    assert report_ready.is_healthy is True

    report_unknown = HealthReport(
        components=[
            ComponentHealth(name="c1", state=HealthState.READY, reason="ok"),
            ComponentHealth(name="c2", state=HealthState.UNKNOWN, reason="no data"),
        ],
    )
    assert report_unknown.state == HealthState.UNKNOWN
    assert report_unknown.is_healthy is False

    report_degraded = HealthReport(
        components=[
            ComponentHealth(name="c1", state=HealthState.READY, reason="ok"),
            ComponentHealth(name="c2", state=HealthState.DEGRADED, reason="high latency"),
            ComponentHealth(name="c3", state=HealthState.UNKNOWN, reason="no data"),
        ],
    )
    assert report_degraded.state == HealthState.DEGRADED
    assert report_degraded.is_healthy is False

    report_unavailable = HealthReport(
        components=[
            ComponentHealth(name="c1", state=HealthState.READY, reason="ok"),
            ComponentHealth(name="c2", state=HealthState.DEGRADED, reason="slow"),
            ComponentHealth(name="c3", state=HealthState.UNAVAILABLE, reason="db down"),
        ],
    )
    assert report_unavailable.state == HealthState.UNAVAILABLE
    assert report_unavailable.is_healthy is False


class _FakeSettings:
    def __init__(self, embedding_model: str = "text-embedding-3-small") -> None:
        self.embedding_model = embedding_model


@pytest.mark.asyncio
async def test_health_checker_configuration_compatibility() -> None:
    """Valid config reports READY; invalid/unknown model reports DEGRADED."""
    settings = _FakeSettings()
    checker = HealthChecker(settings=settings)
    comp = checker.check_configuration()
    assert comp.state == HealthState.READY
    assert "compatible" in comp.reason

    # Invalid embedding model
    bad_settings = _FakeSettings(embedding_model="unsupported-model-999")
    bad_checker = HealthChecker(settings=bad_settings)
    bad_comp = bad_checker.check_configuration()
    assert bad_comp.state == HealthState.DEGRADED
    assert "unknown embedding model" in bad_comp.reason.lower()


@pytest.mark.asyncio
async def test_health_checker_counts_deliberate_empty_vs_unexpected_zero() -> None:
    """Deliberately empty collection is READY; unexpected zero chunks is DEGRADED."""
    manifest_store = InMemoryManifestStore()
    settings = _FakeSettings()

    # Case 1: allow_empty=True -> READY
    checker_deliberate = HealthChecker(
        settings=settings,
        manifest_store=manifest_store,
        allow_empty=True,
    )
    comp1 = await checker_deliberate.check_counts(scope="test-scope")
    assert comp1.state == HealthState.READY
    assert comp1.details.get("deliberate_empty") is True

    # Case 2: allow_empty=False and 0 chunks -> DEGRADED
    checker_unexpected = HealthChecker(
        settings=settings,
        manifest_store=manifest_store,
        allow_empty=False,
    )
    comp2 = await checker_unexpected.check_counts(scope="test-scope")
    assert comp2.state == HealthState.DEGRADED
    assert "unexpectedly indexing zero chunks" in comp2.reason

    # Case 3: active chunks exist -> READY
    await manifest_store.put(
        ManifestRecord(
            scope="test-scope",
            document_id="doc1",
            observed_revision="r1",
            content_hash="h1",
            extractor_fingerprint="ef1",
            chunker_fingerprint="cf1",
            embedding_fingerprint="emf1",
            active_generation=1,
            status=ManifestStatus.ACTIVE,
            chunk_ids=["c1", "c2"],
        )
    )
    comp3 = await checker_unexpected.check_counts(scope="test-scope")
    assert comp3.state == HealthState.READY
    assert comp3.details.get("active_documents") == 1
    assert comp3.details.get("active_chunks") == 2


@pytest.mark.asyncio
async def test_health_checker_sync_status() -> None:
    """Sync status differentiates complete, incomplete, and missing syncs."""
    state_store = InMemorySyncStateStore()
    checker = HealthChecker(state_store=state_store)

    # No checkpoint -> UNKNOWN
    comp0 = await checker.check_sync_status(scope="scope1")
    assert comp0.state == HealthState.UNKNOWN

    # Checkpoint with authoritative complete -> READY
    token = await state_store.acquire_lease("scope1", "w1")
    assert token is not None
    await state_store.save_checkpoint(
        "scope1",
        {"phase": "completed", "authoritative": True},
        token,
    )
    comp1 = await checker.check_sync_status(scope="scope1")
    assert comp1.state == HealthState.READY

    # Checkpoint incomplete / non-authoritative -> DEGRADED
    await state_store.save_checkpoint(
        "scope1",
        {"phase": "partial_scan", "authoritative": False},
        token,
    )
    comp2 = await checker.check_sync_status(scope="scope1")
    assert comp2.state == HealthState.DEGRADED
    assert "incomplete" in comp2.reason.lower()


@pytest.mark.asyncio
async def test_health_checker_jobs_dead_letter_detection() -> None:
    """Dead letter jobs mark component as DEGRADED."""
    from ragnor.jobs.outbox import InMemoryEmbeddingJobStore

    job_store = InMemoryEmbeddingJobStore()
    checker = HealthChecker(job_store=job_store)

    comp_ok = await checker.check_jobs(scope="scope1")
    assert comp_ok.state == HealthState.READY

    # Enqueue a dead-letter job
    await job_store.enqueue(
        {
            "idempotency_key": "k1",
            "scope": "scope1",
            "status": "dead_letter",
            "retries": 5,
        }
    )
    comp_dead = await checker.check_jobs(scope="scope1")
    assert comp_dead.state == HealthState.DEGRADED
    assert "dead-letter" in comp_dead.reason


@pytest.mark.asyncio
async def test_health_checker_evaluation_results() -> None:
    """Evaluation results are reported accurately when provided."""
    checker = HealthChecker()

    # No eval -> UNKNOWN
    comp_none = checker.check_evaluations()
    assert comp_none.state == HealthState.UNKNOWN

    # Passing eval -> READY
    comp_pass = checker.check_evaluations(
        last_eval_results={"score": 0.92, "threshold": 0.85, "passed": True}
    )
    assert comp_pass.state == HealthState.READY

    # Failing eval -> DEGRADED
    comp_fail = checker.check_evaluations(
        last_eval_results={"score": 0.70, "threshold": 0.85, "passed": False}
    )
    assert comp_fail.state == HealthState.DEGRADED
    assert "failed or below threshold" in comp_fail.reason


@pytest.mark.asyncio
async def test_health_checker_expensive_check_bounded() -> None:
    """Expensive consistency check runs only when explicitly requested."""
    manifest_store = InMemoryManifestStore()
    checker = HealthChecker(manifest_store=manifest_store)

    # include_expensive=False: expensive check is not included
    report_fast = await checker.check_all(scope="scope1", include_expensive=False)
    assert "expensive_consistency" not in report_fast.components

    # include_expensive=True: expensive check runs and is bounded
    report_full = await checker.check_all(
        scope="scope1",
        include_expensive=True,
        expensive_limit=50,
    )
    assert "expensive_consistency" in report_full.components
