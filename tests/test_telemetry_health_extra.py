"""Targeted tests for HealthReport component checks in telemetry/health.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnor.telemetry.health import HealthChecker, HealthState


class _DummySettings:
    def __init__(self, embedding_model: str | None = None) -> None:
        self.embedding_model = embedding_model


@pytest.mark.asyncio
async def test_health_configuration_states() -> None:
    # No model specified
    settings = _DummySettings(embedding_model=None)
    checker = HealthChecker(settings=settings)  # type: ignore[arg-type]
    res = checker.check_configuration()
    assert res.state == HealthState.DEGRADED
    assert "Embedding model not specified" in res.reason

    # Unknown model
    settings = _DummySettings(embedding_model="unknown-custom-model")
    checker = HealthChecker(settings=settings)  # type: ignore[arg-type]
    res = checker.check_configuration()
    assert res.state == HealthState.DEGRADED
    assert "Unknown embedding model" in res.reason

    # Dimension mismatch with store
    settings = _DummySettings(embedding_model="text-embedding-3-small")  # 1536 dim
    mock_store = MagicMock()
    mock_store.embedding_dim = 768
    checker = HealthChecker(settings=settings, vector_store=mock_store)  # type: ignore[arg-type]
    res = checker.check_configuration()
    assert res.state == HealthState.UNAVAILABLE
    assert "Dimension mismatch" in res.reason

    # Compatible configuration
    mock_store.embedding_dim = 1536
    checker = HealthChecker(settings=settings, vector_store=mock_store)  # type: ignore[arg-type]
    res = checker.check_configuration()
    assert res.state == HealthState.READY
    assert "Configuration compatible" in res.reason


@pytest.mark.asyncio
async def test_health_backend_connectivity_states() -> None:
    # No store configured
    checker = HealthChecker()
    res = await checker.check_backend_connectivity()
    assert res.state == HealthState.UNKNOWN
    assert "No vector store configured" in res.reason

    # Ready probe
    mock_store = MagicMock()
    mock_store.count = AsyncMock(return_value=42)
    mock_store.capabilities = MagicMock()
    checker = HealthChecker(vector_store=mock_store)
    res = await checker.check_backend_connectivity()
    assert res.state == HealthState.READY

    # Timeout error
    mock_store.count = AsyncMock(side_effect=TimeoutError("Timed out"))
    checker = HealthChecker(vector_store=mock_store, timeout_seconds=0.01)
    res = await checker.check_backend_connectivity()
    assert res.state == HealthState.UNAVAILABLE
    assert "timed out" in res.reason

    # General exception
    mock_store.count = AsyncMock(side_effect=RuntimeError("Connection refused"))
    checker = HealthChecker(vector_store=mock_store)
    res = await checker.check_backend_connectivity()
    assert res.state == HealthState.UNAVAILABLE
    assert "Backend connectivity failure" in res.reason


@pytest.mark.asyncio
async def test_health_counts_states() -> None:
    # Neither store configured
    checker = HealthChecker()
    res = await checker.check_counts()
    assert res.state == HealthState.UNKNOWN

    # Vector store with positive counts
    mock_store = MagicMock()
    mock_store.count = AsyncMock(return_value=15)
    checker = HealthChecker(vector_store=mock_store)
    res = await checker.check_counts()
    assert res.state == HealthState.READY
    assert res.details["active_chunks"] == 15

    # Zero counts, deliberate empty allowed
    mock_store.count = AsyncMock(return_value=0)
    checker = HealthChecker(vector_store=mock_store, allow_empty=True)
    res = await checker.check_counts()
    assert res.state == HealthState.READY
    assert res.details.get("deliberate_empty") is True

    # Zero counts, unexpectedly empty
    checker = HealthChecker(vector_store=mock_store, allow_empty=False)
    res = await checker.check_counts()
    assert res.state == HealthState.DEGRADED
    assert "unexpectedly indexing zero chunks" in res.reason

    # Manifest store failure
    mock_manifest = MagicMock()
    mock_manifest.list_active = AsyncMock(side_effect=RuntimeError("Disk failure"))
    checker = HealthChecker(manifest_store=mock_manifest)
    res = await checker.check_counts()
    assert res.state == HealthState.UNAVAILABLE
    assert "Failed to query manifest records" in res.reason

    # Vector store failure
    mock_store.count = AsyncMock(side_effect=RuntimeError("Store corrupted"))
    checker = HealthChecker(vector_store=mock_store)
    res = await checker.check_counts()
    assert res.state == HealthState.UNAVAILABLE
    assert "Failed to query chunk counts" in res.reason
