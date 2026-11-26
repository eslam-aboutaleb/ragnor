"""SDK-02: qualification of deployed embedding/vector clients (offline slice).

These tests exercise the shipped ``ragnor.qualification`` harness against
behavioral clients using real asyncio scheduling, real deadlines and real
cancellation delivery -- no mocks of provider behavior. They qualify the
*harness*; qualifying a deployment's actual embedding/vector clients
remains a per-deployment responsibility (see docs/sdk/deployment.md).
"""

from __future__ import annotations

import asyncio

import pytest

from ragnor.qualification import (
    QualificationError,
    qualify_client_lifecycle,
    qualify_embedding_client,
)

SECRET = "ragnor_qualification_private_key_do_not_log_9911"


class DeployedEmbedder:
    """Well-behaved deployed embedding client with real latency."""

    def __init__(self, latency: float = 0.01, dimension: int = 3) -> None:
        self.latency = latency
        self.dimension = dimension
        self.close_calls = 0
        self._closed = False

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._closed:
            raise RuntimeError("client is disconnected")
        await asyncio.sleep(self.latency)
        return [[0.1 * (index + 1)] * self.dimension for index, _ in enumerate(texts)]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.close_calls += 1
        self._closed = True


class SlowEmbedder(DeployedEmbedder):
    def __init__(self) -> None:
        super().__init__(latency=30.0)


class CancellationSwallower(DeployedEmbedder):
    """Catches cancellation and returns anyway: the SDK cannot classify it."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._closed:
            raise RuntimeError("client is disconnected")
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            return [[0.0] * self.dimension for _ in texts]
        return [[0.0] * self.dimension for _ in texts]


class NonIdempotentCloseEmbedder(DeployedEmbedder):
    async def aclose(self) -> None:
        await super().aclose()
        if self.close_calls > 1:
            raise RuntimeError("close may only be called once")


class ReconnectingEmbedder(DeployedEmbedder):
    """Silently keeps serving calls after aclose(): shutdown is not terminal."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(self.latency)
        return [[0.1] * self.dimension for _ in texts]


class SecretLeakingEmbedder:
    """Raises provider payloads that contain credentials."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        raise RuntimeError(f"provider request failed api_key={SECRET}")

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)

    async def aclose(self) -> None:
        await asyncio.sleep(0)


class CallableWithoutCleanup:
    """Structural embedder with no cleanup hook at all."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0)
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class NotAnEmbedder:
    async def aclose(self) -> None:
        await asyncio.sleep(0)


async def test_well_behaved_deployed_client_passes_every_check() -> None:
    client = DeployedEmbedder()
    report = await qualify_embedding_client(client, expected_dimension=3)
    assert report.passed, [(c.name, c.detail) for c in report.checks]
    assert report.failed_checks == ()
    report.require_passed()
    assert client.close_calls == 2  # harness verified idempotence with a second call


async def test_report_names_the_client_type() -> None:
    report = await qualify_embedding_client(DeployedEmbedder())
    assert report.target == "DeployedEmbedder"
    assert [check.name for check in report.checks] == [
        "embedding_function_protocol",
        "startup_call_within_deadline",
        "batch_contract",
        "cancellation_propagates",
        "cleanup_contract",
        "disconnect_terminal",
    ]


async def test_startup_deadline_cancels_a_hung_client() -> None:
    report = await qualify_embedding_client(SlowEmbedder(), startup_timeout=0.05)
    assert not report.passed
    assert "startup_call_within_deadline" in report.failed_checks
    with pytest.raises(QualificationError) as excinfo:
        report.require_passed()
    assert "startup_call_within_deadline" in str(excinfo.value)
    assert excinfo.value.code == "ERR_QUALIFICATION_FAILED"
    assert excinfo.value.retryable is False


async def test_dimension_mismatch_fails_the_batch_contract() -> None:
    report = await qualify_embedding_client(DeployedEmbedder(dimension=3), expected_dimension=4)
    assert not report.passed
    assert "batch_contract" in report.failed_checks


async def test_swallowed_cancellation_is_a_qualification_failure() -> None:
    report = await qualify_embedding_client(CancellationSwallower(), startup_timeout=1.0)
    assert not report.passed
    assert "cancellation_propagates" in report.failed_checks
    detail = next(c.detail for c in report.checks if c.name == "cancellation_propagates")
    assert "returned normally" in detail


async def test_fast_completion_before_cancellation_delivery_still_passes() -> None:
    client = DeployedEmbedder(latency=0.0)
    report = await qualify_embedding_client(client)
    cancellation = next(c for c in report.checks if c.name == "cancellation_propagates")
    assert cancellation.passed


async def test_non_idempotent_aclose_fails_the_cleanup_contract() -> None:
    report = await qualify_embedding_client(NonIdempotentCloseEmbedder())
    assert not report.passed
    assert "cleanup_contract" in report.failed_checks


async def test_missing_cleanup_hook_fails_the_cleanup_contract() -> None:
    report = await qualify_embedding_client(CallableWithoutCleanup())
    assert not report.passed
    assert "cleanup_contract" in report.failed_checks
    detail = next(c.detail for c in report.checks if c.name == "cleanup_contract")
    assert "no aclose" in detail


async def test_calls_succeeding_after_disconnect_fail_disconnect_terminal() -> None:
    report = await qualify_embedding_client(ReconnectingEmbedder())
    assert not report.passed
    assert "disconnect_terminal" in report.failed_checks


async def test_provider_payloads_never_leak_credentials_through_reports() -> None:
    report = await qualify_embedding_client(SecretLeakingEmbedder(), startup_timeout=1.0)
    assert not report.passed
    assert "startup_call_within_deadline" in report.failed_checks
    serialized = " ".join(c.detail for c in report.checks)
    assert SECRET not in serialized
    assert "api_key=***" in serialized
    with pytest.raises(QualificationError) as excinfo:
        report.require_passed()
    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


async def test_non_embedder_fails_the_structural_check() -> None:
    report = await qualify_embedding_client(NotAnEmbedder(), startup_timeout=1.0)
    assert "embedding_function_protocol" in report.failed_checks


async def test_qualify_client_lifecycle_covers_vector_store_probes() -> None:
    class DeployedStore:
        def __init__(self) -> None:
            self._closed = False

        async def count(self, **kwargs: object) -> int:
            if self._closed:
                raise RuntimeError("store is disconnected")
            await asyncio.sleep(0)
            return 0

        async def aclose(self) -> None:
            await asyncio.sleep(0)
            self._closed = True

    store = DeployedStore()
    report = await qualify_client_lifecycle(store, store.count, startup_timeout=1.0)
    assert report.passed, [(c.name, c.detail) for c in report.checks]
    assert [check.name for check in report.checks] == [
        "startup_probe_within_deadline",
        "cancellation_propagates",
        "cleanup_contract",
        "disconnect_terminal",
    ]


async def test_qualify_client_lifecycle_flags_a_store_that_hangs_after_close() -> None:
    class HangingAfterClose:
        def __init__(self) -> None:
            self._closed = False

        async def count(self, **kwargs: object) -> int:
            if self._closed:
                await asyncio.sleep(30.0)
            await asyncio.sleep(0)
            return 0

        async def aclose(self) -> None:
            await asyncio.sleep(0)
            self._closed = True

    store = HangingAfterClose()
    report = await qualify_client_lifecycle(store, store.count, startup_timeout=0.05)
    assert not report.passed
    assert "disconnect_terminal" in report.failed_checks


async def test_cancellation_of_the_qualifying_caller_still_propagates() -> None:
    client = DeployedEmbedder(latency=30.0)
    task = asyncio.ensure_future(qualify_embedding_client(client, startup_timeout=30.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
