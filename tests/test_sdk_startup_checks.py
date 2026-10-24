"""Host readiness, cancellation and resource transfer at the builder boundary."""

from __future__ import annotations

import asyncio

import pytest

from ragnor import RagnorBuilder
from ragnor.exceptions import BackendUnavailableError, ConfigurationError, OperationTimeoutError
from tests.test_builder_resource_lifecycle import HostEmbedder, OwnedResource


@pytest.mark.asyncio
async def test_build_waits_for_host_readiness_before_returning_a_usable_client():
    entered, release = asyncio.Event(), asyncio.Event()

    async def ready():
        entered.set()
        await release.wait()

    builder = RagnorBuilder.local(HostEmbedder(), embedding_dim=2).with_startup_check(ready)
    building = asyncio.create_task(builder.build())
    await asyncio.wait_for(entered.wait(), 2)
    assert not building.done()
    release.set()
    async with await building as rag:
        assert await rag.count() == 0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "5", None])
def test_invalid_startup_timeouts_are_configuration_errors(timeout):
    async def ready():
        return None

    with pytest.raises(ConfigurationError, match="timeout"):
        RagnorBuilder.local(HostEmbedder()).with_startup_check(ready, timeout_seconds=timeout)


def test_noncallable_startup_check_is_rejected():
    with pytest.raises(ConfigurationError, match="callable"):
        RagnorBuilder.local(HostEmbedder()).with_startup_check(object())


@pytest.mark.asyncio
async def test_nonawaitable_startup_result_is_rejected_and_owned_clients_are_closed():
    events = []
    builder = (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("client", events))
        .with_startup_check(lambda: None)
    )
    with pytest.raises(ConfigurationError, match="awaitable"):
        await builder.build()
    assert events == ["client"]


@pytest.mark.asyncio
async def test_failed_startup_redacts_provider_payload_and_releases_only_owned_resources():
    events = []
    borrowed = HostEmbedder()
    original = RuntimeError("opaque credential and private customer payload")

    async def unavailable():
        raise original

    builder = (
        RagnorBuilder.local(borrowed, embedding_dim=2)
        .with_owned_resource(OwnedResource("client", events))
        .with_startup_check(unavailable)
    )
    with pytest.raises(BackendUnavailableError) as caught:
        await builder.build()
    assert caught.value.__cause__ is original
    assert "opaque credential" not in str(caught.value)
    assert "private customer" not in str(caught.value)
    assert events == ["client"]
    assert borrowed.close_calls == 0


@pytest.mark.asyncio
async def test_timeout_cancels_startup_probe_and_releases_owned_resources():
    events = []
    cancelled = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    builder = (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("client", events))
        .with_startup_check(blocked, timeout_seconds=0.05)
    )
    with pytest.raises(OperationTimeoutError):
        await builder.build()
    assert cancelled.is_set()
    assert events == ["client"]


@pytest.mark.asyncio
async def test_repeated_startup_cancellation_does_not_abandon_owned_cleanup():
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def blocked():
        entered.set()
        await asyncio.Event().wait()

    class SlowResource:
        async def aclose(self):
            cleaning.set()
            await release.wait()
            events.append("slow")

    builder = (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("first", events))
        .with_owned_resource(SlowResource())
        .with_startup_check(blocked)
    )
    task = asyncio.create_task(builder.build())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert events == ["slow", "first"]


@pytest.mark.asyncio
async def test_cleanup_cancellation_from_one_resource_does_not_skip_remaining_resources():
    events = []

    class CancelledResource:
        async def aclose(self):
            events.append("cancelled")
            raise asyncio.CancelledError

    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("first", events))
        .with_owned_resource(CancelledResource())
        .build()
    )
    with pytest.raises(asyncio.CancelledError):
        await rag.aclose()
    assert events == ["cancelled", "first"]


@pytest.mark.asyncio
async def test_startup_and_disposal_failures_are_both_preserved():
    events = []

    async def unavailable():
        raise ConnectionError("provider disconnected")

    builder = (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("client", events, ValueError("disposal failed")))
        .with_startup_check(unavailable)
    )
    with pytest.raises(ExceptionGroup) as caught:
        await builder.build()
    assert isinstance(caught.value.exceptions[0], BackendUnavailableError)
    assert isinstance(caught.value.exceptions[1], ValueError)
    assert events == ["client"]


@pytest.mark.asyncio
async def test_invalid_production_wiring_does_not_run_check_or_take_resource_ownership():
    events = []

    async def ready():
        events.append("probe")

    builder = (
        RagnorBuilder(HostEmbedder(), production_mode=True)
        .with_owned_resource(OwnedResource("owned", events))
        .with_startup_check(ready)
    )
    with pytest.raises(ConfigurationError, match="production"):
        await builder.build()
    assert events == []


@pytest.mark.asyncio
async def test_failed_startup_transfers_resources_once_even_if_builder_is_reused():
    events = []

    async def unavailable():
        raise ConnectionError

    async def ready():
        return None

    builder = (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_owned_resource(OwnedResource("once", events))
        .with_startup_check(unavailable)
    )
    with pytest.raises(BackendUnavailableError):
        await builder.build()
    async with await builder.with_startup_check(ready).build():
        pass
    assert events == ["once"]
