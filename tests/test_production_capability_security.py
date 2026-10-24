"""Security regression: production SDK never trusts unverified storage capabilities."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.capabilities import (
    BackendFeature,
    StoreCapabilities,
    get_backend_capabilities,
)


class PgVectorPretender:
    """A class name is not a verifiable storage guarantee."""


class ExternalStore:
    def __init__(self, capabilities: StoreCapabilities | None = None) -> None:
        if capabilities is not None:
            self.capabilities = capabilities


class FakeEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


@asynccontextmanager
async def session_factory():
    yield object()


def test_class_name_never_grants_durable_transactions_or_isolation() -> None:
    caps = get_backend_capabilities(PgVectorPretender())
    assert not caps.supports_feature(BackendFeature.TRANSACTION_GUARANTEE)
    assert not caps.supports_feature(BackendFeature.ISOLATION_SCOPE)


@pytest.mark.asyncio
async def test_production_refuses_undeclared_store_capabilities() -> None:
    builder = RagnorBuilder(
        embedder=FakeEmbedder(),
        store=ExternalStore(),
        version_store=object(),
        session_factory=session_factory,
        production_mode=True,
    )
    with pytest.raises(ValueError, match="StoreCapabilities"):
        await builder.build()


@pytest.mark.asyncio
async def test_production_refuses_custom_store_without_isolation_guarantee() -> None:
    builder = RagnorBuilder(
        embedder=FakeEmbedder(),
        store=ExternalStore(capabilities=StoreCapabilities()),
        version_store=object(),
        session_factory=session_factory,
        production_mode=True,
    )
    with pytest.raises(ValueError, match="isolation_scope"):
        await builder.build()
