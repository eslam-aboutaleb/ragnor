"""SDK-02 production startup validation avoids implicit dimensions or leaked config."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.capabilities import BackendFeature, FilterOperator, StoreCapabilities


class Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class HostStore:
    capabilities = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
        features=frozenset({BackendFeature.ISOLATION_SCOPE}),
    )
    embedding_dim = 2


@asynccontextmanager
async def sessions():
    yield object()


def builder(model: str, dimension: int | None):
    return RagnorBuilder(
        embedder=Embedder(),
        embedding_model=model,
        embedding_dim=dimension,
        store=HostStore(),
        version_store=object(),
        session_factory=sessions,
        production_mode=True,
    )


@pytest.mark.asyncio
async def test_production_unknown_dimension_requires_explicit_registration() -> None:
    with pytest.raises(ValueError, match="dimension"):
        await builder("new-unknown-embedding-model", None).build()


@pytest.mark.asyncio
async def test_production_rejects_zero_or_negative_dimensions() -> None:
    for dimension in (0, -5):
        with pytest.raises(ValueError, match="embedding_dim"):
            await builder("known-model", dimension).build()


@pytest.mark.asyncio
async def test_production_rejects_store_dimension_mismatch_before_write() -> None:
    with pytest.raises(ValueError, match="dimension"):
        await builder("host-model", 3).build()


@pytest.mark.asyncio
async def test_production_explicit_dimension_and_matching_host_store_build() -> None:
    rag = await builder("host-model", 2).build()
    assert rag.store.embedding_dim == 2


@pytest.mark.asyncio
async def test_invalid_model_identifier_does_not_enter_registry() -> None:
    with pytest.raises(ValueError, match="embedding_model"):
        await builder("", 2).build()
