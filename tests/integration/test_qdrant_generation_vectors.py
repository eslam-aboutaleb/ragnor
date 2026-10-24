"""True qdrant-client local backend exercise for immutable-generation adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import AsyncQdrantClient  # noqa: E402

from ragnor.jobs.adapters.generation_vectors import ImmutableGenerationVectorAdapter  # noqa: E402
from ragnor.jobs.publication import ActiveGenerationReader  # noqa: E402
from ragnor.stores.qdrant import QdrantStore  # noqa: E402


from tests.generation_doubles import LocalGenerationReservations  # noqa: E402


class Manifest:
    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.generation: str | None = None

    async def get_active_generation(self, document_id: str) -> str | None:
        return self.generation if document_id == "policy" else None


@pytest.mark.asyncio
async def test_real_qdrant_staging_visibility_and_idempotent_replay() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantStore(table_name="generation_vectors", embedding_dim=2, client=client)
        vector = ImmutableGenerationVectorAdapter(
            store=store,
            scope="tenant-alpha",
            reservations=LocalGenerationReservations("tenant-alpha"),
        )
        manifest = Manifest("tenant-alpha")
        reader = ActiveGenerationReader(manifest, vector)
        old = [{"id": "chunk-1", "text": "old-policy", "embedding": [1.0, 0.0]}]
        new = [{"id": "chunk-1", "text": "new-policy", "embedding": [0.0, 1.0]}]
        await vector.stage("generation-1", old)
        assert await reader.read("tenant-alpha", "policy") == ()
        manifest.generation = "generation-1"
        assert await reader.read("tenant-alpha", "policy") == ("old-policy",)
        await vector.stage("generation-2", new)
        assert await reader.read("tenant-alpha", "policy") == ("old-policy",)
        manifest.generation = "generation-2"
        assert await reader.read("tenant-alpha", "policy") == ("new-policy",)
        await vector.stage("generation-2", new)  # deterministic replay, no duplicates
        assert await reader.read("tenant-alpha", "policy") == ("new-policy",)
        with pytest.raises(ValueError, match="immutable"):
            await vector.stage("generation-2", old)
        assert await reader.read("tenant-alpha", "policy") == ("new-policy",)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_qdrant_scopes_do_not_share_identical_generation_keys() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantStore(table_name="scoped_gen", embedding_dim=2, client=client)
        left = ImmutableGenerationVectorAdapter(
            store=store, scope="left", reservations=LocalGenerationReservations("left")
        )
        right = ImmutableGenerationVectorAdapter(
            store=store, scope="right", reservations=LocalGenerationReservations("right")
        )
        await left.stage("same-gen", [{"id": "chunk", "text": "left", "embedding": [1.0, 0.0]}])
        await right.stage("same-gen", [{"id": "chunk", "text": "right", "embedding": [0.0, 1.0]}])
        assert await left.read_generation("same-gen") == ("left",)
        assert await right.read_generation("same-gen") == ("right",)
        with pytest.raises(PermissionError, match="scope"):
            await ActiveGenerationReader(Manifest("left"), left).read("right", "policy")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_invalid_stage_and_duplicate_source_ids_refused_before_writes() -> None:
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantStore(table_name="invalid_gen", embedding_dim=2, client=client)
        adapter = ImmutableGenerationVectorAdapter(
            store=store, scope="tenant", reservations=LocalGenerationReservations("tenant")
        )
        sample = {"id": "dup", "text": "ok", "embedding": [1.0, 0.0]}
        with pytest.raises(ValueError):
            await adapter.stage("", [sample])
        with pytest.raises(ValueError, match="duplicate"):
            await adapter.stage("a", [sample, sample])
        assert await adapter.read_generation("a") == ()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_staging_requires_a_scoped_reservation_port_before_vector_access():
    with pytest.raises(TypeError, match="reservations"):
        ImmutableGenerationVectorAdapter(store=object(), scope="tenant")
    with pytest.raises(PermissionError, match="scope"):
        ImmutableGenerationVectorAdapter(
            store=object(), scope="tenant", reservations=LocalGenerationReservations("other")
        )
