"""Stored-vector fetches preserve geometry and avoid embedding-provider calls."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ragit.stores.chroma import ChromaStore
from ragit.stores.faiss import FaissStore
from ragit.stores.pgvector import PgVectorStore
from ragit.stores.qdrant import QdrantStore, _point_id
from tests.test_vector_backends import _FakeFaissIndex


async def test_faiss_fetch_preserves_original_vectors_and_static_ownership():
    store = FaissStore("docs", embedding_dim=2, index=_FakeFaissIndex(), metric="cosine")
    await store.upsert(
        [
            {
                "id": "a",
                "text": "content",
                "embedding": [3.0, 4.0],
                "metadata": {"tenant_id": "untrusted"},
            }
        ],
        extra_fields={"tenant_id": "trusted"},
    )
    vectors = await store.get_embeddings(["a", "missing"])
    assert vectors == {"a": [3.0, 4.0]}
    vectors["a"][0] = 99
    assert await store.get_embeddings(["a"]) == {"a": [3.0, 4.0]}
    assert len(await store.get(filters={"tenant_id": "trusted"})) == 1


async def test_chroma_fetch_requests_embeddings_and_handles_missing_vectors():
    import numpy as np

    calls = []

    def get(**options):
        calls.append(options)
        return {"ids": ["a"], "embeddings": np.array([[3.0, 4.0]])}

    collection = SimpleNamespace(get=get)
    client = SimpleNamespace(get_or_create_collection=lambda name: collection)
    store = ChromaStore("docs", embedding_dim=2, client=client)
    assert await store.get_embeddings(["a", "missing"]) == {"a": [3.0, 4.0]}
    assert calls == [{"ids": ["a", "missing"], "include": ["embeddings"]}]
    assert await store.get_embeddings([]) == {}


async def test_qdrant_fetch_uses_point_id_mapping_and_original_payload_ids():
    calls = []

    class Client:
        async def collection_exists(self, collection):
            return True

        async def retrieve(self, **options):
            calls.append(options)
            return [SimpleNamespace(payload={"external_id": "external"}, vector=[3.0, 4.0])]

    store = QdrantStore("docs", id_field="external_id", embedding_dim=2, client=Client())
    assert await store.get_embeddings(["external"]) == {"external": [3.0, 4.0]}
    assert calls[0]["ids"] == [_point_id("external")]
    assert calls[0]["with_vectors"] and calls[0]["with_payload"]
    assert await store.get_embeddings([]) == {}


@pytest.mark.parametrize("external_session", [False, True])
async def test_pgvector_fetch_uses_bound_selectors_and_preserves_session_ownership(
    external_session,
):
    calls = []
    entered = []
    session = object()

    class SQL:
        async def fetch_all(self, actual_session, statement, parameters):
            calls.append((actual_session, statement, parameters))
            return [{"key": "a", "embedding": "[3.0, 4.0]"}, {"key": "b", "embedding": [5.0, 6.0]}]

    @asynccontextmanager
    async def sessions():
        entered.append(True)
        yield session

    store = PgVectorStore(
        "docs", id_field="key", embedding_dim=2, sql_factory=SQL(), session_factory=sessions
    )
    selector = "a' OR 1=1 --"
    assert await store.get_embeddings(
        [selector], session=session if external_session else None
    ) == {
        "a": [3.0, 4.0],
        "b": [5.0, 6.0],
    }
    assert calls[0][0] is session
    assert selector not in calls[0][1] and calls[0][2] == {"id_0": selector}
    assert bool(entered) is not external_session
    assert await store.get_embeddings([]) == {}
