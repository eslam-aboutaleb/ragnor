"""Coverage tests for the Qdrant store.

Exercises the paths in ``ragnor.stores.qdrant`` that the
shared backend suite does not reach: the ``_point_id``
document-id mapping, ``hybrid_search``'s ``n_results``
validation, and ``count``'s strict/non-strict error
handling. The fakes are minimal copies of the doubles in
``tests/test_vector_backends.py`` so this file is
self-contained; the real ``qdrant_client`` driver is not
needed because the store's driver imports are lazy.
"""

from __future__ import annotations

import types
import uuid
from typing import Any

import pytest

from ragnor.stores.qdrant import QdrantStore, _point_id
from ragnor.types import RetrievalError

DIM = 3


class _FakeQdrantClient:
    """Minimal in-process Qdrant client double."""

    def __init__(self) -> None:
        self.collections: set[str] = set()

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config: Any = None) -> None:
        self.collections.add(collection_name)

    async def count(self, collection_name: str, count_filter: Any = None) -> Any:
        return types.SimpleNamespace(count=0)


class _FailingCountClient(_FakeQdrantClient):
    """Fake whose collection exists but whose ``count`` fails."""

    def __init__(self) -> None:
        super().__init__()
        self.collections.add("docs")

    async def count(self, collection_name: str, count_filter: Any = None) -> Any:
        raise RuntimeError("qdrant down")


class TestPointId:
    def test_integer_and_uuid_ids_pass_through(self) -> None:
        assert _point_id(7) == 7
        doc_uuid = uuid.uuid4()
        assert _point_id(doc_uuid) is doc_uuid

    def test_string_id_maps_to_deterministic_uuid(self) -> None:
        first = _point_id("doc-1")
        second = _point_id("doc-1")

        assert isinstance(first, uuid.UUID)
        assert first == second
        assert first != _point_id("doc-2")


class TestQdrantStore:
    async def test_hybrid_search_rejects_non_positive_n_results(
        self,
    ) -> None:
        store = QdrantStore(table_name="docs", embedding_dim=DIM, client=_FakeQdrantClient())

        with pytest.raises(ValueError, match="n_results must be a positive integer"):
            await store.hybrid_search("alpha beta", [0.1, 0.2, 0.3], n_results=0, threshold=1.0)

    async def test_count_error_path(self) -> None:
        store = QdrantStore(table_name="docs", embedding_dim=DIM, client=_FailingCountClient())

        with pytest.raises(RetrievalError) as caught:
            await store.count(strict=True)

        assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
        assert str(caught.value.__cause__) == "qdrant down"
        assert "qdrant down" not in str(caught.value)

        assert await store.count() == 0
