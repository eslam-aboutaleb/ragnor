"""Coverage completion tests for the FAISS vector store.

Complements ``tests/test_vector_backends.py`` by exercising the
FaissStore paths that the shared backend suite does not reach:
lazy index creation through a fake ``faiss`` module (the real
driver is optional and not installed here), plus the metadata
validation, unknown-label, extra-field validation, empty-upsert
no-op, and extra-field merge branches of ``hybrid_search`` and
``upsert``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from ragit.stores._scoring import l2_distance
from ragit.stores.faiss import FaissStore

DIM = 3

# Three documents covering every hybrid-search result shape:
# doc-1 is a vector AND keyword hit, doc-2 is a keyword-only
# hit (its embedding is beyond the threshold), and doc-3 is a
# vector-only hit (its text shares no tokens with the query).
DOCUMENTS = [
    {
        "id": "doc-1",
        "text": "alpha beta gamma",
        "metadata": {"tenant": "a"},
        "embedding": [0.1, 0.2, 0.3],
    },
    {
        "id": "doc-2",
        "text": "alpha delta",
        "metadata": {"tenant": "a"},
        "embedding": [0.9, 0.9, 0.9],
    },
    {
        "id": "doc-3",
        "text": "omega",
        "metadata": {"tenant": "b"},
        "embedding": [0.11, 0.21, 0.31],
    },
]

QUERY = "alpha beta"
QUERY_EMBEDDING = [0.1, 0.2, 0.3]
THRESHOLD = 1.0


class _FakeFaissIndex:
    """In-process FAISS index double (exact L2, int64 ids)."""

    def __init__(self) -> None:
        self.vectors: dict[int, list[float]] = {}

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add_with_ids(self, vectors: Any, ids: Any) -> None:
        for vector, int_id in zip(
            np.asarray(vectors, dtype="float32"),
            np.asarray(ids, dtype="int64"),
            strict=False,
        ):
            self.vectors[int(int_id)] = vector.tolist()

    def remove_ids(self, ids: Any) -> int:
        removed = 0
        for int_id in np.asarray(ids, dtype="int64"):
            if int(int_id) in self.vectors:
                del self.vectors[int(int_id)]
                removed += 1
        return removed

    def search(self, query: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vector = np.asarray(query, dtype="float32")[0].tolist()
        scored = sorted(
            (
                (l2_distance(query_vector, vector), int_id)
                for int_id, vector in self.vectors.items()
            ),
            key=lambda item: (item[0], item[1]),
        )[:k]
        distances = np.zeros((1, k), dtype="float32")
        labels = np.full((1, k), -1, dtype="int64")
        for position, (distance, int_id) in enumerate(scored):
            distances[0, position] = distance
            labels[0, position] = int_id
        return distances, labels


class _FakeIndexFlatL2:
    """Stand-in for ``faiss.IndexFlatL2``; records constructions."""

    constructed: list[_FakeIndexFlatL2] = []

    def __init__(self, dim: int) -> None:
        self.dim = dim
        _FakeIndexFlatL2.constructed.append(self)


class _FakeIndexIDMap:
    """Stand-in for ``faiss.IndexIDMap``; records constructions."""

    constructed: list[_FakeIndexIDMap] = []

    def __init__(self, index: Any) -> None:
        self.index = index
        _FakeIndexIDMap.constructed.append(self)


@pytest.fixture
def faiss_index() -> _FakeFaissIndex:
    return _FakeFaissIndex()


@pytest.fixture
def faiss_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``faiss`` module (the real one is optional)."""
    module = types.ModuleType("faiss")
    module.IndexFlatL2 = _FakeIndexFlatL2
    module.IndexIDMap = _FakeIndexIDMap
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def _faiss_store(index: _FakeFaissIndex, **kwargs: Any) -> FaissStore:
    return FaissStore(
        table_name="docs", embedding_dim=DIM, index=index, **kwargs
    )


class TestFaissStoreCoverage:
    def test_resolve_index_creates_index_lazily(
        self, faiss_module: types.ModuleType
    ) -> None:
        _FakeIndexFlatL2.constructed.clear()
        _FakeIndexIDMap.constructed.clear()
        store = FaissStore(table_name="docs", embedding_dim=DIM)

        index = store._resolve_index()

        assert isinstance(index, _FakeIndexIDMap)
        assert store._index is index
        assert _FakeIndexIDMap.constructed == [index]
        assert len(_FakeIndexFlatL2.constructed) == 1
        flat = _FakeIndexFlatL2.constructed[0]
        assert index.index is flat
        assert flat.dim == DIM

    async def test_hybrid_search_validates_metadata_fields(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        with pytest.raises(ValueError, match="Unsafe metadata_field"):
            await store.hybrid_search(
                QUERY,
                QUERY_EMBEDDING,
                n_results=5,
                threshold=THRESHOLD,
                metadata_fields=["bad;field"],
            )

    async def test_hybrid_search_skips_labels_missing_from_id_map(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)
        # An integer id the store never assigned: the vector
        # stage must skip it instead of failing.
        faiss_index.vectors[999] = [0.1, 0.2, 0.3]

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]

    async def test_upsert_validates_extra_fields(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        with pytest.raises(ValueError, match="Unsafe extra_field"):
            await store.upsert(
                documents=[
                    {
                        "id": "doc-1",
                        "text": "alpha beta gamma",
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": {},
                    }
                ],
                extra_fields={"bad;key": "v"},
            )

    async def test_upsert_empty_list_is_noop(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        await store.upsert(documents=[])

        assert faiss_index.vectors == {}
        assert store._records == {}

    async def test_upsert_extra_fields_merged_into_records(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        await store.upsert(
            documents=DOCUMENTS, extra_fields={"tenant_id": "t1"}
        )

        assert store._records["doc-1"]["tenant_id"] == "t1"
        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            metadata_fields=["tenant_id"],
        )
        assert results[0]["metadata"] == {"tenant_id": "t1"}
