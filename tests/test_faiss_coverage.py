"""Coverage completion tests for the FAISS vector store.

Complements ``tests/test_vector_backends.py`` by exercising the
FaissStore paths that the shared backend suite does not reach:
lazy index creation through a fake ``faiss`` module (the real
driver is optional and not installed here), plus the metadata
validation, unknown-label, extra-field validation, empty-upsert
no-op, and extra-field merge branches of ``hybrid_search`` and
``upsert``.

The final section repeats the metric checks against the real
``faiss`` index (skipped when the optional ``faiss-cpu``
dependency is missing), so the cosine normalization and the
``l2_squared / 2.0`` reporting are verified against real float32
arithmetic rather than only the in-process doubles.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from ragit.stores._scoring import l2_distance
from ragit.stores.faiss import FaissStore
from ragit.stores.memory import InMemoryVectorStore

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


class _FakeFaissSquaredL2Index(_FakeFaissIndex):
    """FAISS double whose ``search`` reports SQUARED L2 distances.

    Real ``faiss.IndexFlatL2`` returns the squared Euclidean
    distance, which is what the cosine path's
    ``l2_squared / 2.0`` conversion expects. The base double reports
    the plain L2 distance so the L2-path tests keep matching the
    in-memory store's ``l2_distance``; this subclasses it and squares
    only the ``search`` output.
    """

    def search(self, query: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, labels = super().search(query, k)
        return np.where(labels >= 0, distances**2, distances), labels


@pytest.fixture
def faiss_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``faiss`` module (the real one is optional)."""
    module = types.ModuleType("faiss")
    module.IndexFlatL2 = _FakeIndexFlatL2
    module.IndexIDMap = _FakeIndexIDMap
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def _faiss_store(index: _FakeFaissIndex, **kwargs: Any) -> FaissStore:
    return FaissStore(table_name="docs", embedding_dim=DIM, index=index, **kwargs)


class TestFaissStoreCoverage:
    def test_resolve_index_creates_index_lazily(self, faiss_module: types.ModuleType) -> None:
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

    async def test_upsert_validates_extra_fields(self, faiss_index: _FakeFaissIndex) -> None:
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

    async def test_upsert_empty_list_is_noop(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)

        await store.upsert(documents=[])

        assert faiss_index.vectors == {}
        assert store._records == {}

    async def test_upsert_extra_fields_merged_into_records(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        await store.upsert(documents=DOCUMENTS, extra_fields={"tenant_id": "t1"})

        assert store._records["doc-1"]["tenant_id"] == "t1"
        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            metadata_fields=["tenant_id"],
        )
        assert results[0]["metadata"] == {"tenant_id": "t1"}

    async def test_cosine_metric_matches_memory_cosine_ranking(self) -> None:
        """Cosine FAISS ranks like the memory store's cosine path.

        The cosine metric normalizes the stored and query vectors and
        keeps ``IndexFlatL2``: the FAISS squared L2 distance over unit
        vectors is ``2 - 2 * cosine_similarity``, and the store reports
        half of it, so the ranking and the distances match the
        in-memory store's cosine computation.
        """
        index = _FakeFaissSquaredL2Index()
        store = _faiss_store(index, metric="cosine")
        memory = InMemoryVectorStore(table_name="docs", embedding_dim=DIM, metric="cosine")
        await store.upsert(documents=DOCUMENTS)
        await memory.upsert(documents=DOCUMENTS)

        faiss_results = await store.vector_search(QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)
        memory_results = await memory.vector_search(
            QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [row["id"] for row in faiss_results] == [row["id"] for row in memory_results]
        for faiss_row, memory_row in zip(faiss_results, memory_results, strict=True):
            assert faiss_row["distance"] == pytest.approx(memory_row["distance"], abs=1e-5)

    async def test_cosine_metric_normalizes_stored_and_query_vectors(self) -> None:
        """The cosine metric stores unit vectors and normalizes the query.

        Every index vector has unit length, and a scaled query produces
        the same cosine distances, which is only true when both sides
        of the search are normalized.
        """
        index = _FakeFaissSquaredL2Index()
        store = _faiss_store(index, metric="cosine")
        await store.upsert(documents=DOCUMENTS)

        for vector in index.vectors.values():
            norm = math.sqrt(sum(value * value for value in vector))
            assert norm == pytest.approx(1.0, abs=1e-6)

        base = await store.vector_search(QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)
        scaled_query = [value * 10.0 for value in QUERY_EMBEDDING]
        scaled = await store.vector_search(scaled_query, n_results=5, threshold=THRESHOLD)

        assert [row["id"] for row in scaled] == [row["id"] for row in base]
        for scaled_row, base_row in zip(scaled, base, strict=True):
            assert scaled_row["distance"] == pytest.approx(base_row["distance"], abs=1e-5)


# ---------------------------------------------------------------------------
# FaissStore over the real faiss-cpu index (optional dependency)
# ---------------------------------------------------------------------------

REAL_DIM = 4
REAL_QUERY = [2.0, 1.0, 0.0, 0.0]
# Six 4-D vectors chosen so the cosine distance from REAL_QUERY is
# different for every document and well separated (the closest pair
# is ~0.058 apart), which makes the expected distances and ranking
# unambiguous by hand. "mixed" is deliberately not a unit vector:
# its cosine distance is 1 - 7/sqrt(70) ~ 0.16334, whereas halving
# its unnormalized L2 distance would report 2.5.
REAL_DOCUMENTS = [
    ("x-strong", [1.0, 0.0, 0.0, 0.0]),
    ("mixed", [3.0, 1.0, 2.0, 0.0]),
    ("diag", [1.0, 2.0, 1.0, 0.0]),
    ("y-strong", [0.0, 1.0, 0.0, 0.0]),
    ("z-strong", [0.0, 0.0, 1.0, 0.0]),
    ("opposite", [-1.0, 0.0, 0.0, 0.0]),
]
# Nearest-first order under the cosine metric and, for contrast,
# under the default L2 metric: the two differ, so a metric leak
# changes the ranking, not only the distances.
REAL_COSINE_ORDER = ["x-strong", "mixed", "diag", "y-strong", "z-strong", "opposite"]
REAL_L2_ORDER = ["x-strong", "diag", "y-strong", "mixed", "z-strong", "opposite"]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    """Return the analytic ``1 - cosine_similarity(left, right)``."""
    norms = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return 1.0 - dot / norms


def _squared_l2(left: list[float], right: list[float]) -> float:
    """Return the squared Euclidean distance, as real FAISS reports it."""
    return sum(
        (left_value - right_value) ** 2 for left_value, right_value in zip(left, right, strict=True)
    )


def _norm(vector: Any) -> float:
    """Return the Euclidean norm of ``vector``."""
    return math.sqrt(sum(value * value for value in vector))


def _real_documents() -> list[dict[str, Any]]:
    """Return the real-index fixture shaped for ``upsert``."""
    return [
        {"id": doc_id, "text": doc_id, "embedding": vector, "metadata": {}}
        for doc_id, vector in REAL_DOCUMENTS
    ]


def _real_store(metric: str | None = None, table_name: str = "real_docs") -> FaissStore:
    """Build a store that creates its own real FAISS index (no injection)."""
    return FaissStore(table_name=table_name, embedding_dim=REAL_DIM, metric=metric)


class TestFaissStoreRealIndex:
    """The metric paths against the real ``faiss`` index.

    The tests above drive the cosine conversion with fake index
    doubles; these build the store without injecting an index, so
    ``_resolve_index`` creates a real ``faiss.IndexIDMap`` over
    ``faiss.IndexFlatL2`` and the normalization plus the
    ``l2_squared / 2.0`` reporting run against real float32
    arithmetic. ``faiss`` is an optional dependency, so every test
    starts with ``pytest.importorskip`` -- the same guard the real
    chromadb tests use.
    """

    # Tolerance: the real index keeps the normalized vectors in
    # float32, so its cosine distances differ from the analytic
    # float64 values by ~1e-8 on this 4-D fixture (measured maximum
    # 1.3e-8). 1e-6 leaves ample room, unlike higher dimensions.
    TOLERANCE = 1e-6

    async def test_cosine_distances_and_ranking_match_analytic_values(self) -> None:
        pytest.importorskip("faiss")
        store = _real_store(metric="cosine", table_name="real_cosine")
        await store.upsert(documents=_real_documents())

        results = await store.vector_search(REAL_QUERY, n_results=10, threshold=2.0)

        assert [row["id"] for row in results] == REAL_COSINE_ORDER
        by_id = dict(REAL_DOCUMENTS)
        for row in results:
            expected = _cosine_distance(REAL_QUERY, by_id[row["id"]])
            assert row["distance"] == pytest.approx(expected, abs=self.TOLERANCE)

    async def test_cosine_metric_normalizes_stored_and_query_vectors(self) -> None:
        pytest.importorskip("faiss")
        store = _real_store(metric="cosine", table_name="real_cosine_scale")
        await store.upsert(documents=_real_documents())
        index = store._resolve_index()

        # The real index holds unit vectors: upsert normalized them,
        # including the non-unit "mixed" document.
        for int_id in store._string_to_int.values():
            assert _norm(index.index.reconstruct(int_id)) == pytest.approx(1.0, abs=self.TOLERANCE)

        # REAL_QUERY is not unit length, and neither are the scaled
        # variants: equal distances prove the query is normalized too.
        base = await store.vector_search(REAL_QUERY, n_results=10, threshold=2.0)
        for factor in (10.0, 0.25):
            scaled = await store.vector_search(
                [value * factor for value in REAL_QUERY], n_results=10, threshold=2.0
            )
            assert [row["id"] for row in scaled] == REAL_COSINE_ORDER
            for scaled_row, base_row in zip(scaled, base, strict=True):
                assert scaled_row["distance"] == pytest.approx(
                    base_row["distance"], abs=self.TOLERANCE
                )

    async def test_default_metric_keeps_raw_vectors_and_reports_squared_l2(self) -> None:
        pytest.importorskip("faiss")
        store = _real_store(table_name="real_l2")
        await store.upsert(documents=_real_documents())
        index = store._resolve_index()

        # The default metric indexes the vectors exactly as given:
        # "mixed" reaches the index unnormalized.
        assert np.allclose(
            index.index.reconstruct(store._string_to_int["mixed"]),
            np.asarray(REAL_DOCUMENTS[1][1], dtype="float32"),
            atol=self.TOLERANCE,
        )

        results = await store.vector_search(REAL_QUERY, n_results=10, threshold=100.0)

        assert [row["id"] for row in results] == REAL_L2_ORDER
        by_id = dict(REAL_DOCUMENTS)
        for row in results:
            assert row["distance"] == pytest.approx(
                _squared_l2(REAL_QUERY, by_id[row["id"]]), abs=self.TOLERANCE
            )

        # Guard the contrast: the default path shares nothing with the
        # cosine path, so neither the ranking nor the distances match
        # a cosine store built over the same fixture.
        cosine = _real_store(metric="cosine", table_name="real_l2_contrast")
        await cosine.upsert(documents=_real_documents())
        cosine_results = await cosine.vector_search(REAL_QUERY, n_results=10, threshold=2.0)

        assert [row["id"] for row in results] != [row["id"] for row in cosine_results]
        assert [row["distance"] for row in results] != [row["distance"] for row in cosine_results]
