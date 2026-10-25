"""Tests for the built-in vector database backends.

Qdrant, Chroma, and FAISS stores are exercised against
in-process fakes that mimic each client's API surface, so
the suite runs without the optional driver installed. The
Chroma store also gets real round-trip tests against
``chromadb.EphemeralClient`` when ``chromadb`` is present,
and subprocess tests prove each backend module imports
without its driver until the driver is actually needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragnor.stores._scoring import l2_distance
from ragnor.stores.chroma import ChromaStore
from ragnor.stores.faiss import FaissStore
from ragnor.stores.qdrant import QdrantStore
from ragnor.types import RetrievalError

DIM = 3
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")

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


async def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embedder returning the query embedding."""
    return [QUERY_EMBEDDING for _ in texts]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run python code in a subprocess with src on the path."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else SRC_DIR
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Qdrant fakes
# ---------------------------------------------------------------------------


class _FakeDistance:
    EUCLID = "Euclid"


class _FakeVectorParams:
    def __init__(self, size: int, distance: Any) -> None:
        self.size = size
        self.distance = distance


class _FakeMatchValue:
    def __init__(self, value: Any) -> None:
        self.value = value


class _FakeMatchAny:
    def __init__(self, any: list[Any]) -> None:
        self.any = list(any)


class _FakeFieldCondition:
    def __init__(self, key: str, match: Any) -> None:
        self.key = key
        self.match = match


class _FakeFilter:
    def __init__(self, must: list[Any]) -> None:
        self.must = must


class _FakePointIdsList:
    def __init__(self, points: list[Any]) -> None:
        self.points = list(points)


class _FakeFilterSelector:
    def __init__(self, filter: Any) -> None:
        self.filter = filter


class _FakePointStruct:
    def __init__(self, id: Any, vector: list[float], payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeAsyncQdrantClient:
    """Stand-in for ``qdrant_client.AsyncQdrantClient``."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key


def _qdrant_filter_matches(payload: dict[str, Any], query_filter: Any) -> bool:
    """Apply a fake Qdrant equality filter to a payload."""
    if query_filter is None:
        return True
    for condition in query_filter.must:
        match = condition.match
        if isinstance(match, _FakeMatchAny):
            if payload.get(condition.key) not in match.any:
                return False
        elif payload.get(condition.key) != match.value:
            return False
    return True


class _FakeQdrantClient:
    """In-process Qdrant client double."""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: dict[Any, _FakePointStruct] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vectors_config: Any = None) -> None:
        self.collections.add(collection_name)

    async def upsert(self, collection_name: str, points: list[_FakePointStruct]) -> None:
        for point in points:
            self.points[point.id] = point

    async def query_points(  # noqa: PLR0913, PLR0917
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        query_filter: Any = None,
    ) -> Any:
        scored = [
            (l2_distance(query, list(point.vector)), point)
            for point in self.points.values()
            if _qdrant_filter_matches(point.payload, query_filter)
        ]
        scored.sort(key=lambda item: (item[0], str(item[1].id)))
        return types.SimpleNamespace(points=[point for _, point in scored[:limit]])

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        offset: Any = None,
        with_payload: bool = True,
    ) -> tuple[list[_FakePointStruct], Any]:
        ordered = sorted(self.points.values(), key=lambda point: str(point.id))
        if offset is not None:
            ordered = [point for point in ordered if str(point.id) > str(offset)]
        batch = [
            point for point in ordered if _qdrant_filter_matches(point.payload, scroll_filter)
        ][:limit]
        next_offset = str(batch[-1].id) if len(batch) == limit else None
        return batch, next_offset

    async def count(self, collection_name: str, count_filter: Any = None) -> Any:
        matching = [
            point
            for point in self.points.values()
            if _qdrant_filter_matches(point.payload, count_filter)
        ]
        return types.SimpleNamespace(count=len(matching))

    async def delete(self, collection_name: str, points_selector: Any = None) -> None:
        if isinstance(points_selector, _FakePointIdsList):
            for point_id in points_selector.points:
                self.points.pop(point_id, None)
            return
        if isinstance(points_selector, _FakeFilterSelector):
            for point_id, point in list(self.points.items()):
                if _qdrant_filter_matches(point.payload, points_selector.filter):
                    del self.points[point_id]
            return
        raise AssertionError(f"unexpected points_selector: {points_selector!r}")


@pytest.fixture
def qdrant_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``qdrant_client`` module (the real one is optional)."""
    module = types.ModuleType("qdrant_client")
    module.Distance = _FakeDistance
    module.VectorParams = _FakeVectorParams
    module.MatchValue = _FakeMatchValue
    module.MatchAny = _FakeMatchAny
    module.FieldCondition = _FakeFieldCondition
    module.Filter = _FakeFilter
    module.PointIdsList = _FakePointIdsList
    module.FilterSelector = _FakeFilterSelector
    module.PointStruct = _FakePointStruct
    module.AsyncQdrantClient = _FakeAsyncQdrantClient
    # Modern qdrant-client exposes payload/config types under .models.
    # A package-shaped fake preserves core-only test isolation while the
    # optional-backend matrix runs the same code against the real client.
    module.__path__ = []
    models = types.ModuleType("qdrant_client.models")
    for name in (
        "Distance",
        "VectorParams",
        "MatchValue",
        "MatchAny",
        "FieldCondition",
        "Filter",
        "PointIdsList",
        "FilterSelector",
        "PointStruct",
    ):
        setattr(models, name, getattr(module, name))
    module.models = models
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    return module


@pytest.fixture
def qdrant_client() -> _FakeQdrantClient:
    return _FakeQdrantClient()


def _qdrant_store(client: _FakeQdrantClient, **kwargs: Any) -> QdrantStore:
    return QdrantStore(table_name="docs", embedding_dim=DIM, client=client, **kwargs)


# ---------------------------------------------------------------------------
# Chroma fakes
# ---------------------------------------------------------------------------


def _chroma_matches(metadata: dict[str, Any], where: Any) -> bool:
    """Apply a Chroma ``where`` equality filter to metadata."""
    if not where:
        return True
    return all(metadata.get(key) == condition.get("$eq") for key, condition in where.items())


class _FakeChromaCollection:
    """In-process Chroma collection double."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, ids: Any, documents: Any, embeddings: Any, metadatas: Any) -> None:
        for doc_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas, strict=False
        ):
            self.records[doc_id] = {
                "document": document,
                "embedding": list(embedding),
                "metadata": dict(metadata or {}),
            }

    def query(
        self,
        query_embeddings: Any,
        n_results: int,
        where: Any = None,
    ) -> dict[str, Any]:
        query = list(query_embeddings[0])
        scored = [
            (l2_distance(query, record["embedding"]), doc_id)
            for doc_id, record in self.records.items()
            if _chroma_matches(record["metadata"], where)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        batch = scored[:n_results]
        return {
            "ids": [[doc_id for _, doc_id in batch]],
            "documents": [[self.records[doc_id]["document"] for _, doc_id in batch]],
            "distances": [[distance for distance, _ in batch]],
            "metadatas": [[self.records[doc_id]["metadata"] for _, doc_id in batch]],
        }

    def get(self, ids: Any = None, where: Any = None) -> dict[str, Any]:
        selected = [
            doc_id
            for doc_id, record in self.records.items()
            if _chroma_matches(record["metadata"], where)
        ]
        if ids is not None:
            selected = [doc_id for doc_id in selected if doc_id in set(ids)]
        return {
            "ids": selected,
            "documents": [self.records[doc_id]["document"] for doc_id in selected],
            "metadatas": [self.records[doc_id]["metadata"] for doc_id in selected],
        }

    def count(self) -> int:
        return len(self.records)

    def delete(self, ids: Any = None, where: Any = None) -> None:
        if ids is not None:
            for doc_id in ids:
                self.records.pop(doc_id, None)
            return
        if where is not None:
            for doc_id, record in list(self.records.items()):
                if _chroma_matches(record["metadata"], where):
                    del self.records[doc_id]


class _FakeChromaClient:
    """In-process Chroma client double."""

    def __init__(self) -> None:
        self.collections: dict[str, _FakeChromaCollection] = {}

    def get_or_create_collection(self, name: str) -> _FakeChromaCollection:
        if name not in self.collections:
            self.collections[name] = _FakeChromaCollection()
        return self.collections[name]


@pytest.fixture
def chroma_client() -> _FakeChromaClient:
    return _FakeChromaClient()


def _chroma_store(client: _FakeChromaClient, **kwargs: Any) -> ChromaStore:
    return ChromaStore(table_name="docs", embedding_dim=DIM, client=client, **kwargs)


# ---------------------------------------------------------------------------
# FAISS fakes
# ---------------------------------------------------------------------------


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


@pytest.fixture
def faiss_index() -> _FakeFaissIndex:
    return _FakeFaissIndex()


def _faiss_store(index: _FakeFaissIndex, **kwargs: Any) -> FaissStore:
    return FaissStore(table_name="docs", embedding_dim=DIM, index=index, **kwargs)


# ---------------------------------------------------------------------------
# Backend selection through the registry factory
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_get_vector_store_selects_every_builtin_backend(self) -> None:
        from ragnor.stores import (
            ChromaStore as LazyChromaStore,
        )
        from ragnor.stores import (
            FaissStore as LazyFaissStore,
        )
        from ragnor.stores import (
            InMemoryVectorStore,
        )
        from ragnor.stores import (
            PgVectorStore as LazyPgVectorStore,
        )
        from ragnor.stores import (
            QdrantStore as LazyQdrantStore,
        )
        from ragnor.stores import get_vector_store

        for provider, store_class in (
            ("memory", InMemoryVectorStore),
            ("pgvector", LazyPgVectorStore),
            ("qdrant", LazyQdrantStore),
            ("chroma", LazyChromaStore),
            ("faiss", LazyFaissStore),
        ):
            settings = types.SimpleNamespace(vector_store_provider=provider)
            store = get_vector_store(table_name="docs", embedding_dim=DIM, settings=settings)
            assert isinstance(store, store_class), provider


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------


class TestQdrantStore:
    async def test_upsert_creates_collection_and_stores_points(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        await store.upsert(documents=DOCUMENTS)

        assert "docs" in qdrant_client.collections
        assert len(qdrant_client.points) == 3
        # String document ids map to UUID point ids; the
        # original id is kept in the payload.
        assert {point.payload["id"] for point in qdrant_client.points.values()} == {
            "doc-1",
            "doc-2",
            "doc-3",
        }

    async def test_hybrid_search_merges_vector_and_keyword_stages(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        # doc-1: found by both stages
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        # doc-2: keyword-only hit
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        # doc-3: vector-only hit
        assert by_id["doc-3"]["distance"] == pytest.approx(
            l2_distance(QUERY_EMBEDDING, [0.11, 0.21, 0.31])
        )
        assert by_id["doc-3"]["keyword_score"] is None
        # A document found by both stages outranks single-stage
        # hits; the two single-stage hits tie and break by id.
        assert results[0]["_rrf_score"] > results[1]["_rrf_score"]
        assert results[1]["_rrf_score"] == pytest.approx(results[2]["_rrf_score"])

    async def test_hybrid_search_returns_requested_metadata_fields(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            metadata_fields=["tenant"],
        )

        assert results[0]["metadata"] == {"tenant": "a"}

    async def test_hybrid_search_applies_filters_to_both_stages(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            tenant="a",
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]

    async def test_hybrid_search_threshold_excludes_distant_vectors(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=5, threshold=0.01)

        # doc-3's distance (~0.0173) exceeds the threshold;
        # doc-1 (distance 0) remains, doc-2 is a keyword hit.
        assert [r["id"] for r in results] == ["doc-1", "doc-2"]

    async def test_count_with_and_without_filters(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        assert await store.count() == 3
        assert await store.count(tenant="a") == 2
        assert await store.count(tenant="c") == 0

    async def test_hybrid_search_auto_embeds_query(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client, embedding_fn=_embed)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]

    async def test_hybrid_search_without_embedding_fn_raises(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        with pytest.raises(ValueError, match="embedding_fn is required"):
            await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)

    async def test_strict_mode_raises_retrieval_error(
        self, qdrant_module: types.ModuleType
    ) -> None:
        class _FailingClient(_FakeQdrantClient):
            async def query_points(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("qdrant down")

        store = _qdrant_store(_FailingClient())

        with pytest.raises(RetrievalError) as caught:
            await store.hybrid_search(
                QUERY,
                QUERY_EMBEDDING,
                n_results=5,
                threshold=THRESHOLD,
                strict=True,
            )

        assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
        assert str(caught.value.__cause__) == "qdrant down"
        assert "qdrant down" not in str(caught.value)

        assert (
            await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)
            == []
        )

    async def test_upsert_extra_fields(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        await store.upsert(
            documents=[
                {
                    "id": "doc-1",
                    "text": "alpha beta",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {},
                }
            ],
            extra_fields={"tenant_id": "t1"},
        )

        point = next(iter(qdrant_client.points.values()))
        assert point.payload["tenant_id"] == "t1"

    async def test_upsert_validates_embeddings(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        with pytest.raises(ValueError, match="dimensions"):
            await store.upsert(
                documents=[
                    {
                        "id": "x",
                        "text": "t",
                        "embedding": [0.1],
                        "metadata": {},
                    }
                ]
            )

    async def test_upsert_empty_list_is_noop(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        await store.upsert(documents=[])

        assert qdrant_client.points == {}

    def test_construction_validates_arguments(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        with pytest.raises(ValueError, match="embedding_dim is required"):
            QdrantStore(table_name="docs", client=qdrant_client)
        with pytest.raises(ValueError, match="Unsafe table_name"):
            QdrantStore(table_name="bad;name", embedding_dim=DIM, client=qdrant_client)

    def test_client_created_lazily_from_explicit_url(self, qdrant_module: types.ModuleType) -> None:
        store = QdrantStore(
            table_name="docs",
            embedding_dim=DIM,
            url="http://qdrant:6333",
        )

        client = store._resolve_client()

        assert isinstance(client, _FakeAsyncQdrantClient)
        assert client.url == "http://qdrant:6333"

    def test_client_resolution_without_url_raises(self, qdrant_module: types.ModuleType) -> None:
        store = QdrantStore(table_name="docs", embedding_dim=DIM)

        with pytest.raises(RuntimeError, match="QdrantStore needs an explicit url"):
            store._resolve_client()


class TestQdrantStoreCapabilities:
    """Capability split: vector_search, keyword_search, delete, get."""

    async def test_satisfies_capability_protocols(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        from ragnor.stores.base import (
            DocumentStore,
            LexicalIndex,
            VectorIndex,
        )

        store = _qdrant_store(qdrant_client)
        assert isinstance(store, VectorIndex)
        assert isinstance(store, LexicalIndex)
        assert isinstance(store, DocumentStore)

    async def test_golden_hybrid_search_composite_matches_pre_refactor_output(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        """The inherited composite reproduces the pre-refactor hybrid_search.

        Golden output captured from the fused implementation: ids,
        scores, and ordering must be identical.
        """
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-1"]["_rrf_score"] == pytest.approx(2.0 / 61)
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-2"]["_rrf_score"] == pytest.approx(1.0 / 62)
        assert by_id["doc-3"]["distance"] == pytest.approx(
            l2_distance(QUERY_EMBEDDING, [0.11, 0.21, 0.31])
        )
        assert by_id["doc-3"]["keyword_score"] is None
        assert by_id["doc-3"]["_rrf_score"] == pytest.approx(1.0 / 62)

    async def test_vector_search_only(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.vector_search(QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)

        # doc-1 (distance 0) and doc-3 (distance ~0.017) are within the
        # threshold; doc-2 is beyond it.
        assert [r["id"] for r in results] == ["doc-1", "doc-3"]
        assert results[0]["distance"] == pytest.approx(0.0)
        assert results[0]["keyword_score"] is None
        assert results[1]["distance"] == pytest.approx(
            l2_distance(QUERY_EMBEDDING, [0.11, 0.21, 0.31])
        )
        # The vector stage applies filters.
        scoped = await store.vector_search(
            QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD, tenant="b"
        )
        assert [r["id"] for r in scoped] == ["doc-3"]

    async def test_keyword_search_only(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.keyword_search(QUERY, n_results=5)

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]
        assert results[0]["keyword_score"] == pytest.approx(1.0)
        assert results[1]["keyword_score"] == pytest.approx(0.5)
        assert results[0]["distance"] is None
        # The keyword stage applies filters.
        scoped = await store.keyword_search(QUERY, n_results=5, tenant="b")
        assert scoped == []

    async def test_keyword_search_truncates_to_n_results(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(
            documents=[
                {
                    "id": f"doc-{i}",
                    "text": "shared token",
                    "metadata": {"tenant": "a"},
                    "embedding": [0.1, 0.2, 0.3],
                }
                for i in range(5)
            ]
        )

        results = await store.keyword_search("shared token", n_results=2)

        assert [r["id"] for r in results] == ["doc-0", "doc-1"]

    async def test_delete_by_ids(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(ids=["doc-1", "doc-3"])

        assert deleted == 2
        assert await store.count() == 1
        assert await store.delete(ids=["missing"]) == 0

    async def test_delete_by_filters(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(filters={"tenant": "a"})

        assert deleted == 2
        assert await store.count() == 1

    async def test_delete_by_ids_and_filters_is_conjunction(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        # doc-1 and doc-2 carry tenant "a"; only doc-1 is in the id list.
        deleted = await store.delete(ids=["doc-1", "doc-3"], filters={"tenant": "a"})

        assert deleted == 1
        assert await store.count() == 2

    async def test_delete_rejects_empty_ids(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        with pytest.raises(ValueError, match="empty"):
            await store.delete(ids=[])

    async def test_delete_rejects_missing_selector(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)

        with pytest.raises(ValueError, match="at least one"):
            await store.delete()

    async def test_get_by_ids(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-2", "doc-3"], metadata_fields=["tenant"])

        assert [row["id"] for row in rows] == ["doc-2", "doc-3"]
        assert rows[0]["document"] == "alpha delta"
        assert rows[0]["metadata"] == {"tenant": "a"}
        assert await store.get(ids=["missing"]) == []

    async def test_get_by_filters(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_by_ids_and_filters_is_conjunction(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-1", "doc-3"], filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_with_no_selector_returns_everything(
        self, qdrant_module: types.ModuleType, qdrant_client: _FakeQdrantClient
    ) -> None:
        store = _qdrant_store(qdrant_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get()

        assert {row["id"] for row in rows} == {"doc-1", "doc-2", "doc-3"}

    async def test_delete_and_get_failures_propagate(self, qdrant_module: types.ModuleType) -> None:
        """Lifecycle operations have no strict parameter: failures raise."""

        class _FailingClient(_FakeQdrantClient):
            async def count(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("qdrant down")

            async def scroll(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("qdrant down")

        store = _qdrant_store(_FailingClient())

        with pytest.raises(RuntimeError, match="qdrant down"):
            await store.delete(filters={"tenant": "a"})
        with pytest.raises(RuntimeError, match="qdrant down"):
            await store.get()


# ---------------------------------------------------------------------------
# ChromaStore (fake client)
# ---------------------------------------------------------------------------


class TestChromaStore:
    async def test_upsert_and_hybrid_search(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-3"]["keyword_score"] is None

    async def test_hybrid_search_applies_filters(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            tenant="a",
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]

    async def test_count_with_and_without_filters(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        assert await store.count() == 3
        assert await store.count(tenant="b") == 1

    async def test_hybrid_search_auto_embeds_query(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client, embedding_fn=_embed)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]

    async def test_strict_mode_raises_retrieval_error(self) -> None:
        class _FailingCollection(_FakeChromaCollection):
            def query(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("chroma down")

        class _FailingClient(_FakeChromaClient):
            def get_or_create_collection(self, name: str) -> Any:
                return _FailingCollection()

        store = _chroma_store(_FailingClient())

        with pytest.raises(RetrievalError) as caught:
            await store.hybrid_search(
                QUERY,
                QUERY_EMBEDDING,
                n_results=5,
                threshold=THRESHOLD,
                strict=True,
            )

        assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
        assert str(caught.value.__cause__) == "chroma down"
        assert "chroma down" not in str(caught.value)

        assert (
            await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)
            == []
        )

    async def test_upsert_validates_embeddings(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)

        with pytest.raises(ValueError, match="dimensions"):
            await store.upsert(
                documents=[
                    {
                        "id": "x",
                        "text": "t",
                        "embedding": [0.1],
                        "metadata": {},
                    }
                ]
            )

    async def test_hybrid_search_metadata_fields_and_validation(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            metadata_fields=["tenant"],
        )
        assert results[0]["metadata"] == {"tenant": "a"}

        with pytest.raises(ValueError, match="n_results must be a positive"):
            await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=0, threshold=THRESHOLD)
        with pytest.raises(ValueError, match="embedding_fn is required"):
            await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)
        with pytest.raises(ValueError, match="Unsafe metadata_field"):
            await store.hybrid_search(
                QUERY,
                QUERY_EMBEDDING,
                n_results=5,
                threshold=THRESHOLD,
                metadata_fields=["bad;field"],
            )

    async def test_upsert_extra_fields_and_empty_list(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)

        await store.upsert(documents=[])
        assert chroma_client.collections == {}

        await store.upsert(
            documents=[
                {
                    "id": "doc-1",
                    "text": "alpha",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {},
                }
            ],
            extra_fields={"tenant_id": "t1"},
        )
        record = chroma_client.collections["docs"].records["doc-1"]
        assert record["metadata"] == {"tenant_id": "t1"}

        with pytest.raises(ValueError, match="Unsafe extra_field"):
            await store.upsert(
                documents=[
                    {
                        "id": "doc-2",
                        "text": "beta",
                        "embedding": [0.1, 0.2, 0.3],
                        "metadata": {},
                    }
                ],
                extra_fields={"bad;key": "v"},
            )

    async def test_count_strict_mode_raises_retrieval_error(self) -> None:
        class _FailingCollection(_FakeChromaCollection):
            def count(self) -> int:
                raise RuntimeError("chroma down")

        class _FailingClient(_FakeChromaClient):
            def get_or_create_collection(self, name: str) -> Any:
                return _FailingCollection()

        store = _chroma_store(_FailingClient())

        with pytest.raises(RetrievalError) as caught:
            await store.count(strict=True)

        assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
        assert str(caught.value.__cause__) == "chroma down"
        assert "chroma down" not in str(caught.value)

        assert await store.count() == 0

    async def test_upsert_failure_raises(self) -> None:
        class _FailingCollection(_FakeChromaCollection):
            def upsert(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("chroma down")

        class _FailingClient(_FakeChromaClient):
            def get_or_create_collection(self, name: str) -> Any:
                return _FailingCollection()

        store = _chroma_store(_FailingClient())

        with pytest.raises(RuntimeError, match="chroma down"):
            await store.upsert(documents=DOCUMENTS)

    async def test_default_ephemeral_client(self) -> None:
        pytest.importorskip("chromadb")
        store = ChromaStore(table_name="default_docs", embedding_dim=DIM)

        collection = store._resolve_collection()

        assert collection is not None

    def test_construction_validates_arguments(self, chroma_client: _FakeChromaClient) -> None:
        with pytest.raises(ValueError, match="embedding_dim is required"):
            ChromaStore(table_name="docs", client=chroma_client)
        with pytest.raises(ValueError, match="Unsafe table_name"):
            ChromaStore(table_name="bad;name", embedding_dim=DIM, client=chroma_client)


class TestChromaStoreCapabilities:
    """Capability split: vector_search, keyword_search, delete, get."""

    async def test_satisfies_capability_protocols(self, chroma_client: _FakeChromaClient) -> None:
        from ragnor.stores.base import (
            DocumentStore,
            LexicalIndex,
            VectorIndex,
        )

        store = _chroma_store(chroma_client)
        assert isinstance(store, VectorIndex)
        assert isinstance(store, LexicalIndex)
        assert isinstance(store, DocumentStore)

    async def test_golden_hybrid_search_composite_matches_pre_refactor_output(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        """The inherited composite reproduces the pre-refactor hybrid_search."""
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-1"]["_rrf_score"] == pytest.approx(2.0 / 61)
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-2"]["_rrf_score"] == pytest.approx(1.0 / 62)
        assert by_id["doc-3"]["keyword_score"] is None
        assert by_id["doc-3"]["_rrf_score"] == pytest.approx(1.0 / 62)

    async def test_vector_search_only(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.vector_search(QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)

        assert [r["id"] for r in results] == ["doc-1", "doc-3"]
        assert results[0]["distance"] == pytest.approx(0.0)
        assert results[0]["keyword_score"] is None
        scoped = await store.vector_search(
            QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD, tenant="b"
        )
        assert [r["id"] for r in scoped] == ["doc-3"]

    async def test_keyword_search_only(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        results = await store.keyword_search(QUERY, n_results=5)

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]
        assert results[0]["keyword_score"] == pytest.approx(1.0)
        assert results[1]["keyword_score"] == pytest.approx(0.5)
        assert results[0]["distance"] is None
        scoped = await store.keyword_search(QUERY, n_results=5, tenant="b")
        assert scoped == []

    async def test_keyword_search_truncates_to_n_results(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(
            documents=[
                {
                    "id": f"doc-{i}",
                    "text": "shared token",
                    "metadata": {"tenant": "a"},
                    "embedding": [0.1, 0.2, 0.3],
                }
                for i in range(5)
            ]
        )

        results = await store.keyword_search("shared token", n_results=2)

        assert [r["id"] for r in results] == ["doc-0", "doc-1"]

    async def test_delete_by_ids(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(ids=["doc-1", "doc-3"])

        assert deleted == 2
        assert await store.count() == 1
        assert await store.delete(ids=["missing"]) == 0

    async def test_delete_by_filters(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(filters={"tenant": "a"})

        assert deleted == 2
        assert await store.count() == 1

    async def test_delete_by_ids_and_filters_is_conjunction(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        # doc-1 and doc-2 carry tenant "a"; only doc-1 is in the id list.
        deleted = await store.delete(ids=["doc-1", "doc-3"], filters={"tenant": "a"})

        assert deleted == 1
        assert await store.count() == 2

    async def test_delete_rejects_empty_ids(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)

        with pytest.raises(ValueError, match="empty"):
            await store.delete(ids=[])

    async def test_delete_rejects_missing_selector(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)

        with pytest.raises(ValueError, match="at least one"):
            await store.delete()

    async def test_get_by_ids(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-2", "doc-3"], metadata_fields=["tenant"])

        assert [row["id"] for row in rows] == ["doc-2", "doc-3"]
        assert rows[0]["document"] == "alpha delta"
        assert rows[0]["metadata"] == {"tenant": "a"}
        assert await store.get(ids=["missing"]) == []

    async def test_get_by_filters(self, chroma_client: _FakeChromaClient) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_by_ids_and_filters_is_conjunction(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-1", "doc-3"], filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_with_no_selector_returns_everything(
        self, chroma_client: _FakeChromaClient
    ) -> None:
        store = _chroma_store(chroma_client)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get()

        assert {row["id"] for row in rows} == {"doc-1", "doc-2", "doc-3"}


# ---------------------------------------------------------------------------
# ChromaStore (real client, when chromadb is installed)
# ---------------------------------------------------------------------------


class TestChromaStoreRealClient:
    async def test_roundtrip(self) -> None:
        chromadb = pytest.importorskip("chromadb")
        client = chromadb.EphemeralClient()
        store = ChromaStore(table_name="real_docs", embedding_dim=DIM, client=client)

        await store.upsert(documents=DOCUMENTS)

        assert await store.count() == 3
        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )
        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0, abs=1e-5)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-3"]["keyword_score"] is None

    async def test_filters_and_count(self) -> None:
        chromadb = pytest.importorskip("chromadb")
        client = chromadb.EphemeralClient()
        store = ChromaStore(table_name="real_docs_filtered", embedding_dim=DIM, client=client)

        await store.upsert(documents=DOCUMENTS)

        assert await store.count(tenant="a") == 2
        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            tenant="a",
        )
        assert [r["id"] for r in results] == ["doc-1", "doc-2"]

    async def test_persistent_client(self, tmp_path: Path) -> None:
        pytest.importorskip("chromadb")
        store = ChromaStore(
            table_name="persistent_docs",
            embedding_dim=DIM,
            path=str(tmp_path),
        )

        await store.upsert(documents=DOCUMENTS)

        assert await store.count() == 3


# ---------------------------------------------------------------------------
# FaissStore
# ---------------------------------------------------------------------------


class TestFaissStore:
    async def test_upsert_and_hybrid_search(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-3"]["keyword_score"] is None

    async def test_upsert_replaces_existing_document(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=[DOCUMENTS[0]])
        moved = {**DOCUMENTS[0], "embedding": [0.9, 0.9, 0.9]}
        await store.upsert(documents=[moved])

        assert await store.count() == 1
        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )
        # doc-1 is no longer a vector hit (distance ~1.22);
        # it is still a keyword hit.
        assert [r["id"] for r in results] == ["doc-1"]
        assert results[0]["distance"] is None
        assert results[0]["keyword_score"] == pytest.approx(1.0)

    async def test_hybrid_search_applies_filters(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY,
            QUERY_EMBEDDING,
            n_results=5,
            threshold=THRESHOLD,
            tenant="a",
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]

    async def test_count_with_filters(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        assert await store.count() == 3
        assert await store.count(tenant="b") == 1

    async def test_search_on_empty_index_returns_no_results(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert results == []

    async def test_hybrid_search_auto_embeds_query(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index, embedding_fn=_embed)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]

    async def test_strict_mode_raises_retrieval_error(self) -> None:
        class _FailingIndex(_FakeFaissIndex):
            def search(self, query: Any, k: int) -> Any:
                raise RuntimeError("faiss down")

        store = _faiss_store(_FailingIndex())

        with pytest.raises(RetrievalError) as caught:
            await store.hybrid_search(
                QUERY,
                QUERY_EMBEDDING,
                n_results=5,
                threshold=THRESHOLD,
                strict=True,
            )

        assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
        assert str(caught.value.__cause__) == "faiss down"
        assert "faiss down" not in str(caught.value)

        assert (
            await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)
            == []
        )

    async def test_upsert_validates_embeddings(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)

        with pytest.raises(ValueError, match="dimensions"):
            await store.upsert(
                documents=[
                    {
                        "id": "x",
                        "text": "t",
                        "embedding": [0.1],
                        "metadata": {},
                    }
                ]
            )

    async def test_validation_errors(self, faiss_index: _FakeFaissIndex) -> None:
        with pytest.raises(ValueError, match="embedding_dim is required"):
            FaissStore(table_name="docs", index=faiss_index)
        with pytest.raises(ValueError, match="Unsafe table_name"):
            FaissStore(table_name="bad;name", embedding_dim=DIM, index=faiss_index)
        store = _faiss_store(faiss_index)
        with pytest.raises(ValueError, match="n_results must be a positive"):
            await store.hybrid_search(QUERY, QUERY_EMBEDDING, n_results=0, threshold=THRESHOLD)
        with pytest.raises(ValueError, match="embedding_fn is required"):
            await store.hybrid_search(QUERY, [], n_results=5, threshold=THRESHOLD)


class TestFaissStoreCapabilities:
    """Capability split: vector_search, keyword_search, delete, get."""

    async def test_satisfies_capability_protocols(self, faiss_index: _FakeFaissIndex) -> None:
        from ragnor.stores.base import (
            DocumentStore,
            LexicalIndex,
            VectorIndex,
        )

        store = _faiss_store(faiss_index)
        assert isinstance(store, VectorIndex)
        assert isinstance(store, LexicalIndex)
        assert isinstance(store, DocumentStore)

    async def test_golden_hybrid_search_composite_matches_pre_refactor_output(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        """The inherited composite reproduces the pre-refactor hybrid_search."""
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        results = await store.hybrid_search(
            QUERY, QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD
        )

        assert [r["id"] for r in results] == ["doc-1", "doc-2", "doc-3"]
        by_id = {r["id"]: r for r in results}
        assert by_id["doc-1"]["distance"] == pytest.approx(0.0)
        assert by_id["doc-1"]["keyword_score"] == pytest.approx(1.0)
        assert by_id["doc-1"]["_rrf_score"] == pytest.approx(2.0 / 61)
        assert by_id["doc-2"]["distance"] is None
        assert by_id["doc-2"]["keyword_score"] == pytest.approx(0.5)
        assert by_id["doc-2"]["_rrf_score"] == pytest.approx(1.0 / 62)
        assert by_id["doc-3"]["keyword_score"] is None
        assert by_id["doc-3"]["_rrf_score"] == pytest.approx(1.0 / 62)

    async def test_vector_search_only(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        results = await store.vector_search(QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD)

        assert [r["id"] for r in results] == ["doc-1", "doc-3"]
        assert results[0]["distance"] == pytest.approx(0.0)
        assert results[0]["keyword_score"] is None
        scoped = await store.vector_search(
            QUERY_EMBEDDING, n_results=5, threshold=THRESHOLD, tenant="b"
        )
        assert [r["id"] for r in scoped] == ["doc-3"]

    async def test_keyword_search_only(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        results = await store.keyword_search(QUERY, n_results=5)

        assert [r["id"] for r in results] == ["doc-1", "doc-2"]
        assert results[0]["keyword_score"] == pytest.approx(1.0)
        assert results[1]["keyword_score"] == pytest.approx(0.5)
        assert results[0]["distance"] is None
        scoped = await store.keyword_search(QUERY, n_results=5, tenant="b")
        assert scoped == []

    async def test_keyword_search_truncates_to_n_results(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(
            documents=[
                {
                    "id": f"doc-{i}",
                    "text": "shared token",
                    "metadata": {"tenant": "a"},
                    "embedding": [0.1, 0.2, 0.3],
                }
                for i in range(5)
            ]
        )

        results = await store.keyword_search("shared token", n_results=2)

        assert [r["id"] for r in results] == ["doc-0", "doc-1"]

    async def test_delete_by_ids(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(ids=["doc-1", "doc-3"])

        assert deleted == 2
        assert await store.count() == 1
        # The deleted vectors are gone from the FAISS index.
        assert faiss_index.ntotal == 1
        assert await store.delete(ids=["missing"]) == 0

    async def test_delete_by_filters(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        deleted = await store.delete(filters={"tenant": "a"})

        assert deleted == 2
        assert await store.count() == 1
        assert faiss_index.ntotal == 1

    async def test_delete_by_ids_and_filters_is_conjunction(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        # doc-1 and doc-2 carry tenant "a"; only doc-1 is in the id list.
        deleted = await store.delete(ids=["doc-1", "doc-3"], filters={"tenant": "a"})

        assert deleted == 1
        assert await store.count() == 2
        assert faiss_index.ntotal == 2

    async def test_delete_rejects_empty_ids(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)

        with pytest.raises(ValueError, match="empty"):
            await store.delete(ids=[])

    async def test_delete_rejects_missing_selector(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)

        with pytest.raises(ValueError, match="at least one"):
            await store.delete()

    async def test_get_by_ids(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-2", "doc-3"], metadata_fields=["tenant"])

        assert [row["id"] for row in rows] == ["doc-2", "doc-3"]
        assert rows[0]["document"] == "alpha delta"
        assert rows[0]["metadata"] == {"tenant": "a"}
        assert await store.get(ids=["missing"]) == []

    async def test_get_by_filters(self, faiss_index: _FakeFaissIndex) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_by_ids_and_filters_is_conjunction(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get(ids=["doc-1", "doc-3"], filters={"tenant": "b"})

        assert [row["id"] for row in rows] == ["doc-3"]

    async def test_get_with_no_selector_returns_everything(
        self, faiss_index: _FakeFaissIndex
    ) -> None:
        store = _faiss_store(faiss_index)
        await store.upsert(documents=DOCUMENTS)

        rows = await store.get()

        assert {row["id"] for row in rows} == {"doc-1", "doc-2", "doc-3"}


# ---------------------------------------------------------------------------
# Driver decoupling: backend modules import without their drivers
# ---------------------------------------------------------------------------


class TestDriverDecoupling:
    def test_qdrant_store_usable_without_qdrant_client(self) -> None:
        result = _run_python(
            "import sys\n"
            "sys.modules['qdrant_client'] = None\n"
            "import ragnor.stores\n"
            "store = ragnor.stores.QdrantStore(\n"
            "    table_name='docs', embedding_dim=4,\n"
            ")\n"
            "print(type(store).__name__)\n"
            "try:\n"
            "    store._resolve_client()\n"
            "except ImportError:\n"
            "    print('driver-required')\n"
        )

        assert result.returncode == 0, result.stderr
        assert "QdrantStore" in result.stdout
        assert "driver-required" in result.stdout

    def test_chroma_store_usable_without_chromadb(self) -> None:
        result = _run_python(
            "import sys\n"
            "sys.modules['chromadb'] = None\n"
            "import ragnor.stores\n"
            "store = ragnor.stores.ChromaStore(\n"
            "    table_name='docs', embedding_dim=4,\n"
            ")\n"
            "print(type(store).__name__)\n"
            "try:\n"
            "    store._resolve_collection()\n"
            "except ImportError:\n"
            "    print('driver-required')\n"
        )

        assert result.returncode == 0, result.stderr
        assert "ChromaStore" in result.stdout
        assert "driver-required" in result.stdout

    def test_faiss_store_usable_without_faiss(self) -> None:
        result = _run_python(
            "import sys\n"
            "sys.modules['faiss'] = None\n"
            "import ragnor.stores\n"
            "store = ragnor.stores.FaissStore(\n"
            "    table_name='docs', embedding_dim=4,\n"
            ")\n"
            "print(type(store).__name__)\n"
            "try:\n"
            "    store._resolve_index()\n"
            "except ImportError:\n"
            "    print('driver-required')\n"
        )

        assert result.returncode == 0, result.stderr
        assert "FaissStore" in result.stdout
        assert "driver-required" in result.stdout
