"""Contract tests for the storage capability split.

Pins the structural capability Protocols
(:class:`VectorIndex`, :class:`LexicalIndex`,
:class:`DocumentStore`), the
:class:`LegacyHybridStoreAdapter` delegation contract (it
must claim no capabilities), and :func:`fuse_hybrid`
correctness (a both-stage hit outranks a single-stage hit;
ties break by id).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from ragit.stores._scoring import fuse_hybrid
from ragit.stores.base import (
    DocumentStore,
    LegacyHybridStoreAdapter,
    LexicalIndex,
    VectorIndex,
    VectorStore,
)


class _MinimalCapabilityStore:
    """A hand-written capability object: no ragit inheritance."""

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return []

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return []

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        return 0

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: Any = None,
        session: Any = None,
    ) -> int:
        return 0

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: Any = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []


class _LegacyBackend:
    """A pre-refactor backend: only the fused VectorStore methods."""

    def __init__(self) -> None:
        self.hybrid_search = AsyncMock()
        self.count = AsyncMock(return_value=0)
        self.upsert = AsyncMock()


def test_minimal_capability_object_satisfies_protocols() -> None:
    """A host adapts an existing engine structurally -- no inheritance."""
    store = _MinimalCapabilityStore()
    assert isinstance(store, VectorIndex)
    assert isinstance(store, LexicalIndex)
    assert isinstance(store, DocumentStore)


def test_legacy_adapter_claims_no_capabilities() -> None:
    """A fused hybrid result cannot be decomposed into capability stages."""
    adapter = LegacyHybridStoreAdapter(_LegacyBackend())
    assert isinstance(adapter, VectorStore)
    assert not isinstance(adapter, VectorIndex)
    assert not isinstance(adapter, LexicalIndex)
    assert not isinstance(adapter, DocumentStore)


@pytest.mark.asyncio
async def test_legacy_adapter_delegates_all_three_methods() -> None:
    backend = _LegacyBackend()
    adapter = LegacyHybridStoreAdapter(backend)

    await adapter.hybrid_search(
        query="q",
        embedding=[0.1],
        n_results=5,
        threshold=1.0,
        text_field="body",
        metadata_fields=["section"],
        session="session",
        strict=True,
        tenant_id="t1",
    )
    backend.hybrid_search.assert_awaited_once_with(
        query="q",
        embedding=[0.1],
        n_results=5,
        threshold=1.0,
        text_field="body",
        metadata_fields=["section"],
        session="session",
        strict=True,
        tenant_id="t1",
    )

    await adapter.count(session="session", strict=True, tenant_id="t1")
    backend.count.assert_awaited_once_with(session="session", strict=True, tenant_id="t1")

    await adapter.upsert(
        documents=[{"id": "1", "text": "t", "embedding": [0.1]}],
        session="session",
        text_field="body",
        embedding_field="vector",
        id_field="doc_id",
        extra_fields={"tenant_id": "t1"},
    )
    backend.upsert.assert_awaited_once_with(
        documents=[{"id": "1", "text": "t", "embedding": [0.1]}],
        session="session",
        text_field="body",
        embedding_field="vector",
        id_field="doc_id",
        extra_fields={"tenant_id": "t1"},
    )


@pytest.mark.asyncio
async def test_legacy_adapter_exposes_wrapped_backend() -> None:
    backend = _LegacyBackend()
    adapter = LegacyHybridStoreAdapter(backend)
    assert adapter.backend is backend


def test_fuse_hybrid_both_stage_hit_outranks_single_stage_hit() -> None:
    vector_results = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {"section": "A"},
            "distance": 0.1,
            "keyword_score": None,
        }
    ]
    keyword_results = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {"section": "A"},
            "distance": None,
            "keyword_score": 0.9,
        },
        {
            "id": "b",
            "document": "doc b",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.5,
        },
    ]

    results = fuse_hybrid(vector_results, keyword_results, n_results=5)

    assert [row["id"] for row in results] == ["a", "b"]
    # "a" is found by both stages (two RRF contributions); "b" by one.
    assert results[0]["_rrf_score"] > results[1]["_rrf_score"]
    assert results[0]["_rrf_score"] == pytest.approx(2.0 / 61)
    assert results[1]["_rrf_score"] == pytest.approx(1.0 / 62)
    # distance comes from the vector row, keyword_score from the keyword row.
    assert results[0]["distance"] == pytest.approx(0.1)
    assert results[0]["keyword_score"] == pytest.approx(0.9)
    assert results[1]["distance"] is None
    assert results[1]["keyword_score"] == pytest.approx(0.5)
    assert results[0]["document"] == "doc a"
    assert results[0]["metadata"] == {"section": "A"}


def test_fuse_hybrid_tie_breaks_by_id() -> None:
    vector_results = [
        {
            "id": "b",
            "document": "doc b",
            "metadata": {},
            "distance": 0.2,
            "keyword_score": None,
        }
    ]
    keyword_results = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.5,
        }
    ]

    results = fuse_hybrid(vector_results, keyword_results, n_results=5)

    # Both are single-stage rank-1 hits: identical RRF scores, so the
    # merge breaks the tie by document id.
    assert [row["id"] for row in results] == ["a", "b"]
    assert results[0]["_rrf_score"] == pytest.approx(results[1]["_rrf_score"])
    assert results[0]["_rrf_score"] == pytest.approx(1.0 / 61)


def test_fuse_hybrid_truncates_to_n_results() -> None:
    vector_results = [
        {
            "id": f"v{i}",
            "document": f"doc v{i}",
            "metadata": {},
            "distance": float(i),
            "keyword_score": None,
        }
        for i in range(5)
    ]
    keyword_results = [
        {
            "id": f"k{i}",
            "document": f"doc k{i}",
            "metadata": {},
            "distance": None,
            "keyword_score": 1.0 - i / 10,
        }
        for i in range(5)
    ]

    results = fuse_hybrid(vector_results, keyword_results, n_results=3)

    assert len(results) == 3
    # Rank-1 hits from both stages lead; the remaining slots fill with
    # the next-best single-stage hits, ties broken by id.
    assert results[0]["id"] == "k0"
    assert results[1]["id"] == "v0"
    assert results[2]["id"] == "k1"


def test_fuse_hybrid_preserves_keyword_row_when_vector_row_absent() -> None:
    keyword_results = [
        {
            "id": "only-keyword",
            "document": "keyword doc",
            "metadata": {"tenant": "t"},
            "distance": None,
            "keyword_score": 0.75,
        }
    ]

    results = fuse_hybrid([], keyword_results, n_results=5)

    assert len(results) == 1
    assert results[0]["document"] == "keyword doc"
    assert results[0]["metadata"] == {"tenant": "t"}
    assert results[0]["distance"] is None
    assert results[0]["keyword_score"] == pytest.approx(0.75)
