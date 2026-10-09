"""Tests for the in-memory vector store.

The in-memory store approximates the other backends:
brute-force L2 vector search, token-overlap keyword
matching (not a real full-text search), and the same
RRF merge. Result dicts carry the same keys, so tests
written against one store run against the others.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragit.stores._scoring import token_overlap as _token_overlap
from ragit.stores.memory import InMemoryVectorStore


def _store(**kwargs: object) -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding_dim=3, **kwargs)  # type: ignore[arg-type]


def test_token_overlap_empty_query_tokens():
    assert _token_overlap(set(), ["a", "b"]) == 0.0


@pytest.mark.asyncio
async def test_upsert_and_count():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "water damage", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "fire damage", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_count_empty_store():
    store = _store()
    assert await store.count() == 0
    results = await store.hybrid_search(
        query="water damage",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert results == []


@pytest.mark.asyncio
async def test_upsert_empty_documents_is_noop():
    store = _store()
    await store.upsert([])
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_upsert_replaces_existing_document():
    store = _store()
    await store.upsert([{"id": "1", "text": "old text", "embedding": [1.0, 0.0, 0.0]}])
    await store.upsert([{"id": "1", "text": "new text", "embedding": [1.0, 0.0, 0.0]}])
    assert await store.count() == 1
    results = await store.hybrid_search(
        query="new text",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert [row["document"] for row in results] == ["new text"]


@pytest.mark.asyncio
async def test_vector_only_hit_has_no_keyword_score():
    store = _store()
    await store.upsert([{"id": "1", "text": "alpha", "embedding": [1.0, 0.0, 0.0]}])
    results = await store.hybrid_search(
        query="beta",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert len(results) == 1
    assert results[0]["distance"] == 0.0
    assert results[0]["keyword_score"] is None
    assert results[0]["_rrf_score"] == pytest.approx(1.0 / 61)


@pytest.mark.asyncio
async def test_keyword_only_hit_has_no_distance():
    store = _store()
    await store.upsert([{"id": "1", "text": "alpha beta", "embedding": [0.0, 0.0, 1.0]}])
    # distance([1,0,0], [0,0,1]) == sqrt(2) > 1.3, so the vector
    # stage contributes nothing and only the keyword stage matches.
    results = await store.hybrid_search(
        query="alpha beta",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert len(results) == 1
    assert results[0]["distance"] is None
    assert results[0]["keyword_score"] == pytest.approx(1.0)
    assert results[0]["_rrf_score"] == pytest.approx(1.0 / 61)


@pytest.mark.asyncio
async def test_rrf_merge_combines_both_stages():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "water damage", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "water damage", "embedding": [0.0, 0.0, 1.0]},
        ]
    )
    results = await store.hybrid_search(
        query="water damage",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert [row["id"] for row in results] == ["1", "2"]
    # Doc 1: vector rank 1 + keyword rank 1. Doc 2: keyword rank 2 only.
    assert results[0]["_rrf_score"] == pytest.approx(2.0 / 61)
    assert results[1]["_rrf_score"] == pytest.approx(1.0 / 62)


@pytest.mark.asyncio
async def test_tie_break_uses_id():
    store = _store()
    await store.upsert(
        [
            {"id": "b", "text": "same", "embedding": [1.0, 0.0, 0.0]},
            {"id": "a", "text": "same", "embedding": [1.0, 0.0, 0.0]},
        ]
    )
    results = await store.hybrid_search(
        query="same",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
    )
    assert [row["id"] for row in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_filters_scope_both_stages():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "water damage",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner_id": "a"},
            },
            {
                "id": "2",
                "text": "water damage",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner_id": "b"},
            },
        ]
    )
    results = await store.hybrid_search(
        query="water damage",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
        owner_id="a",
    )
    assert [row["id"] for row in results] == ["1"]
    assert await store.count(owner_id="a") == 1
    assert await store.count(owner_id="b") == 1


@pytest.mark.asyncio
async def test_metadata_fields_are_returned():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "water damage",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"section": "Water Damage"},
            }
        ]
    )
    results = await store.hybrid_search(
        query="water damage",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
        metadata_fields=["section"],
    )
    assert results[0]["metadata"] == {"section": "Water Damage"}


@pytest.mark.asyncio
async def test_upsert_extra_fields_are_stored():
    store = _store()
    await store.upsert(
        [{"id": "1", "text": "water damage", "embedding": [1.0, 0.0, 0.0]}],
        extra_fields={"owner_id": "a"},
    )
    results = await store.hybrid_search(
        query="water damage",
        embedding=[1.0, 0.0, 0.0],
        n_results=5,
        threshold=1.3,
        owner_id="a",
    )
    assert len(results) == 1
    assert await store.count(owner_id="b") == 0


@pytest.mark.asyncio
async def test_empty_embedding_uses_injected_embedding_fn():
    embed_fn = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    store = InMemoryVectorStore(embedding_dim=3, embedding_fn=embed_fn)
    await store.upsert([{"id": "1", "text": "alpha", "embedding": [1.0, 0.0, 0.0]}])
    results = await store.hybrid_search(
        query="alpha",
        embedding=[],
        n_results=5,
        threshold=1.3,
    )
    assert len(results) == 1
    embed_fn.assert_awaited_once_with(["alpha"])


@pytest.mark.asyncio
async def test_empty_embedding_without_embedding_fn_raises():
    store = _store()
    with pytest.raises(ValueError, match="embedding_fn"):
        await store.hybrid_search(
            query="alpha",
            embedding=[],
            n_results=5,
            threshold=1.3,
        )


@pytest.mark.asyncio
async def test_upsert_rejects_unsafe_metadata_key():
    store = _store()
    with pytest.raises(ValueError, match="Unsafe"):
        await store.upsert(
            [
                {
                    "id": "1",
                    "text": "water damage",
                    "embedding": [1.0, 0.0, 0.0],
                    "metadata": {"bad-key": "value"},
                }
            ]
        )


@pytest.mark.asyncio
async def test_upsert_rejects_wrong_dimension_embedding():
    store = _store()
    with pytest.raises(ValueError, match="dimensions"):
        await store.upsert([{"id": "1", "text": "water damage", "embedding": [1.0, 0.0]}])
