"""Tests for the in-memory vector store.

The in-memory store approximates the other backends:
brute-force L2 vector search, token-overlap keyword
matching (not a real full-text search), and the same
RRF merge. Result dicts carry the same keys, so tests
written against one store run against the others.

The store subclasses ``HybridVectorStore``: the vector
stage lives in ``vector_search``, the keyword stage in
``keyword_search``, and the inherited composite
``hybrid_search`` fuses them. Golden tests pin the
composite's output against the pre-refactor behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ragnor.metrics import cosine_distance
from ragnor.stores._scoring import token_overlap as _token_overlap
from ragnor.stores.base import (
    DocumentStore,
    LexicalIndex,
    VectorIndex,
)
from ragnor.stores.memory import InMemoryVectorStore


def _store(**kwargs: object) -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding_dim=3, **kwargs)  # type: ignore[arg-type]


def test_store_satisfies_capability_protocols():
    store = _store()
    assert isinstance(store, VectorIndex)
    assert isinstance(store, LexicalIndex)
    assert isinstance(store, DocumentStore)


@pytest.mark.asyncio
async def test_golden_hybrid_search_composite_matches_pre_refactor_output():
    """The inherited composite reproduces the pre-refactor hybrid_search.

    Golden output captured from the fused implementation for a fixed
    corpus: ids, scores, and ordering must be identical.
    """
    store = _store(table_name="docs", id_field="id")
    await store.upsert(
        [
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
            {
                "id": "doc-4",
                "text": "beta gamma delta",
                "metadata": {"tenant": "b"},
                "embedding": [0.5, 0.5, 0.5],
            },
        ]
    )

    results = await store.hybrid_search("alpha beta", [0.1, 0.2, 0.3], n_results=5, threshold=1.0)

    assert results == [
        {
            "id": "doc-1",
            "document": "alpha beta gamma",
            "metadata": {},
            "distance": 0.0,
            "keyword_score": 1.0,
            "_rrf_score": pytest.approx(2.0 / 61),
        },
        {
            "id": "doc-4",
            "document": "beta gamma delta",
            "metadata": {},
            "distance": pytest.approx(0.5385164807134505),
            "keyword_score": 0.5,
            "_rrf_score": pytest.approx(2.0 / 63),
        },
        {
            "id": "doc-2",
            "document": "alpha delta",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.5,
            "_rrf_score": pytest.approx(1.0 / 62),
        },
        {
            "id": "doc-3",
            "document": "omega",
            "metadata": {},
            "distance": pytest.approx(0.017320508075688763),
            "keyword_score": None,
            "_rrf_score": pytest.approx(1.0 / 62),
        },
    ]

    # n_results truncates each stage before the fusion.
    truncated = await store.hybrid_search("alpha beta", [0.1, 0.2, 0.3], n_results=2, threshold=1.0)
    assert [row["id"] for row in truncated] == ["doc-1", "doc-2"]

    # Filters scope both stages.
    filtered = await store.hybrid_search(
        "alpha beta", [0.1, 0.2, 0.3], n_results=5, threshold=1.0, tenant="a"
    )
    assert [row["id"] for row in filtered] == ["doc-1", "doc-2"]

    # Metadata projection flows through both stages.
    projected = await store.hybrid_search(
        "alpha beta",
        [0.1, 0.2, 0.3],
        n_results=5,
        threshold=1.0,
        metadata_fields=["tenant"],
    )
    assert [row["metadata"] for row in projected] == [
        {"tenant": "a"},
        {"tenant": "b"},
        {"tenant": "a"},
        {"tenant": "b"},
    ]


@pytest.mark.asyncio
async def test_vector_search_only():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "alpha beta", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "gamma delta", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    results = await store.vector_search([1.0, 0.0, 0.0], n_results=5, threshold=1.3)
    assert [row["id"] for row in results] == ["1"]
    assert results[0]["distance"] == 0.0
    assert results[0]["keyword_score"] is None
    assert results[0]["document"] == "alpha beta"
    # The vector stage applies filters.
    scoped = await store.vector_search([1.0, 0.0, 0.0], n_results=5, threshold=1.3, owner="b")
    assert scoped == []


@pytest.mark.asyncio
async def test_vector_search_validates_embedding_dimension():
    store = _store()
    with pytest.raises(ValueError, match="dimensions"):
        await store.vector_search([1.0, 0.0], n_results=5, threshold=1.3)


@pytest.mark.asyncio
async def test_cosine_metric_ranks_by_reference_cosine_distance():
    """The cosine metric ranks by a reference cosine computation.

    With ``metric="cosine"`` the vector stage scores candidates with
    ``1 - cosine_similarity`` and drops candidates beyond the
    threshold, so the ordering and the reported distances must match
    the helpers in :mod:`ragnor.metrics`.
    """
    store = _store(metric="cosine")
    documents = [
        {"id": "a", "text": "alpha", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "beta", "embedding": [0.0, 1.0, 0.0]},
        {"id": "c", "text": "gamma", "embedding": [1.0, 1.0, 0.0]},
        {"id": "d", "text": "delta", "embedding": [-1.0, 0.0, 0.0]},
    ]
    await store.upsert(documents)
    query = [1.0, 0.2, 0.0]

    results = await store.vector_search(query, n_results=10, threshold=2.0)

    expected = sorted(
        ((cosine_distance(query, doc["embedding"]), doc["id"]) for doc in documents),
        key=lambda item: (item[0], item[1]),
    )
    assert [row["id"] for row in results] == [doc_id for _, doc_id in expected]
    for row, (distance, _doc_id) in zip(results, expected, strict=True):
        assert row["distance"] == pytest.approx(distance)

    # The threshold semantics are unchanged: a candidate whose cosine
    # distance reaches the threshold is dropped (doc-b is orthogonal
    # to the query, doc-d points the other way).
    strict = await store.vector_search(query, n_results=10, threshold=0.5)
    assert [row["id"] for row in strict] == ["a", "c"]


@pytest.mark.asyncio
async def test_cosine_metric_hybrid_search_fuses_both_stages():
    store = _store(metric="cosine")
    await store.upsert(
        [
            {"id": "1", "text": "alpha beta", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "alpha gamma", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    results = await store.hybrid_search("alpha beta", [1.0, 0.0, 0.0], n_results=5, threshold=1.0)
    # doc-1 is a perfect cosine match (distance 0.0) and a full keyword
    # hit; doc-2 is orthogonal to the query (cosine distance 1.0, so
    # the vector stage drops it) and a partial keyword hit.
    assert [row["id"] for row in results] == ["1", "2"]
    assert results[0]["distance"] == pytest.approx(0.0)
    assert results[0]["keyword_score"] == 1.0
    assert results[1]["distance"] is None
    assert results[1]["keyword_score"] == 0.5


@pytest.mark.asyncio
async def test_keyword_search_only():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "alpha beta", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "alpha gamma", "embedding": [0.0, 1.0, 0.0]},
            {"id": "3", "text": "omega", "embedding": [0.0, 0.0, 1.0]},
        ]
    )
    results = await store.keyword_search("alpha beta", n_results=5)
    assert [row["id"] for row in results] == ["1", "2"]
    assert results[0]["keyword_score"] == pytest.approx(1.0)
    assert results[1]["keyword_score"] == pytest.approx(0.5)
    assert results[0]["distance"] is None
    # The keyword stage applies filters.
    scoped = await store.keyword_search("alpha", n_results=5, owner="b")
    assert scoped == []


@pytest.mark.asyncio
async def test_keyword_search_truncates_to_n_results():
    store = _store()
    await store.upsert(
        [{"id": str(i), "text": "shared token", "embedding": [0.0, 0.0, 1.0]} for i in range(5)]
    )
    results = await store.keyword_search("shared token", n_results=2)
    assert [row["id"] for row in results] == ["0", "1"]


@pytest.mark.asyncio
async def test_delete_by_ids():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "alpha", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "beta", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    deleted = await store.delete(ids=["1"])
    assert deleted == 1
    assert await store.count() == 1
    missing_deleted = await store.delete(ids=["missing"])
    assert missing_deleted == 0


@pytest.mark.asyncio
async def test_delete_by_filters():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "alpha",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner": "a"},
            },
            {
                "id": "2",
                "text": "beta",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"owner": "b"},
            },
        ]
    )
    deleted = await store.delete(filters={"owner": "a"})
    assert deleted == 1
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_delete_by_ids_and_filters_is_conjunction():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "alpha",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner": "a"},
            },
            {
                "id": "2",
                "text": "beta",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"owner": "a"},
            },
            {
                "id": "3",
                "text": "gamma",
                "embedding": [0.0, 0.0, 1.0],
                "metadata": {"owner": "b"},
            },
        ]
    )
    # id "2" matches the filter; id "1" matches the filter too; id "3"
    # does not. Deleting ids ["1", "3"] with owner="a" deletes only "1".
    deleted = await store.delete(ids=["1", "3"], filters={"owner": "a"})
    assert deleted == 1
    assert await store.count(owner="a") == 1
    assert await store.count(owner="b") == 1


@pytest.mark.asyncio
async def test_delete_rejects_empty_ids():
    store = _store()
    with pytest.raises(ValueError, match="empty"):
        await store.delete(ids=[])


@pytest.mark.asyncio
async def test_delete_rejects_missing_selector():
    store = _store()
    with pytest.raises(ValueError, match="at least one"):
        await store.delete()


@pytest.mark.asyncio
async def test_delete_rejects_unsafe_filter_key():
    store = _store()
    with pytest.raises(ValueError, match="Unsafe"):
        await store.delete(filters={"bad-key": "v"})


@pytest.mark.asyncio
async def test_get_by_ids():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "alpha",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"section": "A"},
            },
            {"id": "2", "text": "beta", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    rows = await store.get(ids=["1", "2"])
    assert [row["id"] for row in rows] == ["1", "2"]
    assert rows[0]["document"] == "alpha"
    assert rows[0]["metadata"] == {}
    assert rows[1]["document"] == "beta"
    # metadata_fields projection
    projected = await store.get(ids=["1"], metadata_fields=["section"])
    assert projected[0]["metadata"] == {"section": "A"}
    # unknown ids are simply absent
    assert await store.get(ids=["missing"]) == []


@pytest.mark.asyncio
async def test_get_by_filters():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "alpha",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner": "a"},
            },
            {
                "id": "2",
                "text": "beta",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"owner": "b"},
            },
        ]
    )
    rows = await store.get(filters={"owner": "b"})
    assert [row["id"] for row in rows] == ["2"]


@pytest.mark.asyncio
async def test_get_with_no_selector_returns_everything():
    store = _store()
    await store.upsert(
        [
            {"id": "1", "text": "alpha", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "text": "beta", "embedding": [0.0, 1.0, 0.0]},
        ]
    )
    rows = await store.get()
    assert {row["id"] for row in rows} == {"1", "2"}


@pytest.mark.asyncio
async def test_get_by_ids_and_filters_is_conjunction():
    store = _store()
    await store.upsert(
        [
            {
                "id": "1",
                "text": "alpha",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"owner": "a"},
            },
            {
                "id": "2",
                "text": "beta",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"owner": "a"},
            },
        ]
    )
    rows = await store.get(ids=["1", "2"], filters={"owner": "a"})
    assert [row["id"] for row in rows] == ["1", "2"]
    rows = await store.get(ids=["2"], filters={"owner": "a"})
    assert [row["id"] for row in rows] == ["2"]
    rows = await store.get(ids=["1"], filters={"owner": "b"})
    assert rows == []


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
