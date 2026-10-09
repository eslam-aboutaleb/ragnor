"""
Tests for the hybrid search SQL and the fields it returns.

The defect this pins down: the outer projection interpolated the metadata expression
as ``f"{meta_coalesce},"``, so an empty ``metadata_fields`` produced
``AS document, , v.distance`` — a syntax error that the broad ``except`` in
``hybrid_search`` swallowed and reported as "no results". The projection is now
assembled as a list, and every result carries the row ``id`` and the keyword rank.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragit.stores.pgvector import PgVectorStore


def _mock_row(values: dict) -> MagicMock:
    """Build a row double that only supports ``__getitem__``, like a RowMapping."""
    row = MagicMock()
    row.__getitem__ = lambda _self, key: values[key]
    row.__contains__ = lambda _self, key: key in values
    return row


def _mocked_store(row: MagicMock) -> tuple[PgVectorStore, MagicMock]:
    """Return a store wired to a session whose single query returns ``row``."""
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = [row]
    session.execute.return_value = result

    factory = MagicMock()
    factory.return_value.__aenter__.return_value = session
    factory.return_value.__aexit__.return_value = False
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="chunk_id",
        embedding_dim=1536,
        session_factory=factory,
    )
    return store, factory


@pytest.mark.asyncio
async def test_hybrid_search_without_metadata_fields_returns_rows():
    """An empty ``metadata_fields`` must not break the projection.

    This is the regression guard for the bare-comma syntax error.
    """
    row = _mock_row(
        {
            "id": "policy_chunk_0",
            "document": "water damage is covered",
            "distance": 0.1,
            "keyword_score": 0.9,
            "rrf_score": 0.03,
        }
    )
    store, _factory = _mocked_store(row)

    results = await store.hybrid_search(
        query="water damage",
        embedding=[0.1] * 1536,
        n_results=5,
        threshold=1.3,
        text_field="text",
        metadata_fields=[],
    )

    assert len(results) == 1
    assert results[0]["document"] == "water damage is covered"


@pytest.mark.asyncio
async def test_hybrid_search_returns_id_and_keyword_score():
    """Every result carries the row id and the PostgreSQL full-text rank."""
    row = _mock_row(
        {
            "id": "policy_chunk_7",
            "document": "burst pipes are covered",
            "section": "Water Damage",
            "distance": 0.25,
            "keyword_score": 0.42,
            "rrf_score": 0.03,
        }
    )
    store, _factory = _mocked_store(row)

    results = await store.hybrid_search(
        query="burst pipes",
        embedding=[0.1] * 1536,
        n_results=5,
        threshold=1.3,
        text_field="text",
        metadata_fields=["section"],
    )

    assert len(results) == 1
    assert results[0]["id"] == "policy_chunk_7"
    assert results[0]["keyword_score"] == pytest.approx(0.42)
    assert results[0]["metadata"] == {"section": "Water Damage"}
    assert results[0]["distance"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_hybrid_search_projection_has_no_empty_element():
    """The generated SQL never contains a bare comma from an empty metadata list."""
    captured: dict[str, str] = {}

    class _CapturingSession:
        async def execute(self, stmt, params=None):  # noqa: ANN001, ANN202, ARG002
            captured["sql"] = str(stmt)
            result = MagicMock()
            result.mappings.return_value.all.return_value = []
            return result

    session = _CapturingSession()
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = session
    factory.return_value.__aexit__.return_value = False

    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="chunk_id",
        embedding_dim=1536,
        session_factory=factory,
    )
    await store.hybrid_search(
        query="q",
        embedding=[0.0] * 1536,
        n_results=3,
        threshold=1.3,
        text_field="text",
        metadata_fields=[],
    )

    sql = captured["sql"]
    outer = sql.split("SELECT\n", 2)[-1].split("FROM vector_search")[0]
    assert ",," not in outer
    assert ", ," not in outer
    assert "AS id" in outer
    assert "AS keyword_score" in outer


def test_embedding_dimension_comes_from_the_constructor():
    """The store takes its dimension from the constructor, not a hardcoded literal."""
    explicit = PgVectorStore(table_name="policy_chunks", embedding_dim=42)
    assert explicit.embedding_dim == 42

    with pytest.raises(ValueError, match="embedding_dim"):
        PgVectorStore(table_name="policy_chunks")
