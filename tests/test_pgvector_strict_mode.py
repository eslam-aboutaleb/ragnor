"""Strict-mode error handling for the pgvector store.

Default mode preserves the historical swallow-and-return-empty
behavior (the frozen error contract); ``strict=True`` raises
:class:`ragnor.types.RetrievalError` so a failed search can no
longer masquerade as "no results".
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragnor.db.session import create_session_factory
from ragnor.stores.pgvector import PgVectorStore
from ragnor.types import RetrievalError


def _failing_factory() -> MagicMock:
    """Return a session factory whose every session fails to open."""
    factory = MagicMock()
    factory.return_value.__aenter__.side_effect = Exception("DB down")
    return factory


def _test_database_url() -> str | None:
    """Return the dedicated test database URL, if configured.

    Database-backed tests run only when ``RAGNOR_TEST_DATABASE_URL``
    names a database (env var or repo ``.env``).
    """
    url = os.environ.get("RAGNOR_TEST_DATABASE_URL")
    if url:
        return url
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return None
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("RAGNOR_TEST_DATABASE_URL="):
            return raw.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


@pytest.mark.asyncio
async def test_hybrid_search_default_returns_empty_on_failure():
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=_failing_factory(),
    )
    results = await store.hybrid_search(
        query="test", embedding=[0.0] * 1536, n_results=5, threshold=1.0
    )
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_search_strict_raises_retrieval_error():
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=_failing_factory(),
    )
    with pytest.raises(RetrievalError) as caught:
        await store.hybrid_search(
            query="test",
            embedding=[0.0] * 1536,
            n_results=5,
            threshold=1.0,
            strict=True,
        )

    assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
    assert str(caught.value.__cause__) == "DB down"
    assert "DB down" not in str(caught.value)


@pytest.mark.asyncio
async def test_count_default_returns_zero_on_failure():
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=_failing_factory(),
    )
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_count_strict_raises_retrieval_error():
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=_failing_factory(),
    )
    with pytest.raises(RetrievalError) as caught:
        await store.count(strict=True)

    assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
    assert str(caught.value.__cause__) == "DB down"
    assert "DB down" not in str(caught.value)


@pytest.mark.asyncio
async def test_hybrid_search_strict_nonexistent_table_raises():
    """strict=True against a nonexistent table raises RetrievalError.

    The default mode returns [] (the frozen error contract).
    """
    database_url = _test_database_url()
    if database_url is None:
        pytest.skip("RAGNOR_TEST_DATABASE_URL is not set")
    store = PgVectorStore(
        table_name="ragnor_03_nonexistent_table",
        id_field="id",
        embedding_dim=1536,
        session_factory=create_session_factory(database_url),
    )
    with pytest.raises(RetrievalError):
        await store.hybrid_search(
            query="water damage",
            embedding=[0.0] * 1536,
            n_results=5,
            threshold=1.3,
            strict=True,
        )
    assert (
        await store.hybrid_search(
            query="water damage",
            embedding=[0.0] * 1536,
            n_results=5,
            threshold=1.3,
        )
        == []
    )


@pytest.mark.asyncio
async def test_count_strict_nonexistent_table_raises():
    """strict=True count against a nonexistent table raises RetrievalError."""
    database_url = _test_database_url()
    if database_url is None:
        pytest.skip("RAGNOR_TEST_DATABASE_URL is not set")
    store = PgVectorStore(
        table_name="ragnor_03_nonexistent_table",
        id_field="id",
        embedding_dim=1536,
        session_factory=create_session_factory(database_url),
    )
    with pytest.raises(RetrievalError):
        await store.count(strict=True)
    assert await store.count() == 0
