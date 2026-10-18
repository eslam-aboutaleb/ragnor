"""Session injection for the pgvector store's hybrid search.

When the caller supplies a ``session``, the store must execute on
that session and leave the transaction to the caller — the
injected session factory is never opened.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from ragnor.db.session import create_session_factory
from ragnor.stores.pgvector import PgVectorStore


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


def hash_embedding(text: str, dim: int = 1536) -> list[float]:
    """Return a deterministic hashing embedding for ``text``.

    Each token is hashed into one signed dimension of a
    ``dim``-dimensional vector, which is then L2-normalised. Two
    texts that share tokens get close vectors; disjoint texts land
    at distance sqrt(2) — beyond the default RAG distance threshold
    of 1.3 — so vector search behaves like bag-of-words similarity
    without any network calls.
    """
    vector = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dim
        vector[index] += 1.0 if digest[2] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        vector = [value / norm for value in vector]
    return vector


@pytest.mark.asyncio
async def test_hybrid_search_reuses_caller_session():
    """hybrid_search(session=...) executes on the caller's session.

    The store's own session factory must never be opened when the
    caller supplies the session.
    """
    captured: dict[str, Any] = {}

    class _SpySession:
        async def execute(self, stmt, params=None):  # noqa: ANN001, ANN202, ARG002
            captured["stmt"] = stmt
            captured["params"] = params
            result = MagicMock()
            result.mappings.return_value.all.return_value = []
            return result

    factory = MagicMock()
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=factory,
    )

    results = await store.hybrid_search(
        query="water damage",
        embedding=[0.0] * 1536,
        n_results=5,
        threshold=1.3,
        session=_SpySession(),
    )

    assert results == []
    assert "stmt" in captured
    factory.return_value.__aenter__.assert_not_called()


@pytest.mark.asyncio
async def test_count_reuses_caller_session():
    """count(session=...) executes on the caller's session."""
    captured: dict[str, Any] = {}

    class _SpySession:
        async def execute(self, stmt, params=None):  # noqa: ANN001, ANN202, ARG002
            captured["stmt"] = stmt
            result = MagicMock()
            result.scalar_one.return_value = 7
            return result

    factory = MagicMock()
    store = PgVectorStore(
        table_name="policy_chunks",
        id_field="id",
        embedding_dim=1536,
        session_factory=factory,
    )

    assert await store.count(session=_SpySession()) == 7
    assert "stmt" in captured
    factory.return_value.__aenter__.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_search_sees_caller_uncommitted_data():
    """hybrid_search(session=...) observes the caller's uncommitted rows.

    The claim is upserted on the caller's session without a commit;
    the search on the same session must still find it, proving the
    store reused the caller's transaction. The session is rolled
    back afterwards so the dedicated test database stays clean.
    """
    database_url = _test_database_url()
    if database_url is None:
        pytest.skip("RAGNOR_TEST_DATABASE_URL is not set")

    factory = create_session_factory(database_url)
    store = PgVectorStore(
        table_name="claims",
        id_field="id",
        embedding_dim=1536,
        session_factory=factory,
    )

    user_id = uuid.uuid4()
    claim_id = str(uuid.uuid4())
    text_content = "Claim CLM-RAGNOR03: Water Damage - Pipe burst causing kitchen flooding."
    embedding = hash_embedding(text_content)

    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO users (id, username, password_hash) "
                "VALUES (:id, :username, :password_hash)"
            ),
            {
                "id": user_id,
                "username": f"ragnor-03-{uuid.uuid4().hex[:8]}",
                "password_hash": "ragnor-03-test-placeholder-hash",
            },
        )
        await store.upsert(
            documents=[
                {
                    "id": claim_id,
                    "text": text_content,
                    "embedding": embedding,
                    "metadata": {
                        "claim_id": "CLM-RAGNOR03",
                        "policy_number": "POL-RAGNOR03",
                        "claim_type": "Water Damage",
                        "status": "Submitted",
                        "amount": 100.0,
                        "owner_id": str(user_id),
                        "description": "Pipe burst causing kitchen flooding.",
                    },
                }
            ],
            session=session,
        )

        results = await store.hybrid_search(
            query="pipe burst kitchen flooding",
            embedding=embedding,
            n_results=5,
            threshold=1.3,
            text_field="text",
            metadata_fields=["claim_id"],
            session=session,
        )

        matches = [row for row in results if row["id"] == claim_id]
        assert matches
        assert matches[0]["distance"] == 0.0
        assert matches[0]["metadata"] == {"claim_id": "CLM-RAGNOR03"}

        await session.rollback()
