"""Coverage tests for ragnor.stores.pgvector uncovered paths.

The private validation helpers are now the public
``ragnor.validation`` functions, and the session factory
is injected through the constructor.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ragnor.stores.pgvector import PgVectorStore
from ragnor.validation import validate_embedding, validate_identifier


class TestPgVectorStore:
    def test_validate_identifier_unsafe_name(self):
        with pytest.raises(ValueError, match="Unsafe"):
            validate_identifier("bad-name", "label")

    def test_validate_embedding_wrong_dimension(self):
        with pytest.raises(ValueError, match="has 2 dimensions"):
            validate_embedding([1.0, 2.0], expected_dim=1536)

    def test_validate_embedding_non_finite_values(self):
        with pytest.raises(ValueError, match="not finite"):
            validate_embedding([1.0, float("nan"), 3.0], expected_dim=3)

    def test_validate_embedding_non_numeric_value(self):
        with pytest.raises(TypeError, match="not a number"):
            validate_embedding(["a", "b"], expected_dim=2)

    @pytest.mark.asyncio
    async def test_hybrid_search_exception_returns_empty(self):
        factory = MagicMock()
        factory.return_value.__aenter__.side_effect = Exception("DB down")
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            session_factory=factory,
        )
        result = await store.hybrid_search(
            query="test", embedding=[0.0] * 1536, n_results=5, threshold=1.0
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_count_with_external_session(self):
        store = PgVectorStore(table_name="policy_chunks", id_field="id", embedding_dim=1536)
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 42
        mock_session.execute.return_value = mock_result
        count = await store.count(session=mock_session)
        assert count == 42

    def test_upsert_metadata_key_validation(self):
        store = PgVectorStore(table_name="policy_chunks", id_field="id", embedding_dim=1536)
        with pytest.raises(ValueError, match="Unsafe"):
            asyncio.run(
                store.upsert(
                    documents=[
                        {
                            "id": "1",
                            "text": "test",
                            "embedding": [0.0] * 1536,
                            "metadata": {"bad-key": "value"},
                        }
                    ]
                )
            )

    @pytest.mark.asyncio
    async def test_upsert_external_session(self):
        store = PgVectorStore(table_name="policy_chunks", id_field="id", embedding_dim=1536)
        mock_session = AsyncMock()
        await store.upsert(
            documents=[{"id": "1", "text": "test", "embedding": [0.0] * 1536}],
            session=mock_session,
        )
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_exception_logged(self):
        factory = MagicMock()
        factory.return_value.__aenter__.side_effect = Exception("DB down")
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            session_factory=factory,
        )
        with pytest.raises(Exception, match="DB down"):
            await store.upsert(documents=[{"id": "1", "text": "test", "embedding": [0.0] * 1536}])

    def test_upsert_metadata_key_validation_in_loop(self):
        store = PgVectorStore(table_name="policy_chunks", id_field="id", embedding_dim=1536)
        with pytest.raises(ValueError, match="Unsafe"):
            asyncio.run(
                store.upsert(
                    documents=[
                        {
                            "id": "1",
                            "text": "test",
                            "embedding": [0.0] * 1536,
                            "metadata": {"bad-key": "value", "another-bad": "x"},
                        }
                    ]
                )
            )

    def test_upsert_with_extra_fields(self):
        mock_session = AsyncMock()
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = mock_session
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            session_factory=factory,
        )
        asyncio.run(
            store.upsert(
                documents=[
                    {
                        "id": "1",
                        "text": "test",
                        "embedding": [0.0] * 1536,
                        "metadata": {"section": "Section 1"},
                    }
                ],
                extra_fields={"owner_id": "user-1"},
            )
        )
        mock_session.execute.assert_called_once()

    def test_embedding_dim_required(self):
        with pytest.raises(ValueError, match="embedding_dim is required"):
            PgVectorStore(
                table_name="policy_chunks",
                id_field="id",
                session_factory=object(),
                embedding_dim=None,
            )

    def test_tsquery_config_must_be_a_lowercase_identifier(self):
        with pytest.raises(ValueError, match="tsquery_config must be a lowercase"):
            PgVectorStore(
                table_name="policy_chunks",
                id_field="id",
                embedding_dim=1536,
                tsquery_config="English",
            )

    def test_resolve_session_factory_raises_without_factory_or_settings(self):
        store = PgVectorStore(table_name="policy_chunks", id_field="id", embedding_dim=1536)
        with pytest.raises(RuntimeError, match="host-provided session_factory"):
            store._resolve_session_factory()

    def test_settings_database_url_never_creates_an_internal_engine(self):
        settings = SimpleNamespace(database_url="sqlite+aiosqlite:///:memory:")
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            settings=settings,
        )
        with pytest.raises(RuntimeError, match="host-provided session_factory"):
            store._resolve_session_factory()

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_embedding_without_embedding_fn_raises(self):
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            session_factory=MagicMock(),
        )
        with pytest.raises(ValueError, match="embedding_fn is required"):
            await store.hybrid_search(
                query="test",
                embedding=[],
                n_results=5,
                threshold=1.0,
            )

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_embedding_uses_injected_embedding_fn(self):
        embed_fn = AsyncMock(return_value=[[0.1] * 1536])
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = mock_session
        store = PgVectorStore(
            table_name="policy_chunks",
            id_field="id",
            embedding_dim=1536,
            session_factory=factory,
            embedding_fn=embed_fn,
        )
        result = await store.hybrid_search(
            query="test",
            embedding=[],
            n_results=5,
            threshold=1.0,
        )
        assert result == []
        embed_fn.assert_awaited_once_with(["test"])
