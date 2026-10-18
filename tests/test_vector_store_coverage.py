"""Coverage tests for the ragnor vector store registry.

The if/else provider chain is replaced by the ragnor store
registry, so an unsupported provider surfaces as the
registry's ``KeyError``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragnor.stores import InMemoryVectorStore, PgVectorStore, get_vector_store


class TestVectorStore:
    def test_get_vector_store_unsupported_provider(self):
        settings = SimpleNamespace(vector_store_provider="unsupported")
        with pytest.raises(KeyError, match="Unknown provider"):
            get_vector_store(table_name="test", id_field="id", settings=settings)

    def test_get_vector_store_defaults_to_memory(self):
        store = get_vector_store(table_name="test", id_field="id", embedding_dim=1536)
        assert isinstance(store, InMemoryVectorStore)

    def test_get_vector_store_memory_provider(self):
        settings = SimpleNamespace(vector_store_provider="memory")
        store = get_vector_store(table_name="test", id_field="id", settings=settings)
        assert isinstance(store, InMemoryVectorStore)

    def test_get_vector_store_provider_lookup_is_case_insensitive(self):
        settings = SimpleNamespace(vector_store_provider="PgVector")
        store = get_vector_store(
            table_name="test", id_field="id", embedding_dim=1536, settings=settings
        )
        assert isinstance(store, PgVectorStore)
