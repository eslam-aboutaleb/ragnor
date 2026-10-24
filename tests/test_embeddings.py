"""Tests for ragnor.embeddings.

Covers the LiteLLM embedding function (with
``litellm.aembedding`` mocked), the settings-driven provider
factory, the provider registry, and the dimension registry
including its warn-and-fallback path.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.dimensions import (
    FALLBACK_DIMENSION,
    get_embedding_dimension,
    register_dimension,
)
from ragnor.embeddings.litellm import LitellmEmbeddingFunction
from ragnor.embeddings.registry import (
    LiteLLMEmbeddingProvider,
    embedding_registry,
    get_embedding_function,
)


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str = "test-key",
        openai_api_base: str = "",
        embedding_provider: str = "litellm",
    ) -> None:
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.embedding_provider = embedding_provider


def _mock_response(embeddings: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": embedding} for embedding in embeddings]
    return response


class TestLitellmEmbeddingFunction:
    async def test_call_embeds_inputs_via_litellm(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="test-key", model_name="text-embedding-3-small")
        response = _mock_response([[0.1, 0.2], [0.3, 0.4]])

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding", return_value=response
        ) as mock_embed:
            result = await fn(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embed.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello", "world"],
            api_key="test-key",
        )

    async def test_call_forwards_api_base(self) -> None:
        fn = LitellmEmbeddingFunction(
            api_key="test-key",
            model_name="text-embedding-3-small",
            api_base="https://example.com/v1",
        )
        response = _mock_response([[0.1, 0.2]])

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding", return_value=response
        ) as mock_embed:
            result = await fn(["hello"])

        assert result == [[0.1, 0.2]]
        mock_embed.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello"],
            api_key="test-key",
            api_base="https://example.com/v1",
        )

    async def test_embed_query_accepts_a_string(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")
        response = _mock_response([[0.5, 0.6]])

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding", return_value=response
        ) as mock_embed:
            result = await fn.embed_query("hello")

        assert result == [0.5, 0.6]
        assert mock_embed.call_args.kwargs["input"] == ["hello"]

    async def test_embed_query_accepts_a_single_element_list(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")
        response = _mock_response([[0.7, 0.8]])

        with patch("ragnor.embeddings.litellm.litellm.aembedding", return_value=response):
            result = await fn.embed_query(["hello"])

        assert result == [0.7, 0.8]

    async def test_embed_documents_delegates_to_call(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")
        response = _mock_response([[0.1, 0.2], [0.3, 0.4]])

        with patch("ragnor.embeddings.litellm.litellm.aembedding", return_value=response):
            result = await fn.embed_documents(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_satisfies_the_embedding_function_protocol(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")
        assert isinstance(fn, EmbeddingFunction)


class TestGetEmbeddingFunction:
    def test_reads_model_and_key_from_settings(self) -> None:
        settings = _FakeSettings(embedding_model="custom-model", openai_api_key="key-1")

        fn = get_embedding_function(settings)

        assert isinstance(fn, LitellmEmbeddingFunction)
        assert fn.model_name == "custom-model"
        assert fn.api_key == "key-1"

    def test_defaults_to_litellm_when_settings_lacks_the_provider_key(self) -> None:
        settings = _FakeSettings()
        del settings.embedding_provider

        fn = get_embedding_function(settings)

        assert isinstance(fn, LitellmEmbeddingFunction)

    def test_unknown_provider_raises_key_error_listing_registered(self) -> None:
        settings = _FakeSettings(embedding_provider="no-such-provider")

        with pytest.raises(KeyError, match="no-such-provider"):
            get_embedding_function(settings)

    def test_litellm_provider_is_registered_case_insensitively(self) -> None:
        assert embedding_registry.get("litellm") is LiteLLMEmbeddingProvider
        assert embedding_registry.get("LiteLLM") is LiteLLMEmbeddingProvider

    def test_provider_factory_wraps_the_litellm_function(self) -> None:
        settings = _FakeSettings(embedding_model="m", openai_api_key="k")

        fn = LiteLLMEmbeddingProvider(settings).create()

        assert isinstance(fn, LitellmEmbeddingFunction)
        assert fn.model_name == "m"


class TestEmbeddingDimensions:
    def test_known_models_resolve_to_their_dimensions(self) -> None:
        assert get_embedding_dimension("text-embedding-3-small") == 1536
        assert get_embedding_dimension("text-embedding-3-large") == 3072
        assert get_embedding_dimension("text-embedding-ada-002") == 1536

    def test_explicit_model_overrides_the_configured_default(self) -> None:
        assert get_embedding_dimension("text-embedding-3-large") == 3072

    def test_unknown_model_warns_and_falls_back(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="ragnor.embeddings.dimensions"):
            dimension = get_embedding_dimension("brand-new-model")

        assert dimension == FALLBACK_DIMENSION == 1536
        assert any(
            "No embedding dimension registered for model 'brand-new-model'" in record.message
            for record in caplog.records
        )

    def test_bare_call_warns_and_falls_back(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="ragnor.embeddings.dimensions"):
            assert get_embedding_dimension() == 1536

    def test_register_dimension_extends_the_table(self) -> None:
        register_dimension("ragnor-test-model", 768)

        assert get_embedding_dimension("ragnor-test-model") == 768
