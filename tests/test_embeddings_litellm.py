"""Tests for the LiteLLM embedding function cache and batching.

Covers the ``cache_max_entries`` LRU cache (hit, eviction, mixed
hit/miss ordering), the ``max_batch_size`` splitting, the unchanged
defaults (single call, no cache), the ``embed_query`` /
``embed_documents`` surface, and constructor validation.
``litellm.aembedding`` is mocked so no network call happens.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ragnor.embeddings.litellm import LitellmEmbeddingFunction, _cache_key


def _mock_response(embeddings: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": embedding} for embedding in embeddings]
    return response


def _expected_vector(text: str) -> list[float]:
    """The canned vector the fake litellm pair returns for ``text``."""
    return [float(sum(ord(character) for character in text)), float(len(text))]


def _recording_embedding(
    calls: list[tuple[list[str], dict[str, Any]]],
) -> Callable[..., Awaitable[MagicMock]]:
    """Build a fake ``litellm.aembedding`` recording inputs and kwargs."""

    async def _fake_embedding(**kwargs: Any) -> MagicMock:
        batch = list(kwargs["input"])
        calls.append((batch, kwargs))
        return _mock_response([_expected_vector(text) for text in batch])

    return _fake_embedding


class TestLitellmEmbeddingFunctionDefaults:
    async def test_default_makes_one_call_per_invocation(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="test-key", model_name="text-embedding-3-small")
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ) as mock_embed:
            first = await fn(["hello", "world"])
            second = await fn(["hello", "world"])

        assert [batch for batch, _ in calls] == [["hello", "world"], ["hello", "world"]]
        assert mock_embed.call_count == 2
        assert (
            first
            == second
            == [
                _expected_vector("hello"),
                _expected_vector("world"),
            ]
        )

    async def test_default_forwards_api_base(self) -> None:
        fn = LitellmEmbeddingFunction(
            api_key="test-key",
            model_name="text-embedding-3-small",
            api_base="https://example.com/v1",
        )
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ):
            await fn(["hello"])

        assert [batch for batch, _ in calls] == [["hello"]]
        assert calls[0][1]["api_base"] == "https://example.com/v1"
        assert calls[0][1]["model"] == "text-embedding-3-small"
        assert calls[0][1]["api_key"] == "test-key"

    async def test_embed_query_returns_a_single_vector(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")

        with patch("ragnor.embeddings.litellm.litellm.aembedding") as mock_embed:
            mock_embed.return_value = _mock_response([_expected_vector("hello")])
            result = await fn.embed_query("hello")

        assert result == _expected_vector("hello")
        assert mock_embed.call_args.kwargs["input"] == ["hello"]

    async def test_embed_query_accepts_a_single_element_list(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")

        with patch("ragnor.embeddings.litellm.litellm.aembedding") as mock_embed:
            mock_embed.return_value = _mock_response([_expected_vector("hello")])
            result = await fn.embed_query(["hello"])

        assert result == _expected_vector("hello")

    async def test_embed_documents_returns_one_vector_per_document(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m")

        with patch("ragnor.embeddings.litellm.litellm.aembedding") as mock_embed:
            mock_embed.return_value = _mock_response([_expected_vector("a"), _expected_vector("b")])
            result = await fn.embed_documents(["a", "b"])

        assert result == [_expected_vector("a"), _expected_vector("b")]


class TestLitellmEmbeddingFunctionCache:
    async def test_second_identical_call_skips_litellm(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m", cache_max_entries=8)
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ) as mock_embed:
            first = await fn(["alpha", "beta"])
            second = await fn(["alpha", "beta"])

        assert mock_embed.call_count == 1
        assert [batch for batch, _ in calls] == [["alpha", "beta"]]
        assert (
            second
            == first
            == [
                _expected_vector("alpha"),
                _expected_vector("beta"),
            ]
        )

    async def test_cache_evicts_the_oldest_entry(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m", cache_max_entries=2)
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ) as mock_embed:
            await fn(["alpha"])
            await fn(["beta"])
            await fn(["gamma"])
            evicted_again = await fn(["alpha"])

        assert mock_embed.call_count == 4
        assert evicted_again == [_expected_vector("alpha")]

    async def test_mixed_hits_and_misses_preserve_input_order(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m", cache_max_entries=8)
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ):
            await fn(["alpha", "omega"])
            calls.clear()
            result = await fn(["alpha", "nova", "omega"])

        assert [batch for batch, _ in calls] == [["nova"]]
        assert result == [
            _expected_vector("alpha"),
            _expected_vector("nova"),
            _expected_vector("omega"),
        ]

    def test_cache_key_is_scoped_per_model(self) -> None:
        same_text = "alpha"

        assert _cache_key("model-a", same_text) != _cache_key("model-b", same_text)
        assert _cache_key("model-a", same_text) == _cache_key("model-a", same_text)


class TestLitellmEmbeddingFunctionBatching:
    async def test_max_batch_size_splits_into_ordered_chunks(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m", max_batch_size=2)
        texts = ["t0", "t1", "t2", "t3", "t4"]
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ) as mock_embed:
            result = await fn(texts)

        assert mock_embed.call_count == 3
        assert [batch for batch, _ in calls] == [["t0", "t1"], ["t2", "t3"], ["t4"]]
        assert result == [_expected_vector(text) for text in texts]

    async def test_batching_forwards_api_base_on_every_chunk(self) -> None:
        fn = LitellmEmbeddingFunction(
            api_key="test-key",
            model_name="m",
            api_base="https://example.com/v1",
            max_batch_size=2,
        )
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ):
            await fn(["a", "b", "c"])

        assert [batch for batch, _ in calls] == [["a", "b"], ["c"]]
        assert all(kwargs.get("api_base") == "https://example.com/v1" for _, kwargs in calls)

    async def test_cache_and_batching_combine(self) -> None:
        fn = LitellmEmbeddingFunction(
            api_key="k", model_name="m", cache_max_entries=8, max_batch_size=1
        )
        calls: list[tuple[list[str], dict[str, Any]]] = []

        with patch(
            "ragnor.embeddings.litellm.litellm.aembedding",
            side_effect=_recording_embedding(calls),
        ) as mock_embed:
            await fn(["alpha", "beta"])
            mock_embed.reset_mock()
            calls.clear()
            result = await fn(["beta", "nova"])

        assert [batch for batch, _ in calls] == [["nova"]]
        assert result == [_expected_vector("beta"), _expected_vector("nova")]


class TestLitellmEmbeddingFunctionValidation:
    @pytest.mark.parametrize("cache_max_entries", [-1, -100])
    def test_negative_cache_max_entries_raises(self, cache_max_entries: int) -> None:
        with pytest.raises(ValueError, match="cache_max_entries"):
            LitellmEmbeddingFunction(
                api_key="k", model_name="m", cache_max_entries=cache_max_entries
            )

    @pytest.mark.parametrize("max_batch_size", [0, -1])
    def test_max_batch_size_below_one_raises(self, max_batch_size: int) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            LitellmEmbeddingFunction(api_key="k", model_name="m", max_batch_size=max_batch_size)

    def test_zero_values_are_accepted(self) -> None:
        fn = LitellmEmbeddingFunction(api_key="k", model_name="m", cache_max_entries=0)

        assert fn.cache_max_entries == 0
        assert fn.max_batch_size is None
