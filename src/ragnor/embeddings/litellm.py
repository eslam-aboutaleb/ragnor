"""LiteLLM embedding function.

Wraps ``litellm.aembedding`` so any LiteLLM-supported
provider can serve embeddings. Hosts can opt into a content-addressed
LRU cache and deterministic batch splitting; both default to the
historical single-call, no-cache behavior.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AsyncOpenAI


def _cache_key(model_name: str, text: str) -> str:
    """Return the content-addressed cache key for a model/text pair."""
    digest = hashlib.sha256(f"{model_name}\0{text}".encode())
    return digest.hexdigest()


class LitellmEmbeddingFunction:
    """Callable embedding function backed by LiteLLM.

    Uses the provider's asynchronous embedding path. For native async
    transports, cancellation reaches the in-flight request instead of
    abandoning a blocking request in an SDK executor thread.
    """

    def __init__(  # noqa: PLR0913
        self,
        api_key: str,
        model_name: str,
        api_base: str | None = None,
        *,
        cache_max_entries: int = 0,
        max_batch_size: int | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ) -> None:
        """Initialize the embedding function.

        Args:
            api_key: API key forwarded to the provider.
            model_name: Embedding model name forwarded to litellm.
            api_base: Optional provider base URL override.
            cache_max_entries: Number of vectors kept in the bounded
                LRU cache. ``0`` disables caching (the default).
            max_batch_size: Maximum number of inputs sent in a single
                ``litellm.aembedding`` call. ``None`` (the default)
                sends every input in one call.
            client: Optional borrowed OpenAI/Azure async client. Configure
                transport, deadlines and retries on the client. The host
                retains ownership and is responsible for closing it.

        Raises:
            ValueError: If ``cache_max_entries`` is negative, or
                ``max_batch_size`` is given and smaller than one.
        """
        if cache_max_entries < 0:
            raise ValueError(f"cache_max_entries must be >= 0, got {cache_max_entries}")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.cache_max_entries = cache_max_entries
        self.max_batch_size = max_batch_size
        self._client = client
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _cache_key(self, text: str) -> str:
        """Return the content-addressed cache key for ``text``."""
        return _cache_key(self.model_name, text)

    def _cache_store(self, key: str, vector: list[float]) -> None:
        """Store a vector, evicting the oldest entries past the bound."""
        self._cache[key] = vector
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)

    async def _embed(self, input: list[str]) -> list[list[float]]:
        """Await the provider's async request, preserving cancellation."""
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self._client is not None:
            kwargs["client"] = self._client
        response = await litellm.aembedding(
            model=self.model_name,
            input=input,
            **kwargs,
        )
        return [item["embedding"] for item in response.data]

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, splitting into ``max_batch_size`` chunks when set."""
        if self.max_batch_size is None:
            return await self._embed(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            chunk = texts[start : start + self.max_batch_size]
            chunk_vectors = await self._embed(chunk)
            vectors.extend(chunk_vectors)
        return vectors

    async def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of text strings asynchronously.

        With caching enabled, only cache misses are embedded and the
        results are returned in the original input order. With caching
        disabled, the whole input is embedded in a single
        ``litellm.aembedding`` call (or one call per
        ``max_batch_size`` chunk when batch splitting is enabled).

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats) in the
            same order as ``input``. Every returned vector is a fresh
            copy, so mutating one cannot poison the cache.
        """
        if self.cache_max_entries == 0:
            return await self._embed_batched(input)

        cache_keys = [self._cache_key(text) for text in input]
        results: dict[int, list[float]] = {}
        misses: list[int] = []
        for index, key in enumerate(cache_keys):
            cached = self._cache.get(key)
            if cached is None:
                misses.append(index)
            else:
                self._cache.move_to_end(key)
                results[index] = list(cached)

        if misses:
            miss_texts = [input[index] for index in misses]
            miss_vectors = await self._embed_batched(miss_texts)
            for index, vector in zip(misses, miss_vectors, strict=True):
                self._cache_store(cache_keys[index], vector)
                results[index] = list(vector)

        return [results[index] for index in range(len(input))]

    async def embed_query(self, input: str | list[str]) -> list[float]:
        """Embed a single query string asynchronously.

        Args:
            input: Query string or list containing one string.

        Returns:
            A single embedding vector as a list of floats.
        """
        if isinstance(input, str):
            input = [input]
        result = await self(input)
        return result[0]

    async def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed a list of document strings asynchronously.

        Args:
            input: List of document text strings.

        Returns:
            List of embedding vectors.
        """
        return await self(input)
