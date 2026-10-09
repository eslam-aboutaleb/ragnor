"""LiteLLM embedding function.

Wraps ``litellm.embedding`` so any LiteLLM-supported
provider can serve embeddings.
"""

from __future__ import annotations

import asyncio

import litellm


class LitellmEmbeddingFunction:
    """Callable embedding function backed by LiteLLM.

    Wraps ``litellm.embedding`` so any LiteLLM-supported
    provider can serve embeddings.
    """

    def __init__(self, api_key: str, model_name: str, api_base: str | None = None):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base

    def _embed_sync(self, input: list[str]) -> list[list[float]]:
        """Synchronous embedding call (runs in threadpool)."""
        kwargs = {"api_key": self.api_key}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = litellm.embedding(
            model=self.model_name,
            input=input,
            **kwargs,
        )
        return [item["embedding"] for item in response.data]

    async def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of text strings asynchronously.

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        return await asyncio.to_thread(self._embed_sync, input)

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
