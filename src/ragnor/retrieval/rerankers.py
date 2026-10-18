"""Optional reranker integration implementing the existing Reranker seam.

Provides :class:`LiteLLMReranker`, an implementation of the
:class:`~ragnor.retrieval.strategies.Reranker` protocol that uses
``litellm.rerank`` to re-score candidate documents with cross-encoders
or commercial reranking APIs (e.g. Cohere, Voyage, Jina, etc.).

This module ships behind the ``rerank`` extra and is imported lazily
so ``ragnor.retrieval`` remains completely dependency-free.
"""

from __future__ import annotations

import asyncio
from typing import Any


class LiteLLMReranker:
    """Reranker implementation backed by LiteLLM.

    Implements the :class:`~ragnor.retrieval.strategies.Reranker`
    protocol. Requires the ``rerank`` extra (``pip install 'ragnor[rerank]'``).

    Attributes:
        model: Model identifier passed to ``litellm.rerank`` (e.g.
            ``"cohere/rerank-v3.5"``).
        api_key: Optional provider API key.
        api_base: Optional provider API base URL.
        top_n: Optional limit on the number of reranked results returned.
    """

    def __init__(
        self,
        model: str = "cohere/rerank-v3.5",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        top_n: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.top_n = top_n

    def _rerank_sync(
        self,
        query: str,
        documents: list[str],
    ) -> list[dict[str, Any]]:
        """Synchronous call to litellm.rerank running in worker thread."""
        try:
            import litellm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The LiteLLM reranker requires the 'rerank' extra: pip install 'ragnor[rerank]'"
            ) from exc

        kwargs: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.top_n is not None:
            kwargs["top_n"] = self.top_n

        response = litellm.rerank(**kwargs)
        # Standard litellm rerank response format: response.results or list of dicts
        raw_results = getattr(response, "results", response)
        items: list[dict[str, Any]] = []
        for item in raw_results:
            idx = item.get("index") if isinstance(item, dict) else getattr(item, "index", 0)
            score = (
                item.get("relevance_score")
                if isinstance(item, dict)
                else getattr(item, "relevance_score", 0.0)
            )
            items.append({"index": idx, "score": float(score)})
        return items

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank ``results`` for ``query``, sorted by relevance score descending.

        Args:
            query: Natural language search query.
            results: List of candidate result dicts carrying at least
                a ``"document"`` key.

        Returns:
            Re-ordered list of result dicts, each carrying a
            ``"_reranker_score"`` key with the raw score.
        """
        if not results:
            return []

        doc_texts = [str(r.get("document", "")) for r in results]
        reranked_items = await asyncio.to_thread(self._rerank_sync, query, doc_texts)

        output: list[dict[str, Any]] = []
        for item in reranked_items:
            idx = item["index"]
            score = item["score"]
            if 0 <= idx < len(results):
                updated = dict(results[idx])
                updated["_reranker_score"] = score
                output.append(updated)

        return output
