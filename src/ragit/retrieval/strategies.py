"""Retrieval strategies over storage capability interfaces.

Strategies compose the capability Protocols
(:class:`ragit.stores.base.VectorIndex` and
:class:`ragit.stores.base.LexicalIndex`) instead of
the fused :class:`ragit.stores.base.VectorStore`
contract, so retrieval topologies the fused contract
cannot express become possible:

* :class:`DenseRetrieval` -- vector search only;
* :class:`KeywordRetrieval` -- keyword search only,
  with **no embedding dependency** (a keyword-only
  application, such as low-latency game NPC
  retrieval, needs no embedding-provider
  configuration at all);
* :class:`HybridRetrieval` -- both stages, fused
  with Reciprocal Rank Fusion, optionally reranked.

The two capability interfaces may be different
objects: a pgvector :class:`VectorIndex` plus an
external lexical search engine adapted to
:class:`LexicalIndex` is a first-class composition.

Every strategy accepts ``strict: bool = False`` in
its ``retrieve`` kwargs and applies the legacy
swallow/raise contract (``strict=False`` swallows
failures into ``[]``; ``strict=True`` raises
:class:`ragit.types.RetrievalError`), so failure
behavior is uniform across the legacy and strategy
paths.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ragit.stores._scoring import fuse_hybrid
from ragit.stores.base import EmbeddingFunction, LexicalIndex, VectorIndex
from ragit.types import RetrievalError

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Structural interface for a retrieval strategy."""

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents matching ``query``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Structural interface for an optional reranking stage.

    No built-in implementation ships; hosts inject
    their own reranker (a cross-encoder, an LLM
    reranker, ...) into :class:`HybridRetrieval`.
    """

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank ``results`` for ``query``."""
        ...


class DenseRetrieval:
    """Dense-only retrieval over a :class:`VectorIndex`.

    Embeds the query with the injected embedder and
    calls the index's ``vector_search``.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: EmbeddingFunction,
    ) -> None:
        """Initialize the strategy.

        Args:
            vector_index: The vector index to search.
            embedder: Async callable that embeds a list
                of texts.
        """
        self._vector_index = vector_index
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by vector similarity.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            **kwargs: ``threshold`` (the distance cutoff,
                required), ``text_field``,
                ``metadata_fields``, ``session``, equality
                ``filters``, and ``strict`` (the
                swallow/raise contract).

        Returns:
            list[dict]: Result dicts, or ``[]`` when a
            failure is swallowed.

        Raises:
            ValueError: If ``threshold`` is not supplied.
            ragit.types.RetrievalError: If retrieval
                fails and ``strict=True``.
        """
        strict = kwargs.pop("strict", False)
        threshold = kwargs.pop("threshold", None)
        if threshold is None:
            raise ValueError("dense retrieval requires a distance threshold; pass threshold=...")
        try:
            embeddings = await self._embedder([query])
            return await self._vector_index.vector_search(
                embedding=embeddings[0],
                n_results=n_results,
                threshold=threshold,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Dense retrieval failed: %s", exc)
            if strict:
                raise RetrievalError(f"Dense retrieval failed: {exc}") from exc
            return []


class KeywordRetrieval:
    """Keyword-only retrieval over a :class:`LexicalIndex`.

    Calls the index's ``keyword_search`` -- a
    keyword-only application needs no embedding
    dependency at all.
    """

    def __init__(
        self,
        lexical_index: LexicalIndex,
    ) -> None:
        """Initialize the strategy.

        Args:
            lexical_index: The lexical index to search.
        """
        self._lexical_index = lexical_index

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by keyword search.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            **kwargs: ``text_field``, ``metadata_fields``,
                ``session``, equality ``filters``, and
                ``strict`` (the swallow/raise contract).

        Returns:
            list[dict]: Result dicts, or ``[]`` when a
            failure is swallowed.

        Raises:
            ragit.types.RetrievalError: If retrieval
                fails and ``strict=True``.
        """
        strict = kwargs.pop("strict", False)
        try:
            return await self._lexical_index.keyword_search(
                query=query,
                n_results=n_results,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Keyword retrieval failed: %s", exc)
            if strict:
                raise RetrievalError(f"Keyword retrieval failed: {exc}") from exc
            return []


class HybridRetrieval:
    """Hybrid retrieval over two capability stages, RRF-fused.

    Runs the vector stage and the keyword stage, fuses
    the two truncated result lists with Reciprocal Rank
    Fusion (:func:`ragit.stores._scoring.fuse_hybrid`),
    then applies the optional reranker. The two
    capability interfaces may be different objects --
    for example a pgvector :class:`VectorIndex` plus
    an external lexical engine adapted to
    :class:`LexicalIndex`.
    """

    def __init__(  # noqa: PLR0913
        self,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        embedder: EmbeddingFunction,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            vector_index: The vector index to search.
            lexical_index: The lexical index to search.
            embedder: Async callable that embeds a list
                of texts.
            reranker: Optional reranker applied to the
                fused results.
        """
        self._vector_index = vector_index
        self._lexical_index = lexical_index
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by hybrid search.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            **kwargs: ``threshold`` (the distance cutoff,
                required), ``text_field``,
                ``metadata_fields``, ``session``, equality
                ``filters``, and ``strict`` (the
                swallow/raise contract).

        Returns:
            list[dict]: Fused (and optionally reranked)
            result dicts, or ``[]`` when a failure is
            swallowed.

        Raises:
            ValueError: If ``threshold`` is not supplied.
            ragit.types.RetrievalError: If retrieval
                fails and ``strict=True``.
        """
        strict = kwargs.pop("strict", False)
        threshold = kwargs.pop("threshold", None)
        if threshold is None:
            raise ValueError("hybrid retrieval requires a distance threshold; pass threshold=...")
        try:
            embeddings = await self._embedder([query])
            vector_results = await self._vector_index.vector_search(
                embedding=embeddings[0],
                n_results=n_results,
                threshold=threshold,
                **kwargs,
            )
            keyword_results = await self._lexical_index.keyword_search(
                query=query,
                n_results=n_results,
                **kwargs,
            )
            results = fuse_hybrid(vector_results, keyword_results, n_results)
            if self._reranker is not None:
                results = await self._reranker.rerank(query, results)
            return results
        except Exception as exc:
            logger.error("Hybrid retrieval failed: %s", exc)
            if strict:
                raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc
            return []
