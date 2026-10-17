"""Query transformation and multi-query expansion hooks.

Provides host-injected transformation hooks (e.g. multi-query expansion,
synonym expansion, or query rewriting) with **strict hard limits**:

* **Hard query limits**: maximum number of generated queries is bounded
  by ``max_queries`` (default 5); runaway generators are clamped.
* **Timeout bounds**: transformations are subject to a strict timeout,
  falling back gracefully to the original query on expiry or failure.
* **No mandatory agentic retrieval or LLM reasoning**: this is a clean,
  bounded, host-injected hook. When omitted, retrieval runs the original
  query directly with zero overhead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class QueryTransformer(Protocol):
    """Host-injected query transformation / rewrite seam."""

    async def transform(self, query: str) -> Sequence[str]:
        """Transform or expand ``query`` into one or more search queries."""
        ...


@dataclass
class BoundedQueryTransformer:
    """Wraps an arbitrary :class:`QueryTransformer` with hard safety limits.

    Attributes:
        transformer: The underlying query transformer.
        max_queries: Strict upper limit on the number of queries returned.
        timeout_seconds: Maximum time allowed for query transformation.
        include_original: Whether the original query is guaranteed to be
            included in the output list (defaults to ``True``).
    """

    transformer: QueryTransformer
    max_queries: int = 5
    timeout_seconds: float = 5.0
    include_original: bool = True

    def __post_init__(self) -> None:
        if self.max_queries < 1:
            raise ValueError(f"max_queries must be >= 1, got {self.max_queries}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

    async def transform(self, query: str) -> list[str]:
        """Transform ``query``, enforcing bounds, deduplication, and timeouts.

        Returns at least ``[query]`` on error or timeout, never raising.
        """
        raw_query = query.strip()
        if not raw_query:
            return []

        try:
            expanded = await asyncio.wait_for(
                self.transformer.transform(raw_query),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Query transformer failed or timed out (%.2fs): %s; "
                "falling back to original query.",
                self.timeout_seconds,
                exc,
            )
            return [raw_query]

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []

        if self.include_original:
            seen.add(raw_query.lower())
            result.append(raw_query)

        for q in expanded:
            norm = q.strip()
            if norm and norm.lower() not in seen:
                seen.add(norm.lower())
                result.append(norm)
            if len(result) >= self.max_queries:
                break

        return result if result else [raw_query]
