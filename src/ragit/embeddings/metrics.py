"""Embedding metrics: counters and usage accounting.

:class:`EmbeddingMetrics` accumulates observable events from an
:class:`~ragit.embeddings.executor.EmbeddingExecutor` run: provider
calls, cache hits, input size estimates, retries, and any usage
statistics the provider reports.

All token-count fields are explicitly labeled as *estimates* because
the true token count is model-specific and rarely returned by providers.
Callers must not treat these as billing-accurate figures.

Usage::

    metrics = EmbeddingMetrics()
    metrics.embed_calls += 1
    metrics.total_input_tokens_estimate += len(text.split())

    combined = metrics.merge(other_metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingMetrics:
    """Accumulated metrics from one or more embedding operations.

    Attributes:
        embed_calls: Number of provider embedding calls made (after
            cache resolution; does not count cache hits).
        cache_hits: Number of vectors served from the local cache.
        cache_misses: Number of vectors that required a provider call.
        total_input_tokens_estimate: Estimated total input tokens across
            all provider calls.  This is a *rough estimate* (e.g. word
            count × 1.3) and must not be presented as exact billing usage.
        total_retry_count: Total number of retry attempts across all calls
            (not counting the initial attempt).
        provider_usage: Raw usage dict reported by the provider (e.g.
            ``{"prompt_tokens": N, "total_tokens": M}``).  Provider-specific
            and may be empty when the provider does not report usage.
        is_token_estimate: Always ``True``; signals to callers that
            ``total_input_tokens_estimate`` is an estimate, not an exact count.
    """

    embed_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_input_tokens_estimate: int = 0
    total_retry_count: int = 0
    provider_usage: dict[str, Any] = field(default_factory=dict)
    is_token_estimate: bool = True

    def merge(self, other: EmbeddingMetrics) -> EmbeddingMetrics:
        """Return a new :class:`EmbeddingMetrics` combining ``self`` and ``other``.

        ``provider_usage`` values are summed when both sides carry the same
        integer key; non-integer values from ``other`` overwrite ``self``.

        Args:
            other: Metrics to combine with ``self``.

        Returns:
            A fresh :class:`EmbeddingMetrics` instance with summed counters.
        """
        merged_usage: dict[str, Any] = dict(self.provider_usage)
        for key, value in other.provider_usage.items():
            both_int = isinstance(merged_usage.get(key), int) and isinstance(value, int)
            if key in merged_usage and both_int:
                merged_usage[key] = merged_usage[key] + value
            else:
                merged_usage[key] = value

        return EmbeddingMetrics(
            embed_calls=self.embed_calls + other.embed_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            cache_misses=self.cache_misses + other.cache_misses,
            total_input_tokens_estimate=(
                self.total_input_tokens_estimate + other.total_input_tokens_estimate
            ),
            total_retry_count=self.total_retry_count + other.total_retry_count,
            provider_usage=merged_usage,
            is_token_estimate=True,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EmbeddingMetrics(calls={self.embed_calls}, hits={self.cache_hits}, "
            f"misses={self.cache_misses}, retries={self.total_retry_count})"
        )
