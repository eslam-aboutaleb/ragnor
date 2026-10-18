"""Retrieval evaluation harness for RAG systems.

Provides labeled test sets
and metrics for evaluating retrieval quality (Recall@K,
Precision@K, MRR) and retrieval latency. The harness is
retrieval-function agnostic: any async callable mapping a
query string to a list of result dicts works.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    total_queries: int = 0
    errors: int = 0


@dataclass
class LabeledQuery:
    """A test query with expected relevant chunk IDs."""

    query: str
    expected_chunk_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class RagEvaluationHarness:
    """Evaluation harness for RAG retrieval quality."""

    def __init__(self, test_set: list[LabeledQuery] | None = None):
        self.test_set = test_set or []

    def add_query(self, query: LabeledQuery) -> None:
        """Add a labeled query to the test set."""
        self.test_set.append(query)

    async def evaluate(self, retrieval_fn: Any) -> RetrievalMetrics:
        """Run the test set against a retrieval function and compute metrics.

        Args:
            retrieval_fn: Async function that takes a query string and returns
                a list of result dicts with at least an ``id`` or ``metadata``
                containing ``chunk_id``.

        Returns:
            RetrievalMetrics with aggregated scores.
        """
        metrics = RetrievalMetrics(total_queries=len(self.test_set))
        latencies: list[float] = []

        for labeled in self.test_set:
            start = time.monotonic()
            try:
                results = await retrieval_fn(labeled.query)
                latency_ms = (time.monotonic() - start) * 1000
                latencies.append(latency_ms)

                retrieved_ids = set()
                for r in results:
                    chunk_id = r.get("id") or r.get("metadata", {}).get("chunk_id")
                    if chunk_id:
                        retrieved_ids.add(chunk_id)

                if labeled.expected_chunk_ids:
                    hit = labeled.expected_chunk_ids & retrieved_ids
                    metrics.recall_at_k += len(hit) / len(labeled.expected_chunk_ids)
                    metrics.precision_at_k += len(hit) / max(len(retrieved_ids), 1)

                    for rank, result in enumerate(results, 1):
                        cid = result.get("id") or result.get("metadata", {}).get("chunk_id")
                        if cid in labeled.expected_chunk_ids:
                            metrics.mrr += 1.0 / rank
                            break
            except Exception as exc:
                logger.error("Evaluation query failed: %s", exc)
                metrics.errors += 1

        if metrics.total_queries > 0:
            metrics.recall_at_k /= metrics.total_queries
            metrics.precision_at_k /= metrics.total_queries
            metrics.mrr /= metrics.total_queries
            metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        return metrics
