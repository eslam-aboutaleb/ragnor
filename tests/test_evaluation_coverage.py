"""Coverage tests for ragit.evaluation.retrieval uncovered paths."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from ragit.evaluation import RagEvaluationHarness


class TestEvaluation:
    def test_retrieval_metrics_empty_test_set(self):
        harness = RagEvaluationHarness(test_set=[])
        metrics = asyncio.run(harness.evaluate(lambda q: []))
        assert metrics.total_queries == 0

    def test_evaluation_harness_query_exception_counts_error(self):
        async def failing_retrieval_fn(query):
            raise Exception("search failed")

        harness = RagEvaluationHarness(
            test_set=[MagicMock(query="test", expected_chunk_ids={"c1"})]
        )
        metrics = asyncio.run(harness.evaluate(failing_retrieval_fn))
        assert metrics.errors == 1
