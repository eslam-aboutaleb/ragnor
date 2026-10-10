"""Coverage tests for ragit.evaluation.runner uncovered paths.

The runner takes its retrieval function and dataset as
injected dependencies, so the tests inject fakes instead
of patching the application's retriever and dataset modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragit.evaluation.dataset import (
    Difficulty,
    EvalSample,
)
from ragit.evaluation.runner import (
    EvalConfig,
    _run_answer_eval,
    _run_retrieval_eval,
    main,
    run_evaluation,
)


def _sample(query: str = "test") -> EvalSample:
    return EvalSample(
        query=query,
        expected_sections=["Section 1"],
        gold_answer="gold answer",
        category="coverage",
        difficulty=Difficulty.EASY,
    )


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_retrieval_eval_exception_in_sample(self):
        async def failing_retrieval_fn(query: str) -> list[dict]:
            raise Exception("search failed")

        metrics, per_sample = await _run_retrieval_eval(
            retrieval_fn=failing_retrieval_fn,
            dataset=[_sample()],
        )
        assert metrics.errors == 1
        assert per_sample[0].get("error") is not None

    @pytest.mark.asyncio
    async def test_run_answer_eval_no_generated_answer(self):
        from ragit.evaluation.answer import evaluate_answer_heuristic

        sample = _sample()
        per_sample = [{"context": "", "chunks_found": 0}]
        with patch(
            "ragit.evaluation.runner.evaluate_answer_heuristic",
            wraps=evaluate_answer_heuristic,
        ):
            metrics, details = await _run_answer_eval(
                dataset=[sample],
                per_sample_retrieval=per_sample,
                judge="heuristic",
            )
        assert metrics.errors == 0

    @pytest.mark.asyncio
    async def test_run_evaluation_no_dataset_returns_early(self):
        result = await run_evaluation(
            EvalConfig(mode="context"),
            retrieval_fn=lambda q: [],
            dataset=[],
        )
        assert result.summary == "No evaluation samples matched the filters."

    @pytest.mark.asyncio
    async def test_run_evaluation_requires_injected_dependencies(self):
        with pytest.raises(ValueError):
            await run_evaluation(EvalConfig())

    @pytest.mark.asyncio
    async def test_run_evaluation_defaults_config_when_none(self):
        async def retrieval_fn(query: str) -> list[dict]:
            return []

        async def answer_fn(query: str, context: str) -> str:
            return "generated answer"

        result = await run_evaluation(
            config=None,
            retrieval_fn=retrieval_fn,
            dataset=[_sample()],
            answer_fn=answer_fn,
        )

        assert result.config["mode"] == "end-to-end"
        assert result.config["judge"] == "heuristic"
        assert result.config["n_results"] == 5

    @pytest.mark.asyncio
    async def test_run_evaluation_retrieval_only_mode(self):
        @dataclass
        class FakeMetrics:
            recall_at_k: float = 1.0
            precision_at_k: float = 1.0
            mrr: float = 1.0
            avg_latency_ms: float = 10.0
            total_queries: int = 1
            errors: int = 0

        with patch("ragit.evaluation.runner._run_retrieval_eval") as mock_retrieval:
            mock_retrieval.return_value = (FakeMetrics(), [{}])
            result = await run_evaluation(
                EvalConfig(mode="retrieval-only"),
                retrieval_fn=lambda q: [],
                dataset=[_sample()],
            )
        assert result.retrieval_metrics["recall_at_k"] == 1.0

    @pytest.mark.asyncio
    async def test_run_answer_eval_llm_judge(self):
        from ragit.evaluation.answer import AnswerScore

        sample = _sample()
        per_sample = [{"context": "some context", "chunks_found": 1}]

        llm_score = AnswerScore(
            faithfulness=0.9,
            relevance=0.8,
            completeness=0.7,
            conciseness=0.6,
            overall=0.75,
            reasoning="llm",
        )

        with patch(
            "ragit.evaluation.runner.evaluate_answer_llm",
            new_callable=AsyncMock,
            return_value=llm_score,
        ):
            metrics, details = await _run_answer_eval(
                dataset=[sample],
                per_sample_retrieval=per_sample,
                judge="llm",
                settings=MagicMock(llm_model="openai/gpt-4o-mini"),
            )
        assert metrics.errors == 0
        assert metrics.avg_faithfulness == 0.9

    @pytest.mark.asyncio
    async def test_run_answer_eval_exception_in_sample(self):
        sample = _sample()
        per_sample = [{"context": "", "chunks_found": 0}]

        with patch(
            "ragit.evaluation.runner.evaluate_answer_heuristic",
            side_effect=Exception("eval failed"),
        ):
            metrics, details = await _run_answer_eval(
                dataset=[sample],
                per_sample_retrieval=per_sample,
                judge="heuristic",
            )
        assert metrics.errors == 1
        assert details[0].get("error") is not None

    def test_eval_runner_main_cli(self):
        with patch("sys.argv", ["eval_runner", "--mode", "retrieval-only"]):
            with patch(
                "ragit.evaluation.runner.run_evaluation",
                new_callable=AsyncMock,
            ) as mock_run:

                @dataclass
                class FakeResult:
                    summary: str = "done"
                    per_sample_results: list = None

                    def __post_init__(self):
                        if self.per_sample_results is None:
                            self.per_sample_results = []

                mock_run.return_value = FakeResult()
                main()

    def test_eval_runner_main_cli_with_output(self, tmp_path):
        output_file = tmp_path / "result.json"
        with patch(
            "sys.argv",
            [
                "eval_runner",
                "--mode",
                "retrieval-only",
                "--output",
                str(output_file),
            ],
        ):
            with patch(
                "ragit.evaluation.runner.run_evaluation",
                new_callable=AsyncMock,
            ) as mock_run:

                @dataclass
                class FakeResult:
                    summary: str = "done"
                    per_sample_results: list = None

                    def __post_init__(self):
                        if self.per_sample_results is None:
                            self.per_sample_results = []

                mock_run.return_value = FakeResult()
                main()
        assert output_file.exists()

    def test_eval_runner_main_output_writes_context_pop(self, tmp_path):
        output_file = tmp_path / "eval_result.json"
        with patch(
            "sys.argv",
            [
                "eval_runner",
                "--mode",
                "retrieval-only",
                "--output",
                str(output_file),
            ],
        ):
            with patch(
                "ragit.evaluation.runner.run_evaluation",
                new_callable=AsyncMock,
            ) as mock_run:

                @dataclass
                class FakeResult:
                    summary: str = "done"
                    per_sample_results: list = None

                    def __post_init__(self):
                        if self.per_sample_results is None:
                            self.per_sample_results = [{"context": "some context", "query": "test"}]

                mock_run.return_value = FakeResult()
                main()
        assert output_file.exists()
        import json

        data = json.loads(output_file.read_text())
        for sample in data.get("per_sample_results", []):
            assert "context" not in sample
