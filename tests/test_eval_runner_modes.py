"""Tests for the evaluation runner's three modes.

Pins the mode contract: ``retrieval-only`` measures retrieval
alone, ``context`` scores the joined retrieved context against
the gold answer (the legacy behavior under an honest name), and
``end-to-end`` judges a real model output produced by an
injected ``answer_fn`` -- never the retrieved context itself.
"""

from __future__ import annotations

import pytest

from ragit.evaluation.dataset import Difficulty, EvalSample
from ragit.evaluation.runner import EvalConfig, run_evaluation


def _dataset() -> list[EvalSample]:
    """A single-sample dataset about water damage coverage."""
    return [
        EvalSample(
            query="Is water damage from a pipe burst covered?",
            expected_sections=["Section 1: Home Water Damage Coverage"],
            gold_answer=(
                "Yes, water damage caused by sudden pipe bursts is covered "
                "up to $25,000 with a $500 deductible."
            ),
            category="coverage",
            difficulty=Difficulty.EASY,
        )
    ]


CONTEXT = (
    "Water damage caused by sudden pipe bursts is covered "
    "up to $25,000 with a $500 deductible."
)


async def _retrieval_fn(query: str) -> list[dict]:
    """Retrieve one chunk whose document is the context."""
    return [
        {
            "document": CONTEXT,
            "metadata": {"section": "Section 1: Home Water Damage Coverage"},
            "distance": 0.2,
            "_rrf_score": 0.95,
        }
    ]


@pytest.mark.asyncio
async def test_end_to_end_without_answer_fn_raises():
    """end-to-end without an answer generator fails loudly."""
    with pytest.raises(ValueError, match="end-to-end"):
        await run_evaluation(
            EvalConfig(mode="end-to-end"),
            _retrieval_fn,
            _dataset(),
        )


@pytest.mark.asyncio
async def test_end_to_end_error_names_all_three_modes():
    """The ValueError names the three modes and the requirement."""
    with pytest.raises(ValueError) as excinfo:
        await run_evaluation(
            EvalConfig(mode="end-to-end"),
            _retrieval_fn,
            _dataset(),
        )
    message = str(excinfo.value)
    for mode in ("retrieval-only", "context", "end-to-end"):
        assert mode in message
    assert "answer generator" in message


@pytest.mark.asyncio
async def test_end_to_end_calls_answer_fn_with_query_and_context():
    """The injected answer generator receives (query, context)."""
    calls: list[tuple[str, str]] = []

    async def answer_fn(query: str, context: str) -> str:
        calls.append((query, context))
        return "Yes, sudden pipe bursts are covered up to $25,000."

    result = await run_evaluation(
        EvalConfig(mode="end-to-end", judge="heuristic"),
        _retrieval_fn,
        _dataset(),
        answer_fn=answer_fn,
    )

    assert calls == [(_dataset()[0].query, CONTEXT)]

    sample = result.per_sample_results[0]
    assert sample["generated_answer"] == (
        "Yes, sudden pipe bursts are covered up to $25,000."
    )
    assert sample["context"] == CONTEXT
    assert sample["gold_answer"] == _dataset()[0].gold_answer
    assert sample["chunks_found"] == 1
    assert "answer_faithfulness" in sample
    assert "answer_relevance" in sample
    assert "answer_overall" in sample


@pytest.mark.asyncio
async def test_end_to_end_unrelated_answer_scores_low_faithfulness():
    """Regression: the retrieved context must not be judged as the answer.

    An answer generator returning text unrelated to the retrieved
    context and the gold answer must score low faithfulness.
    Previously the runner scored the retrieved context against
    itself, so faithfulness could reach 1.0 regardless of what
    any model would have answered.
    """

    async def answer_fn(query: str, context: str) -> str:
        return "The capital of France is Paris and elephants eat peanuts."

    result = await run_evaluation(
        EvalConfig(mode="end-to-end", judge="heuristic"),
        _retrieval_fn,
        _dataset(),
        answer_fn=answer_fn,
    )

    assert result.answer_metrics["avg_faithfulness"] < 0.2


@pytest.mark.asyncio
async def test_context_mode_scores_context_against_gold():
    """context mode preserves the legacy behavior under an honest name.

    The joined retrieved context is the answer being scored, so
    the heuristic faithfulness (token overlap of the context with
    itself) is 1.0 -- exactly what the old end-to-end path did.
    """
    result = await run_evaluation(
        EvalConfig(mode="context", judge="heuristic"),
        _retrieval_fn,
        _dataset(),
    )

    sample = result.per_sample_results[0]
    assert sample["generated_answer"] == CONTEXT
    assert sample["context"] == CONTEXT
    assert sample["gold_answer"] == _dataset()[0].gold_answer
    assert sample["chunks_found"] == 1
    assert sample["answer_faithfulness"] == 1.0
    assert "Answer Quality Metrics:" in result.summary


@pytest.mark.asyncio
async def test_retrieval_only_mode_unchanged():
    """retrieval-only measures retrieval quality and nothing else."""
    result = await run_evaluation(
        EvalConfig(mode="retrieval-only"),
        _retrieval_fn,
        _dataset(),
    )

    assert "Answer Quality Metrics:" not in result.summary
    assert result.answer_metrics == {}
    assert result.retrieval_metrics["recall_at_k"] == 1.0
    assert result.per_sample_results[0]["context"] == CONTEXT
    assert "generated_answer" not in result.per_sample_results[0]


@pytest.mark.asyncio
async def test_invalid_mode_raises():
    """An unknown mode fails loudly instead of silently skipping."""
    with pytest.raises(ValueError, match="invalid evaluation mode"):
        await run_evaluation(
            EvalConfig(mode="bogus"),
            _retrieval_fn,
            _dataset(),
        )
