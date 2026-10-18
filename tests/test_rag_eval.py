"""
Tests for the RAG evaluation subsystem (ragnor):
  - Heuristic & LLM-as-judge answer evaluation
  - RAG evaluation harness & metrics
  - Evaluation runner (retrieval-only & end-to-end)

The dataset tests that exercise curated sample data stay
in the host application; here the runner is driven with
an injected retrieval function and a synthetic dataset.
The HTTP endpoint tests also stay in the host application.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragnor.evaluation.answer import (
    _key_fact_recall,
    _normalize_text,
    _token_overlap,
    evaluate_answer_heuristic,
    evaluate_answer_llm,
)
from ragnor.evaluation.dataset import (
    Difficulty,
    EvalSample,
)
from ragnor.evaluation.retrieval import (
    LabeledQuery,
    RagEvaluationHarness,
    RetrievalMetrics,
)
from ragnor.evaluation.runner import (
    EvalConfig,
    run_evaluation,
)


def _dataset() -> list[EvalSample]:
    """A small synthetic dataset shaped like the curated one."""
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
        ),
        EvalSample(
            query="What is covered under personal property protection?",
            expected_sections=["Section 2: Personal Property Protection"],
            gold_answer=("Electronics, furniture, and jewelry are covered up to $10,000 total."),
            category="coverage",
            difficulty=Difficulty.EASY,
        ),
    ]


def _settings() -> MagicMock:
    return MagicMock(llm_model="openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# Answer evaluator tests (heuristic)
# ---------------------------------------------------------------------------


def test_normalize_text():
    raw = "  Hello, WORLD!! 123...  "
    norm = _normalize_text(raw)
    assert norm == "hello world 123"


def test_token_overlap():
    text_a = "Water damage from pipe burst"
    text_b = "Sudden pipe burst causing water damage"
    overlap = _token_overlap(text_a, text_b)
    assert overlap > 0.5

    assert _token_overlap("", "text") == 0.0
    assert _token_overlap("abc", "xyz") == 0.0


def test_key_fact_recall():
    gold = "Water damage is covered up to $25,000 with a $500 deductible."
    answer_good = "Sudden pipe burst is covered up to $25,000 with $500 deductible."
    score = _key_fact_recall(answer_good, gold)
    assert score >= 0.8

    answer_poor = "It is covered."
    score_poor = _key_fact_recall(answer_poor, gold)
    assert score_poor < score


def test_evaluate_answer_heuristic_empty():
    score = evaluate_answer_heuristic("query", "", "gold")
    assert score.overall == 0.0
    assert "Empty answer" in score.reasoning


def test_evaluate_answer_heuristic_good_match():
    query = "Is water damage from a pipe burst covered?"
    gold = "Yes, water damage caused by sudden pipe bursts is covered up to $25,000 with a $500 deductible."
    generated = (
        "According to Section 1, water damage caused by sudden pipe bursts "
        "is covered up to $25,000 with a $500 deductible."
    )
    context = (
        "## Section 1: Home Water Damage Coverage\n"
        "Water damage caused by sudden pipe bursts is covered up to $25,000 with a $500 deductible."
    )

    score = evaluate_answer_heuristic(
        query=query,
        answer=generated,
        gold_answer=gold,
        retrieved_context=context,
    )

    assert score.faithfulness > 0.4
    assert score.relevance > 0.3
    assert score.completeness > 0.6
    assert score.overall > 0.5


# ---------------------------------------------------------------------------
# Answer evaluator tests (LLM judge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_answer_llm_success():
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "faithfulness": 0.95,
            "relevance": 0.90,
            "completeness": 0.85,
            "conciseness": 1.0,
            "reasoning": "Accurate and grounded in context.",
        }
    )
    mock_response.choices = [mock_choice]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_response

        score = await evaluate_answer_llm(
            query="Is water damage covered?",
            answer="Yes, sudden pipe bursts up to $25,000.",
            gold_answer="Yes, covered up to $25,000.",
            retrieved_context="Policy covers pipe bursts up to $25,000.",
            settings=_settings(),
        )

        assert score.faithfulness == 0.95
        assert score.relevance == 0.90
        assert score.completeness == 0.85
        assert score.overall > 0.8
        assert "Accurate" in score.reasoning


@pytest.mark.asyncio
async def test_evaluate_answer_llm_fallback_on_error():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = RuntimeError("LiteLLM connection error")

        score = await evaluate_answer_llm(
            query="Is water damage covered?",
            answer="Yes, sudden pipe bursts up to $25,000 with $500 deductible.",
            gold_answer="Yes, sudden pipe bursts up to $25,000 with $500 deductible.",
            retrieved_context="Policy covers pipe bursts up to $25,000.",
            settings=_settings(),
        )

        assert score.reasoning.startswith("llm_judge_unavailable")
        assert "heuristic" in score.reasoning
        assert score.overall > 0.0


@pytest.mark.asyncio
async def test_evaluate_answer_llm_fallback_without_settings():
    """A judge without settings cannot run and falls back to the heuristic."""
    score = await evaluate_answer_llm(
        query="Is water damage covered?",
        answer="Yes, sudden pipe bursts up to $25,000.",
        gold_answer="Yes, covered up to $25,000.",
        retrieved_context="Policy covers pipe bursts up to $25,000.",
    )
    assert score.reasoning.startswith("llm_judge_unavailable")
    assert score.overall > 0.0


@pytest.mark.asyncio
async def test_evaluate_answer_llm_fallback_without_llm_model():
    """Settings without an llm_model cannot run the judge and fall back."""
    settings = SimpleNamespace()  # no llm_model attribute

    score = await evaluate_answer_llm(
        query="Is water damage covered?",
        answer="Yes, sudden pipe bursts up to $25,000.",
        gold_answer="Yes, covered up to $25,000.",
        retrieved_context="Policy covers pipe bursts up to $25,000.",
        settings=settings,
    )

    assert score.reasoning.startswith("llm_judge_unavailable (ValueError)")
    assert score.overall > 0.0


# ---------------------------------------------------------------------------
# Harness tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_evaluation_harness():
    harness = RagEvaluationHarness()
    harness.add_query(
        LabeledQuery(
            query="water damage",
            expected_chunk_ids={"chunk_1"},
        )
    )
    harness.add_query(
        LabeledQuery(
            query="jewelry limits",
            expected_chunk_ids={"chunk_2"},
        )
    )

    async def mock_retrieval(query: str):
        if "water" in query:
            return [{"id": "chunk_1", "document": "water text"}]
        return [{"id": "chunk_wrong", "document": "wrong"}]

    metrics: RetrievalMetrics = await harness.evaluate(mock_retrieval)
    assert metrics.total_queries == 2
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.5
    assert metrics.mrr == 0.5
    assert metrics.errors == 0


# ---------------------------------------------------------------------------
# Runner tests (injected retrieval function and dataset)
# ---------------------------------------------------------------------------


def _hybrid_results(section: str, document: str) -> list[dict]:
    return [
        {
            "document": document,
            "metadata": {
                "section": section,
                "source": "sample_policy.md",
            },
            "distance": 0.2,
            "_rrf_score": 0.95,
        }
    ]


@pytest.mark.asyncio
async def test_run_evaluation_retrieval_only():
    async def retrieval_fn(query: str) -> list[dict]:
        if "water" in query:
            return _hybrid_results(
                "Section 1: Home Water Damage Coverage",
                "Water damage caused by sudden pipe bursts is covered up to $25,000.",
            )
        return _hybrid_results(
            "Section 2: Personal Property Protection",
            "Electronics, furniture, and jewelry are covered up to $10,000 total.",
        )

    config = EvalConfig(
        mode="retrieval-only",
        category="coverage",
        difficulty=Difficulty.EASY,
    )

    result = await run_evaluation(config, retrieval_fn, _dataset())

    assert result.duration_seconds >= 0.0
    assert "Retrieval Metrics:" in result.summary
    assert "Answer Quality Metrics:" not in result.summary
    assert result.retrieval_metrics["recall_at_k"] > 0
    assert len(result.per_sample_results) > 0


@pytest.mark.asyncio
async def test_run_evaluation_context_mode_heuristic():
    async def retrieval_fn(query: str) -> list[dict]:
        if "water" in query:
            return _hybrid_results(
                "Section 1: Home Water Damage Coverage",
                "Water damage caused by sudden pipe bursts is covered up to $25,000 with a $500 deductible.",
            )
        return _hybrid_results(
            "Section 2: Personal Property Protection",
            "Electronics, furniture, and jewelry are covered up to $10,000 total.",
        )

    config = EvalConfig(
        mode="context",
        judge="heuristic",
        category="coverage",
    )

    result = await run_evaluation(config, retrieval_fn, _dataset())

    assert "Answer Quality Metrics:" in result.summary
    assert result.answer_metrics["avg_overall"] > 0.0
    assert len(result.per_sample_results) > 0
    assert "answer_faithfulness" in result.per_sample_results[0]


@pytest.mark.asyncio
async def test_run_evaluation_answer_metrics_reuse_retrieval_context():
    """Answer metrics are computed from the retrieval phase's context.

    The runner must not re-retrieve: the retrieval function counts
    its invocations, and a context-mode run over N samples calls it
    exactly N times (once per sample, in the retrieval phase only).
    """
    calls: list[str] = []

    async def retrieval_fn(query: str) -> list[dict]:
        calls.append(query)
        return _hybrid_results(
            "Section 1: Home Water Damage Coverage",
            "Water damage caused by sudden pipe bursts is covered up to $25,000 with a $500 deductible.",
        )

    dataset = _dataset()
    await run_evaluation(
        EvalConfig(mode="context", judge="heuristic"),
        retrieval_fn,
        dataset,
    )

    assert calls == [sample.query for sample in dataset]
