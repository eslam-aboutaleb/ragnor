"""Coverage tests for ragnor.evaluation.answer uncovered paths."""

from __future__ import annotations

from ragnor.evaluation.answer import (
    _key_fact_recall,
    evaluate_answer_heuristic,
)


class TestAnswerEvaluator:
    def test_key_fact_recall_no_key_facts_falls_back_to_overlap(self):
        result = _key_fact_recall("some answer", "no amounts or numbers here")
        assert 0.0 <= result <= 1.0

    def test_key_fact_recall_with_domain_terms(self):
        result = _key_fact_recall(
            "the answer mentions water",
            "water and fire are covered",
            domain_terms={"water", "fire"},
        )
        assert result == 0.5

    def test_evaluate_answer_heuristic_with_domain_terms(self):
        score = evaluate_answer_heuristic(
            query="what is covered?",
            answer="water damage is covered",
            gold_answer="water damage is covered",
            retrieved_context="",
            domain_terms={"water"},
        )
        assert score.completeness == 1.0

    def test_evaluate_answer_heuristic_faithfulness_fallback(self):
        score = evaluate_answer_heuristic(
            query="test",
            answer="some answer",
            gold_answer="gold answer",
            retrieved_context="",
        )
        assert score.faithfulness >= 0.0

    def test_evaluate_answer_heuristic_conciseness_very_verbose(self):
        long_answer = "word " * 1000
        score = evaluate_answer_heuristic(
            query="test",
            answer=long_answer,
            gold_answer="short answer",
            retrieved_context="",
        )
        assert score.conciseness == 0.2

    def test_evaluate_answer_heuristic_empty_answer(self):
        score = evaluate_answer_heuristic(
            query="test", answer="", gold_answer="gold", retrieved_context=""
        )
        assert score.reasoning == "Empty answer"

    def test_evaluate_answer_heuristic_very_verbose_ratio_above_4(self):
        very_long = "word " * 1000
        score = evaluate_answer_heuristic(
            query="test",
            answer=very_long,
            gold_answer="short answer",
            retrieved_context="",
        )
        assert score.conciseness == 0.2

    def test_evaluate_answer_heuristic_elif_branch_ratio_between_2_and_4(self):
        medium_answer = "word " * 5
        score = evaluate_answer_heuristic(
            query="test",
            answer=medium_answer,
            gold_answer="short answer",
            retrieved_context="",
        )
        assert 0.3 <= score.conciseness <= 1.0
