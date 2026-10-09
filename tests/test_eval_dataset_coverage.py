"""Coverage tests for the ragit evaluation dataset types.

The dataset **types** live in ``ragit.evaluation.dataset``;
curated sample datasets live in the host application, not
in ragit, and their filtering tests belong to the host
application's own test suite.
"""

from __future__ import annotations

from ragit.evaluation.dataset import (
    Difficulty,
    EvalSample,
)


class TestEvalDatasetTypes:
    def test_eval_sample_defaults(self):
        sample = EvalSample(
            query="Is water damage covered?",
            expected_sections=["Section 1"],
            gold_answer="Yes.",
            category="coverage",
            difficulty=Difficulty.EASY,
        )
        assert sample.metadata == {}
        assert sample.category == "coverage"
        assert sample.difficulty.value == "easy"

    def test_eval_sample_category_accepts_arbitrary_strings(self):
        """Category is a free-form string; hosts define their own taxonomy."""
        categories = ["coverage", "billing", "onboarding"]
        for category in categories:
            sample = EvalSample(
                query="q",
                expected_sections=[],
                gold_answer="g",
                category=category,
                difficulty=Difficulty.EASY,
            )
            assert sample.category == category

    def test_difficulty_values(self):
        assert [d.value for d in Difficulty] == ["easy", "medium", "hard"]

    def test_eval_sample_category_is_a_plain_string(self):
        sample = EvalSample(
            query="q",
            expected_sections=[],
            gold_answer="g",
            category="limits",
            difficulty=Difficulty.HARD,
        )
        assert sample.category == "limits"
        assert sample.difficulty == "hard"
