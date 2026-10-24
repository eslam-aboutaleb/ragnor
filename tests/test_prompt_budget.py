"""Provider-attested full prompt budget accounting, including message overhead."""

from __future__ import annotations

import pytest

from ragnor.evidence.prompt_budget import (
    PromptBudgetError,
    PromptBudgetPlanner,
)


class ExactProviderCounter:
    is_exact = True
    model_id = "example-model-rev1"
    name = "host-provider-wire-protocol-rev1"

    def count_prompt_tokens(self, messages) -> int:
        # The provider adapter owns framing, roles, tools and multimodal accounting.
        return sum(4 + len(str(message.get("content", ""))) for message in messages) + 3


class ApproximateProviderCounter(ExactProviderCounter):
    is_exact = False


def test_full_prompt_reserves_answer_and_counts_system_user_tool_messages() -> None:
    messages = [
        {"role": "system", "content": "Rule"},
        {"role": "user", "content": "Question"},
        {"role": "tool", "content": "Citation [1]: evidence text"},
    ]
    planner = PromptBudgetPlanner(
        counter=ExactProviderCounter(),
        model_id="example-model-rev1",
        total_context_tokens=100,
        reserved_completion_tokens=12,
    )
    result = planner.assess(messages)
    assert result.prompt_tokens == sum(4 + len(x["content"]) for x in messages) + 3
    assert result.reserved_completion_tokens == 12
    assert result.remaining_tokens == 100 - result.prompt_tokens - 12
    assert result.is_exact
    assert result.counter_name == "host-provider-wire-protocol-rev1"


def test_whole_prompt_must_fit_not_just_evidence_content() -> None:
    planner = PromptBudgetPlanner(
        counter=ExactProviderCounter(),
        model_id="example-model-rev1",
        total_context_tokens=40,
        reserved_completion_tokens=10,
    )
    with pytest.raises(PromptBudgetError, match="exceeds"):
        planner.assess(
            [
                {"role": "system", "content": "Safety rules and system context"},
                {"role": "user", "content": "A user question"},
                {"role": "tool", "content": "[1] Evidence"},
            ]
        )


def test_model_identity_mismatch_is_refused_before_accounting() -> None:
    with pytest.raises(ValueError, match="model"):
        PromptBudgetPlanner(
            counter=ExactProviderCounter(),
            model_id="different-model",
            total_context_tokens=500,
            reserved_completion_tokens=50,
        )


def test_nonexact_counter_cannot_promise_a_hard_context_limit() -> None:
    with pytest.raises(ValueError, match="exact"):
        PromptBudgetPlanner(
            counter=ApproximateProviderCounter(),
            model_id="example-model-rev1",
            total_context_tokens=100,
            reserved_completion_tokens=20,
        )


@pytest.mark.parametrize("context,reserve", [(0, 0), (20, -1), (20, 20), (-1, 0)])
def test_invalid_context_reserves_rejected(context: int, reserve: int) -> None:
    with pytest.raises(ValueError):
        PromptBudgetPlanner(
            counter=ExactProviderCounter(),
            model_id="example-model-rev1",
            total_context_tokens=context,
            reserved_completion_tokens=reserve,
        )


@pytest.mark.parametrize("token_count", [-1, 1.5, True])
def test_malformed_provider_count_fails_closed(token_count) -> None:
    class InvalidCounter(ExactProviderCounter):
        def count_prompt_tokens(self, messages):
            return token_count

    planner = PromptBudgetPlanner(
        counter=InvalidCounter(),
        model_id="example-model-rev1",
        total_context_tokens=100,
        reserved_completion_tokens=10,
    )
    with pytest.raises(PromptBudgetError, match="invalid"):
        planner.assess([{"role": "user", "content": "hi"}])


def test_prompt_message_content_is_never_logged_or_serialized_in_budget_result() -> None:
    secret = "private-key-or-source-data"
    planner = PromptBudgetPlanner(
        counter=ExactProviderCounter(),
        model_id="example-model-rev1",
        total_context_tokens=300,
        reserved_completion_tokens=20,
    )
    result = planner.assess([{"role": "tool", "content": secret}])
    assert secret not in repr(result)
