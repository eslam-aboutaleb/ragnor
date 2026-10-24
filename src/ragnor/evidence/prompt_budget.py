"""Strict full-prompt token accounting delegated to a provider-aware counter.

Unlike an evidence-only token budget, this port counts provider serialization
overhead (roles, tool frames, media and system/user content). Ragnor does not
guess those semantics and will not claim exactness without a host attestation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ragnor.core.exceptions import ConfigurationError


class PromptBudgetError(ConfigurationError):
    """The proposed full prompt cannot safely fit the target model context."""

    default_code = "ERR_PROMPT_BUDGET_EXCEEDED"


@runtime_checkable
class ProviderPromptCounter(Protocol):
    """Host implementation responsible for counting actual provider input wire format.

    The model identifier and tokenization implementation MUST match the
    deployed model/revision. A tokenizer that counts concatenated strings but
    ignores role/tool/image framing is not an exact provider prompt counter.
    """

    name: str
    model_id: str
    is_exact: bool

    def count_prompt_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        """Count the complete provider-specific serialized prompt."""
        raise NotImplementedError


@dataclass(frozen=True)
class PromptBudgetResult:
    """No customer text or credentials are retained in budget diagnostics."""

    prompt_tokens: int
    reserved_completion_tokens: int
    total_context_tokens: int
    remaining_tokens: int
    counter_name: str
    model_id: str
    is_exact: bool


class PromptBudgetPlanner:
    """Fail-closed context check for a caller-supplied model-attested counter.

    Optional and independent from EvidenceAssembler: callers may first select
    evidence using its own reserve-aware budget, then assess the *entire* final
    provider request before sending it to a model.
    """

    def __init__(
        self,
        *,
        counter: ProviderPromptCounter,
        model_id: str,
        total_context_tokens: int,
        reserved_completion_tokens: int,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            not isinstance(total_context_tokens, int)
            or isinstance(total_context_tokens, bool)
            or total_context_tokens <= 0
        ):
            raise ValueError("total_context_tokens must be a positive integer")
        if (
            not isinstance(reserved_completion_tokens, int)
            or isinstance(reserved_completion_tokens, bool)
            or not 0 <= reserved_completion_tokens < total_context_tokens
        ):
            raise ValueError("reserved_completion_tokens must be within the context window")
        if not counter.is_exact:
            raise ValueError("exact provider prompt counter is required for hard context limits")
        if counter.model_id != model_id:
            raise ValueError("prompt counter model_id does not match target model")
        if not counter.name:
            raise ValueError("provider prompt counter must have a stable name")
        self._counter = counter
        self._model_id = model_id
        self._total = total_context_tokens
        self._reserve = reserved_completion_tokens

    def assess(self, messages: Sequence[Mapping[str, Any]]) -> PromptBudgetResult:
        try:
            count = self._counter.count_prompt_tokens(messages)
        except Exception as exc:
            # Suppress details: a vendor exception may echo raw prompt contents.
            raise PromptBudgetError("provider prompt counter failed") from exc
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise PromptBudgetError("provider returned an invalid prompt token count")
        remaining = self._total - count - self._reserve
        if remaining < 0:
            raise PromptBudgetError(
                "full provider prompt exceeds context after completion token reservation"
            )
        return PromptBudgetResult(
            prompt_tokens=count,
            reserved_completion_tokens=self._reserve,
            total_context_tokens=self._total,
            remaining_tokens=remaining,
            counter_name=self._counter.name,
            model_id=self._model_id,
            is_exact=True,
        )
