"""Strict evidence accounting and honest approximate fallback diagnostics."""

from __future__ import annotations

import logging

import pytest

from ragnor.evidence.assembler import EvidenceAssembler
from ragnor.evidence.tokenizer import ApproximateTokenizer, ExactTokenizerAdapter
from ragnor.evidence.types import ContextBudget


def test_strict_evidence_requires_an_exact_tokenizer() -> None:
    with pytest.raises(ValueError, match="exact"):
        EvidenceAssembler(exact_required=True)
    with pytest.raises(ValueError, match="exact"):
        EvidenceAssembler(ApproximateTokenizer(), exact_required=True)


def test_strict_evidence_uses_exact_full_rendered_context() -> None:
    tokenizer = ExactTokenizerAdapter(len, name="one-codepoint-one-token")
    assembler = EvidenceAssembler(tokenizer=tokenizer, exact_required=True)
    budget = ContextBudget(total_budget=160, reserved_answer=20)
    hits = [{"id": "doc", "document": "refund information " * 25}]
    result = assembler.assemble(hits, budget)
    assert result.total_tokens <= budget.available_evidence_tokens
    assert all(
        tokenizer.count_tokens(x.text) <= budget.available_evidence_tokens for x in result.items
    )


def test_approximate_warning_is_emitted_once(caplog: pytest.LogCaptureFixture) -> None:
    assembler = EvidenceAssembler(tokenizer=ApproximateTokenizer())
    budget = ContextBudget(total_budget=200)
    with caplog.at_level(logging.WARNING):
        assembler.assemble([{"id": "one", "document": "refund"}], budget)
        assembler.assemble([{"id": "two", "document": "coverage"}], budget)
    warnings = [r for r in caplog.records if "approximate tokenizer" in r.message]
    assert len(warnings) == 1
