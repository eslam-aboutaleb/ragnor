"""Tokenizer / prompt-counter attestation against provider-reported counts.

The provider side here is a deterministic local counter with real per-message
framing. It stands in for a deployed provider: these tests qualify the
comparison and fail-closed logic, not any live model.
"""

from __future__ import annotations

import re

import pytest

from ragnor.core.exceptions import ConfigurationError
from ragnor.evidence import (
    DEFAULT_PROBES,
    ApproximateTokenizer,
    ExactTokenizerAdapter,
    TokenizerAttestationError,
    attest_prompt_counter,
    attest_tokenizer,
)

MODEL = "demo-model-rev1"


def _wordpiece(text: str, piece: int) -> int:
    # Words split into pieces of `piece` characters; punctuation counts alone.
    total = 0
    for tok in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        total += -(-len(tok) // piece)
    return total


class FramedProvider:
    """Counts like a chat API: 3 tokens per message frame + 1 reply primer."""

    is_exact = True
    name = "framed-provider"

    def __init__(self, piece: int = 4, model_id: str = MODEL) -> None:
        self.piece = piece
        self.model_id = model_id
        self.calls = 0

    def count_prompt_tokens(self, messages) -> int:
        self.calls += 1
        return sum(3 + _wordpiece(str(m.get("content", "")), self.piece) for m in messages) + 1


def _tokenizer(piece: int) -> ExactTokenizerAdapter:
    return ExactTokenizerAdapter(lambda t: _wordpiece(t, piece), name=f"wordpiece-{piece}")


def test_matching_tokenizer_passes_and_calibrates_frame_overhead() -> None:
    result = attest_tokenizer(_tokenizer(4), FramedProvider(4), model_id=MODEL)
    assert result.passed
    assert result.frame_overhead_tokens == 4
    assert result.probe_count == len(DEFAULT_PROBES)
    assert result.require() is result


def test_declared_exact_tokenizer_that_miscounts_is_caught() -> None:
    result = attest_tokenizer(_tokenizer(3), FramedProvider(4), model_id=MODEL)
    assert not result.passed
    assert result.mismatches
    with pytest.raises(TokenizerAttestationError, match="disagrees") as info:
        result.require()
    assert info.value.code == "ERR_TOKENIZER_ATTESTATION_FAILED"
    assert isinstance(info.value, ConfigurationError)


def test_mismatch_report_keeps_no_probe_text() -> None:
    result = attest_tokenizer(_tokenizer(3), FramedProvider(4), model_id=MODEL)
    text = repr(result) + str(result.fingerprint())
    assert all(probe[:12] not in text for probe in DEFAULT_PROBES)


def test_approximate_tokenizer_cannot_be_attested() -> None:
    with pytest.raises(TokenizerAttestationError, match="approximate"):
        attest_tokenizer(ApproximateTokenizer(), FramedProvider(), model_id=MODEL)


def test_wrong_provider_model_is_refused() -> None:
    with pytest.raises(TokenizerAttestationError, match="model_id"):
        attest_tokenizer(_tokenizer(4), FramedProvider(model_id="other"), model_id=MODEL)


def test_inexact_provider_counter_is_refused() -> None:
    provider = FramedProvider()
    provider.is_exact = False
    with pytest.raises(ValueError, match="exact"):
        attest_tokenizer(_tokenizer(4), provider, model_id=MODEL)


@pytest.mark.parametrize(
    "probes",
    [("a", "b"), ("a", "a", "b"), ("a", "", "b")],
)
def test_invalid_probe_sets_are_rejected(probes) -> None:
    with pytest.raises(ValueError, match="probes"):
        attest_tokenizer(_tokenizer(4), FramedProvider(), model_id=MODEL, probes=probes)


def test_provider_failure_is_sanitized() -> None:
    class Leaky(FramedProvider):
        def count_prompt_tokens(self, messages) -> int:
            raise RuntimeError("secret-prompt-contents sk-live-123")

    with pytest.raises(TokenizerAttestationError) as info:
        attest_tokenizer(_tokenizer(4), Leaky(), model_id=MODEL)
    assert "sk-live" not in str(info.value)
    assert "secret-prompt" not in str(info.value)


@pytest.mark.parametrize("bad", [-1, True, "7", 1.5])
def test_invalid_provider_count_is_refused(bad) -> None:
    class Bad(FramedProvider):
        def count_prompt_tokens(self, messages):
            return bad

    with pytest.raises(TokenizerAttestationError, match="invalid"):
        attest_tokenizer(_tokenizer(4), Bad(), model_id=MODEL)


def test_fingerprint_changes_with_tokenizer_model_and_probe_set() -> None:
    base = attest_tokenizer(_tokenizer(4), FramedProvider(4), model_id=MODEL)
    assert (
        base.fingerprint()
        == attest_tokenizer(_tokenizer(4), FramedProvider(4), model_id=MODEL).fingerprint()
    )
    other_probes = attest_tokenizer(
        _tokenizer(4), FramedProvider(4), model_id=MODEL, probes=DEFAULT_PROBES[:4]
    )
    failed = attest_tokenizer(_tokenizer(3), FramedProvider(4), model_id=MODEL)
    assert len({base.fingerprint(), other_probes.fingerprint(), failed.fingerprint()}) == 3


MESSAGE_SETS = [
    [{"role": "user", "content": "hello there"}],
    [
        {"role": "system", "content": "Answer from evidence only."},
        {"role": "user", "content": "What is the refund window?"},
    ],
    [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Q"},
        {"role": "tool", "content": "Citation [1]: thirty days، 三十天"},
    ],
]


class HostCounter(FramedProvider):
    name = "host-counter"


def test_full_prompt_counter_matches_provider_framing() -> None:
    result = attest_prompt_counter(
        HostCounter(4), FramedProvider(4), model_id=MODEL, message_sets=MESSAGE_SETS
    )
    assert result.passed
    assert result.kind == "prompt_counter"


def test_prompt_counter_that_ignores_framing_is_caught() -> None:
    class NoFraming(HostCounter):
        def count_prompt_tokens(self, messages) -> int:
            return sum(_wordpiece(str(m.get("content", "")), 4) for m in messages)

    result = attest_prompt_counter(
        NoFraming(4), FramedProvider(4), model_id=MODEL, message_sets=MESSAGE_SETS
    )
    assert len(result.mismatches) == len(MESSAGE_SETS)
    with pytest.raises(TokenizerAttestationError):
        result.require()


def test_prompt_counter_requires_enough_message_sets_and_matching_model() -> None:
    with pytest.raises(ValueError, match="message sets"):
        attest_prompt_counter(
            HostCounter(4), FramedProvider(4), model_id=MODEL, message_sets=MESSAGE_SETS[:1]
        )
    with pytest.raises(TokenizerAttestationError, match="model_id"):
        attest_prompt_counter(
            HostCounter(4, model_id="x"),
            FramedProvider(4),
            model_id=MODEL,
            message_sets=MESSAGE_SETS,
        )
