"""Tokenizer attestation against a provider-reported token count.

A caller-declared ``is_exact`` flag is a claim, not evidence. This module
compares a candidate tokenizer (or full serialized-prompt counter) with the
token counts the deployed provider/model itself reports for the same probe
inputs, and records the outcome as an immutable, secret-free
:class:`TokenizerAttestation`.

Ragnor performs no network calls here. The host supplies the provider side as a
:class:`ProviderPromptCounter` backed by the real provider (for example its
token-count endpoint or the ``usage.prompt_tokens`` of a real request). An
attestation is only as strong as that oracle: passing it against a stand-in
proves the comparison logic, not the deployed model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ragnor.core.exceptions import ConfigurationError
from ragnor.evidence.prompt_budget import ProviderPromptCounter
from ragnor.evidence.tokenizer import Tokenizer

DEFAULT_PROBES: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "Refunds are available within thirty days of purchase.",
    "مرحبا بكم في نظام الاسترجاع. السياسة سارية لمدة ثلاثين يوما.",
    "中文文档和代码混合:返回值为整数。",
    'def f(x):\n    return {"a": [1, 2, 3], "b": x ** 2}\n',
    '{"id": 17, "tags": ["a", "b"], "nested": {"k": null, "v": 1.5e-3}}',
    "emoji 👩🏻‍💻🇪🇬 combining e\u0301 zero\u200bwidth",
    "word " * 200,
)
"""Multilingual, code, JSON, emoji and long-input probes."""

MIN_PROBES = 3
FRAME_PROBE_ROLE = "user"


class TokenizerAttestationError(ConfigurationError):
    """The tokenizer or prompt counter does not match the provider's counts."""

    default_code = "ERR_TOKENIZER_ATTESTATION_FAILED"


@dataclass(frozen=True)
class ProbeMismatch:
    """One probe where the candidate and provider disagree (no probe text kept)."""

    probe_index: int
    provider_tokens: int
    candidate_tokens: int


@dataclass(frozen=True)
class TokenizerAttestation:
    """Outcome of comparing a candidate counter to the provider's counts.

    Contains no probe text and no credentials. ``passed`` means every probe
    matched exactly at attestation time for ``model_id``; it is not a claim
    about inputs outside the probe set.
    """

    kind: str
    subject_name: str
    model_id: str
    probe_set_hash: str
    probe_count: int
    frame_overhead_tokens: int
    mismatches: tuple[ProbeMismatch, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def fingerprint(self) -> str:
        """Stable hash of what was attested, usable in a snapshot/config hash."""
        payload = {
            "kind": self.kind,
            "subject": self.subject_name,
            "model": self.model_id,
            "probes": self.probe_set_hash,
            "frame": self.frame_overhead_tokens,
            "passed": self.passed,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def require(self) -> TokenizerAttestation:
        """Return self when passed, otherwise raise (fail closed)."""
        if not self.passed:
            raise TokenizerAttestationError(
                f"{self.kind} {self.subject_name!r} disagrees with provider counts for "
                f"model {self.model_id!r} on {len(self.mismatches)} of "
                f"{self.probe_count} probes"
            )
        return self


def _probe_hash(probes: Sequence[str]) -> str:
    raw = json.dumps(list(probes), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate(probes: Sequence[str], provider: ProviderPromptCounter, model_id: str) -> None:
    if not model_id or not model_id.strip():
        raise ValueError("model_id must be non-empty")
    if len(probes) < MIN_PROBES:
        raise ValueError(f"at least {MIN_PROBES} probes are required")
    if len(set(probes)) != len(probes):
        raise ValueError("probes must be unique")
    if not all(isinstance(p, str) and p for p in probes):
        raise ValueError("probes must be non-empty strings")
    if not provider.is_exact:
        raise ValueError("provider counter must report exact counts to attest against")
    if provider.model_id != model_id:
        raise TokenizerAttestationError("provider counter model_id does not match target model")


def _provider_count(provider: ProviderPromptCounter, messages: Sequence[Mapping[str, Any]]) -> int:
    try:
        value = provider.count_prompt_tokens(messages)
    except Exception as exc:
        # A vendor exception can echo prompt contents; never propagate it.
        raise TokenizerAttestationError("provider token count failed") from exc
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TokenizerAttestationError("provider returned an invalid token count")
    return value


def attest_tokenizer(
    tokenizer: Tokenizer,
    provider: ProviderPromptCounter,
    *,
    model_id: str,
    probes: Sequence[str] = DEFAULT_PROBES,
) -> TokenizerAttestation:
    """Check a text tokenizer against the provider's own count of the same text.

    The provider counts a one-message prompt per probe. Constant message framing
    is calibrated away using the provider's count for an empty message, so the
    comparison isolates the text tokenization. Providers whose framing is not
    constant should be attested with :func:`attest_prompt_counter` instead.
    """
    _validate(probes, provider, model_id)
    if not tokenizer.is_exact:
        raise TokenizerAttestationError("an approximate tokenizer cannot be attested as exact")
    frame = _provider_count(provider, [{"role": FRAME_PROBE_ROLE, "content": ""}])
    mismatches: list[ProbeMismatch] = []
    for index, probe in enumerate(probes):
        provider_tokens = _provider_count(provider, [{"role": FRAME_PROBE_ROLE, "content": probe}])
        provider_text_tokens = provider_tokens - frame
        candidate = tokenizer.count_tokens(probe)
        if candidate != provider_text_tokens:
            mismatches.append(ProbeMismatch(index, provider_text_tokens, candidate))
    return TokenizerAttestation(
        kind="tokenizer",
        subject_name=tokenizer.name,
        model_id=model_id,
        probe_set_hash=_probe_hash(probes),
        probe_count=len(probes),
        frame_overhead_tokens=frame,
        mismatches=tuple(mismatches),
    )


def attest_prompt_counter(
    candidate: ProviderPromptCounter,
    provider: ProviderPromptCounter,
    *,
    model_id: str,
    message_sets: Sequence[Sequence[Mapping[str, Any]]],
) -> TokenizerAttestation:
    """Check a full serialized-prompt counter against the provider's count.

    ``message_sets`` should cover system, user, tool and multi-turn framing the
    host will actually send.
    """
    if len(message_sets) < MIN_PROBES:
        raise ValueError(f"at least {MIN_PROBES} message sets are required")
    if candidate.model_id != model_id:
        raise TokenizerAttestationError("candidate counter model_id does not match target model")
    if not candidate.is_exact:
        raise TokenizerAttestationError("an approximate prompt counter cannot be attested")
    _validate(tuple(str(i) for i in range(len(message_sets))), provider, model_id)
    mismatches: list[ProbeMismatch] = []
    for index, messages in enumerate(message_sets):
        expected = _provider_count(provider, messages)
        try:
            actual = candidate.count_prompt_tokens(messages)
        except Exception as exc:
            raise TokenizerAttestationError("candidate prompt counter failed") from exc
        if actual != expected:
            mismatches.append(ProbeMismatch(index, expected, actual))
    serialized = json.dumps(
        [list(m) for m in message_sets], ensure_ascii=False, sort_keys=True, default=str
    )
    return TokenizerAttestation(
        kind="prompt_counter",
        subject_name=candidate.name,
        model_id=model_id,
        probe_set_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        probe_count=len(message_sets),
        frame_overhead_tokens=0,
        mismatches=tuple(mismatches),
    )
