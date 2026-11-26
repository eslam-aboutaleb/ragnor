"""Evidence assembly, contextual grounding, and citation tracking."""

from __future__ import annotations

from ragnor.evidence.assembler import EvidenceAssembler, EvidenceCompressor
from ragnor.evidence.attestation import (
    DEFAULT_PROBES,
    ProbeMismatch,
    TokenizerAttestation,
    TokenizerAttestationError,
    attest_prompt_counter,
    attest_tokenizer,
)
from ragnor.evidence.source_lookup import (
    BoundedSpanResult,
    MemorySnapshotStore,
    SnapshotRecord,
    SnapshotStore,
    SourceLookup,
)
from ragnor.evidence.tokenizer import (
    ApproximateTokenizer,
    ExactTokenizerAdapter,
    Tokenizer,
)
from ragnor.evidence.types import (
    ContextBudget,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStatus,
    RevisionUnavailableError,
    ScopeAuthorizationError,
    SearchHit,
    escape_untrusted_delimiters,
    sanitize_uri,
)

__all__ = [
    "DEFAULT_PROBES",
    "ProbeMismatch",
    "TokenizerAttestation",
    "TokenizerAttestationError",
    "attest_prompt_counter",
    "attest_tokenizer",
    "ApproximateTokenizer",
    "BoundedSpanResult",
    "ContextBudget",
    "EvidenceAssembler",
    "EvidenceBundle",
    "EvidenceCompressor",
    "EvidenceItem",
    "EvidenceStatus",
    "ExactTokenizerAdapter",
    "MemorySnapshotStore",
    "RevisionUnavailableError",
    "ScopeAuthorizationError",
    "SearchHit",
    "SnapshotRecord",
    "SnapshotStore",
    "SourceLookup",
    "Tokenizer",
    "escape_untrusted_delimiters",
    "sanitize_uri",
]
