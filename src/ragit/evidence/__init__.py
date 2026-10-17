"""Evidence assembly, contextual grounding, and citation tracking."""

from __future__ import annotations

from ragit.evidence.assembler import EvidenceAssembler, EvidenceCompressor
from ragit.evidence.source_lookup import (
    BoundedSpanResult,
    MemorySnapshotStore,
    SnapshotRecord,
    SnapshotStore,
    SourceLookup,
)
from ragit.evidence.tokenizer import (
    ApproximateTokenizer,
    ExactTokenizerAdapter,
    Tokenizer,
)
from ragit.evidence.types import (
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
