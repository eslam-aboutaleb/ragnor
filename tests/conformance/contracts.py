"""Reusable adapter contract assertions for SDK extension authors.

Backward-compatible re-exports: the published conformance suite now ships
inside the distribution as :mod:`ragnor.conformance` so third-party adapter
packages can run it against an installed wheel. This module remains so
existing in-repo suites keep their import path.
"""

from __future__ import annotations

from ragnor.conformance import (
    check_chunker,
    check_document_source,
    check_embedder,
    check_event_sink,
    check_tokenizer,
)

__all__ = [
    "check_chunker",
    "check_document_source",
    "check_embedder",
    "check_event_sink",
    "check_tokenizer",
]
