"""Fixed, content-free extraction errors.

Messages never carry provider exception text, document content, prompts or
credentials: provider failures are reported by stable code only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ERROR_CODES = frozenset(
    {
        "unknown_provider",
        "duplicate_provider",
        "invalid_provider",
        "invalid_options",
        "unsupported_format",
        "incompatible_capabilities",
        "remote_not_enabled",
        "invalid_mode",
        "no_provider",
        "invalid_output",
        "provider_failed",
        "timeout",
        "cancelled",
        "partial_only",
        "fallback_exhausted",
        "limit_exceeded",
        "invalid_document",
        "schema_version",
    }
)


class ExtractionError(RuntimeError):
    """Typed, sanitized failure of structured extraction."""

    def __init__(self, code: str, *, attempts: Sequence[Any] = ()) -> None:
        self.code = code if code in ERROR_CODES else "provider_failed"
        self.attempts = tuple(attempts)
        super().__init__(f"Structured extraction failed: {self.code}")


class ExtractionConfigError(ExtractionError, ValueError):
    """Host configuration problem (unknown id, bad options, unsupported format)."""


class ExtractionLimitError(ExtractionError):
    """A size or cell budget was exceeded; nothing was truncated silently."""

    def __init__(self) -> None:
        super().__init__("limit_exceeded")
