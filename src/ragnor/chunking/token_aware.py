"""Opt-in exact token-aware chunking with source-preserving text spans.

No vendor tokenizer is imported. The caller must provide a compatible
Tokenizer (or opt into explicitly approximate sizing for local demos).
"""

from __future__ import annotations

import uuid
from typing import Any

from ragnor.evidence.tokenizer import Tokenizer
from ragnor.core.types import Chunk


class TokenAwareChunker:
    """Split source into bounded character spans using an injected token counter.

    This reference implementation deliberately checks the token budget on
    every candidate prefix; tokenizer counts are not generally monotonic
    because merges may change across boundaries. It never decodes raw token
    byte slices, preserving the exact original Unicode code points.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        exact_required: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < max_tokens")
        if exact_required and not tokenizer.is_exact:
            raise ValueError(
                "exact tokenizer required for a hard token limit; "
                "approximate counting cannot guarantee model limits"
            )
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.exact_required = exact_required

    @property
    def version(self) -> str:
        return "token-aware-spans-v1"

    @property
    def config(self) -> dict[str, Any]:
        return {
            "chunk_size": self.max_tokens,
            "overlap": self.overlap_tokens,
            "unit": "model_tokens",
            "tokenizer": self.tokenizer.name,
            "is_exact": self.tokenizer.is_exact,
            "exact_required": self.exact_required,
        }

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []
        out: list[Chunk] = []
        start = 0
        while start < len(text):
            end = start
            # Evaluate each candidate; no false assumption that token count
            # increases monotonically with text length.
            while end < len(text):
                count = self.tokenizer.count_tokens(text[start : end + 1])
                if count > self.max_tokens:
                    break
                end += 1
            if end == start:
                raise ValueError(
                    "single Unicode code point exceeds requested token budget; "
                    "increase max_tokens or select the matching tokenizer"
                )
            out.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=text[start:end],
                    metadata={
                        "start_offset": start,
                        "end_offset": end,
                        "token_count": self.tokenizer.count_tokens(text[start:end]),
                        "tokenizer": self.tokenizer.name,
                        "is_exact": self.tokenizer.is_exact,
                    },
                )
            )
            if end == len(text):
                break
            next_start = end
            if self.overlap_tokens:
                for candidate in range(start + 1, end):
                    if self.tokenizer.count_tokens(text[candidate:end]) <= self.overlap_tokens:
                        next_start = candidate
                        break
            start = next_start
        return out
