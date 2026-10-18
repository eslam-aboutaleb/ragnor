"""Tokenizer abstractions and estimators for context budget calculation.

Provides the :class:`Tokenizer` protocol, a conservative
:class:`ApproximateTokenizer` estimator, and an :class:`ExactTokenizerAdapter`
for model-specific exact limits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting implementations."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in ``text``."""
        ...

    @property
    def is_exact(self) -> bool:
        """Return True if this tokenizer produces exact model-specific token counts."""
        ...

    @property
    def name(self) -> str:
        """Identifying name of the tokenizer implementation."""
        ...


class ApproximateTokenizer:
    """Conservative heuristic token estimator.

    NOTICE: This is an APPROXIMATE estimator. It uses a conservative
    character-to-token and word-to-token ratio to estimate token counts without
    requiring third-party model tokenizers or network calls.

    WARNING: Never claim this approximate estimate guarantees a model's context
    limit. For strict compliance with model-specific context windows, inject an
    :class:`ExactTokenizerAdapter`.
    """

    @property
    def is_exact(self) -> bool:
        """Always False for heuristic estimators."""
        return False

    @property
    def name(self) -> str:
        return "approximate_heuristic"

    def count_tokens(self, text: str) -> int:
        """Conservatively estimate token count for ``text``.

        Uses the upper bound of ~3.3 characters per token and ~1.35 tokens per word
        to avoid underestimating token usage on punctuation-heavy or technical texts.
        """
        if not text:
            return 0
        char_est = math.ceil(len(text) / 3.3)
        words = text.split()
        word_est = math.ceil(len(words) * 1.35) if words else 0
        return max(char_est, word_est, 1)


class ExactTokenizerAdapter:
    """Adapter wrapping an exact model-specific token counter.

    Enforces exact model-specific context limits.

    Can wrap any callable `Callable[[str], int]`, or objects with an
    `encode(text)` method (such as tiktoken or HuggingFace tokenizers).
    """

    def __init__(
        self,
        tokenizer_or_fn: Any,
        name: str = "exact_model_tokenizer",
    ) -> None:
        """Initialize the exact tokenizer adapter.

        Args:
            tokenizer_or_fn: A callable taking a string and returning int,
                or a tokenizer object with an `encode(text)` method.
            name: Identifier for the model or tokenizer backend.
        """
        self._name = name
        if callable(tokenizer_or_fn):
            self._count_fn: Callable[[str], int] = tokenizer_or_fn
        elif hasattr(tokenizer_or_fn, "encode"):
            self._count_fn = lambda text: len(tokenizer_or_fn.encode(text))
        else:
            raise TypeError(
                "Expected a callable or an object with an 'encode' method, "
                f"got {type(tokenizer_or_fn)}"
            )

    @property
    def is_exact(self) -> bool:
        """Always True for exact adapters."""
        return True

    @property
    def name(self) -> str:
        return self._name

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using the underlying model tokenizer."""
        if not text:
            return 0
        return int(self._count_fn(text))
