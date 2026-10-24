"""Compatibility import for :mod:`ragnor.core.exceptions`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import exceptions as _implementation

if TYPE_CHECKING:
    from ragnor.core.exceptions import (
        Any,
        BackendUnavailableError,
        ConfigurationError,
        EmbeddingProviderError,
        IndexConsistencyError,
        Mapping,
        OperationTimeoutError,
        RagnorError,
        RetrievalError,
        ScopeDeniedError,
        UnsupportedCapabilityError,
        classify_exception,
        is_retryable_exception,
        re,
        sanitize_message,
    )

__all__ = [
    "Any",
    "BackendUnavailableError",
    "ConfigurationError",
    "EmbeddingProviderError",
    "IndexConsistencyError",
    "Mapping",
    "OperationTimeoutError",
    "RagnorError",
    "RetrievalError",
    "ScopeDeniedError",
    "UnsupportedCapabilityError",
    "classify_exception",
    "is_retryable_exception",
    "re",
    "sanitize_message",
]

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
