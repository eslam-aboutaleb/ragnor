"""Compatibility import for :mod:`ragnor.core.exceptions`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import exceptions as _implementation

if TYPE_CHECKING:
    from ragnor.core.exceptions import (
        Any as Any,
        BackendUnavailableError as BackendUnavailableError,
        ConfigurationError as ConfigurationError,
        EmbeddingProviderError as EmbeddingProviderError,
        IndexConsistencyError as IndexConsistencyError,
        Mapping as Mapping,
        OperationTimeoutError as OperationTimeoutError,
        RagnorError as RagnorError,
        RetrievalError as RetrievalError,
        ScopeDeniedError as ScopeDeniedError,
        UnsupportedCapabilityError as UnsupportedCapabilityError,
        classify_exception as classify_exception,
        is_retryable_exception as is_retryable_exception,
        re as re,
        sanitize_message as sanitize_message,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
