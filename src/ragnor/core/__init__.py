"""Core foundational primitives, types, configuration, and exception taxonomy for Ragnor."""

from __future__ import annotations

from ragnor.core.config import DeploymentMode, RagSettings, RagnorConfig, SettingsProvider
from ragnor.core.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    EmbeddingProviderError,
    IndexConsistencyError,
    OperationTimeoutError,
    RagnorError,
    RetrievalError,
    ScopeDeniedError,
    UnsupportedCapabilityError,
    classify_exception,
    is_retryable_exception,
    sanitize_message,
)
from ragnor.core.lifecycle import AsyncCloseable
from ragnor.core.registry import Registry
from ragnor.core.types import Chunk, Document, SearchResult
from ragnor.core.validation import validate_embedding, validate_identifier

__all__ = [
    "AsyncCloseable",
    "BackendUnavailableError",
    "Chunk",
    "ConfigurationError",
    "DeploymentMode",
    "Document",
    "EmbeddingProviderError",
    "IndexConsistencyError",
    "OperationTimeoutError",
    "RagSettings",
    "RagnorConfig",
    "RagnorError",
    "Registry",
    "RetrievalError",
    "ScopeDeniedError",
    "SearchResult",
    "SettingsProvider",
    "UnsupportedCapabilityError",
    "classify_exception",
    "is_retryable_exception",
    "sanitize_message",
    "validate_embedding",
    "validate_identifier",
]
