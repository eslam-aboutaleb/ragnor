"""Framework-neutral tool bridges and adapters for agent and API integration."""

from __future__ import annotations

from ragit.tools.bridge import CollectionConfig, KnowledgeBridge
from ragit.tools.security import (
    AuthenticationError,
    RateLimitExceededError,
    RateLimiter,
    RemoteTransportConfig,
    ScopeAuthorizer,
)
from ragit.tools.validation import (
    SecurityValidationError,
    validate_safe_filters,
    validate_safe_identifier,
    validate_safe_string,
    validate_safe_uri,
)

__all__ = [
    "AuthenticationError",
    "CollectionConfig",
    "KnowledgeBridge",
    "RateLimitExceededError",
    "RateLimiter",
    "RemoteTransportConfig",
    "ScopeAuthorizer",
    "SecurityValidationError",
    "validate_safe_filters",
    "validate_safe_identifier",
    "validate_safe_string",
    "validate_safe_uri",
]
