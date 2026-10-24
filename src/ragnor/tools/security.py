"""Security, rate limiting, authentication, and scope mapping for remote transports."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragnor.evidence.types import ScopeAuthorizationError


class RateLimitExceededError(Exception):
    """Raised when request rate or concurrency limit is exceeded."""


class AuthenticationError(Exception):
    """Raised when authentication fails or credentials are missing."""


class RateLimiter:
    """Sliding-window rate limiter with concurrency tracking.

    Thread-safe and async-friendly in-memory limiter.
    """

    def __init__(
        self,
        requests_per_minute: int = 120,
        max_concurrent: int = 20,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._active_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def acquire(self, client_id: str = "global") -> None:
        """Acquire a request slot or raise RateLimitExceededError."""
        async with self._lock:
            now = time.monotonic()
            window_start = now - 60.0

            # Prune timestamps older than 60s
            queue = self._history[client_id]
            while queue and queue[0] < window_start:
                queue.popleft()

            # Check rate limit
            if len(queue) >= self.requests_per_minute:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {len(queue)} requests in last 60s "
                    f"(maximum {self.requests_per_minute})"
                )

            # Check concurrency limit
            active = self._active_counts[client_id]
            if active >= self.max_concurrent:
                raise RateLimitExceededError(
                    f"Concurrency limit exceeded: {active} active requests "
                    f"(maximum {self.max_concurrent})"
                )

            # Record request
            queue.append(now)
            self._active_counts[client_id] += 1

    async def release(self, client_id: str = "global") -> None:
        """Release an active concurrent request slot."""
        async with self._lock:
            if self._active_counts[client_id] > 0:
                self._active_counts[client_id] -= 1


class ScopeAuthorizer:
    """Authorizes client requests against collections and isolation scopes."""

    def __init__(
        self,
        scope_mapping: (
            Mapping[str, Sequence[str] | set[str]]
            | Callable[[str | None, str, str | None], bool]
            | None
        ) = None,
        default_allow: bool = True,
    ) -> None:
        """Initialize ScopeAuthorizer.

        Args:
            scope_mapping: Mapping of client/token IDs to allowed scopes or a custom callback.
            default_allow: Whether to allow access when no explicit scope map is configured.
        """
        self._mapping = scope_mapping
        self._default_allow = default_allow

    @property
    def has_explicit_restrictions(self) -> bool:
        """Whether authorization is mapped or defaults to denying access."""
        return self._mapping is not None or not self._default_allow

    def is_authorized(
        self,
        client_id: str | None,
        collection: str,
        scope: str | None,
    ) -> bool:
        """Check whether client is authorized for the given collection and scope."""
        if callable(self._mapping):
            return self._mapping(client_id, collection, scope)

        if self._mapping is None:
            return self._default_allow

        if client_id is None:
            return False

        allowed = self._mapping.get(client_id)
        if allowed is None:
            return False

        if "*" in allowed:
            return True

        target_key = f"{collection}:{scope}" if scope else collection
        return target_key in allowed or scope in allowed or collection in allowed

    def enforce(self, client_id: str | None, collection: str, scope: str | None) -> None:
        """Raise ScopeAuthorizationError if access is not permitted."""
        if not self.is_authorized(client_id, collection, scope):
            raise ScopeAuthorizationError(
                f"Client {client_id!r} is not authorized for collection {collection!r} "
                f"under scope {scope!r}"
            )


@dataclass
class RemoteTransportConfig:
    """Configuration for remote MCP transport deployment.

    Attributes:
        auth_token_validator: Callable validating host-provided auth token / API key.
        rate_limiter: Configured request rate limiter.
        scope_authorizer: Scope mapping and authorization engine.
        host: Host IP or hostname to bind to.
        port: Port number for HTTP/SSE/Streamable HTTP transport.
        endpoint_path: Root endpoint path for the MCP server.
        cors_allowed_origins: Permitted CORS origins for browser/client connections.
        request_timeout_seconds: Bounded timeout per tool invocation.
        metadata: Deployment environment metadata.
    """

    auth_token_validator: Callable[[str], bool] | None = None
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    scope_authorizer: ScopeAuthorizer = field(default_factory=ScopeAuthorizer)
    host: str = "127.0.0.1"
    port: int = 8000
    endpoint_path: str = "/mcp"
    cors_allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    request_timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_token(self, token: str | None) -> bool:
        """Validate auth token against configured validator."""
        if self.auth_token_validator is None:
            return True
        if not token:
            return False
        return self.auth_token_validator(token)
