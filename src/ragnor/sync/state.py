"""Sync state store and fencing token protocols for distributed coordination.

Guarantees mutual exclusion and monotonic fencing epochs per scope, prevents
stale workers from activating obsolete revisions or writing checkpoints,
and tracks collection generations.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from dataclasses import dataclass
import time
from typing import Any, Protocol, runtime_checkable


@dataclass
class FenceToken:
    """Monotonic fencing token and lease proof for a specific sync scope.

    Attributes:
        scope: Scoped namespace, collection, or source identifier.
        owner: Unique worker or coordinator identifier.
        epoch: Monotonically increasing epoch number.
        lease_expires_at: Epoch timestamp (seconds) when lease expires.
    """

    scope: str
    owner: str
    epoch: int
    lease_expires_at: float

    def is_valid(self, now: float | None = None) -> bool:
        """Check if lease has not yet expired."""
        if now is None:
            now = time.time()
        return self.lease_expires_at > now


@runtime_checkable
class SyncStateStore(Protocol):
    """Storage seam for coordination leases, fencing tokens, and checkpoints."""

    async def acquire_lease(
        self, scope: str, owner: str, ttl_seconds: float = 30.0
    ) -> FenceToken | None:
        """Acquire a scoped lease with an incremented epoch.

        Returns FenceToken if acquired, None if lease is currently held by an active worker.
        """
        ...

    async def renew_lease(self, token: FenceToken, ttl_seconds: float = 30.0) -> FenceToken | None:
        """Renew lease expiry if the fencing token is still valid and active."""
        ...

    async def release_lease(self, token: FenceToken) -> bool:
        """Release the lease if held by the given fencing token."""
        ...

    async def validate_token(self, token: FenceToken) -> bool:
        """Verify that the token's epoch and owner match the active valid lease."""
        ...

    async def get_active_generation(self, scope: str) -> int:
        """Get the current collection-level active generation for the scope."""
        ...

    async def advance_generation(self, scope: str, expected_generation: int) -> int:
        """Atomically advance the collection active generation if matching expected."""
        ...

    async def get_checkpoint(self, scope: str) -> dict[str, Any] | None:
        """Retrieve the last committed sync checkpoint for the scope."""
        ...

    async def save_checkpoint(
        self, scope: str, checkpoint: dict[str, Any], token: FenceToken
    ) -> bool:
        """Persist a checkpoint guarded by the active fencing token."""
        ...

    def fenced(self, token: FenceToken) -> AbstractAsyncContextManager[None]:
        """Exclude lease takeover while committing, and reject invalid tokens.

        Durable adapters must provide an equivalent transaction/advisory lock;
        a separate token check followed by an unguarded write is insufficient.
        """
        ...


class InMemorySyncStateStore:
    """Thread-safe and async-safe in-memory SyncStateStore."""

    def __init__(self) -> None:
        self._leases: dict[str, FenceToken] = {}
        self._epochs: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._scope_locks: dict[str, asyncio.Lock] = {}
        self._guarded: set[tuple[str, int]] = set()

    @asynccontextmanager
    async def fenced(self, token: FenceToken) -> AsyncIterator[None]:
        async with self._scope_locks.setdefault(token.scope, asyncio.Lock()):
            if not await self.validate_token(token):
                raise ValueError("Expired or superseded fencing token")
            self._guarded.add((token.scope, token.epoch))
            try:
                yield
            finally:
                self._guarded.discard((token.scope, token.epoch))

    async def acquire_lease(
        self, scope: str, owner: str, ttl_seconds: float = 30.0
    ) -> FenceToken | None:
        async with self._scope_locks.setdefault(scope, asyncio.Lock()), self._lock:
            now = time.time()
            current_lease = self._leases.get(scope)
            if current_lease is not None and current_lease.is_valid(now):
                return None

            epoch = self._epochs.get(scope, 0) + 1
            self._epochs[scope] = epoch
            token = FenceToken(
                scope=scope,
                owner=owner,
                epoch=epoch,
                lease_expires_at=now + ttl_seconds,
            )
            self._leases[scope] = token
            if scope not in self._generations:
                self._generations[scope] = 1
            return token

    async def renew_lease(self, token: FenceToken, ttl_seconds: float = 30.0) -> FenceToken | None:
        async with self._lock:
            now = time.time()
            current = self._leases.get(token.scope)
            if (
                current is None
                or current.owner != token.owner
                or current.epoch != token.epoch
                or not current.is_valid(now)
            ):
                return None

            renewed = FenceToken(
                scope=token.scope,
                owner=token.owner,
                epoch=token.epoch,
                lease_expires_at=now + ttl_seconds,
            )
            self._leases[token.scope] = renewed
            return renewed

    async def release_lease(self, token: FenceToken) -> bool:
        async with self._lock:
            current = self._leases.get(token.scope)
            if (
                current is not None
                and current.owner == token.owner
                and current.epoch == token.epoch
            ):
                del self._leases[token.scope]
                return True
            return False

    async def validate_token(self, token: FenceToken) -> bool:
        async with self._lock:
            now = time.time()
            current = self._leases.get(token.scope)
            return (
                current is not None
                and current.owner == token.owner
                and current.epoch == token.epoch
                and current.is_valid(now)
            )

    async def get_active_generation(self, scope: str) -> int:
        async with self._lock:
            return self._generations.get(scope, 1)

    async def advance_generation(self, scope: str, expected_generation: int) -> int:
        async with self._lock:
            current = self._generations.get(scope, 1)
            if current != expected_generation:
                return -1
            self._generations[scope] = current + 1
            return self._generations[scope]

    async def get_checkpoint(self, scope: str) -> dict[str, Any] | None:
        async with self._lock:
            cp = self._checkpoints.get(scope)
            return dict(cp) if cp is not None else None

    async def save_checkpoint(
        self, scope: str, checkpoint: dict[str, Any], token: FenceToken
    ) -> bool:
        async with self._lock:
            now = time.time()
            current = self._leases.get(scope)
            if (
                current is None
                or current.owner != token.owner
                or current.epoch != token.epoch
                or (not current.is_valid(now) and (scope, token.epoch) not in self._guarded)
            ):
                return False
            self._checkpoints[scope] = dict(checkpoint)
            return True
