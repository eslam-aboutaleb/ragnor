"""Structural lifecycle port for SDK-owned asynchronous resources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncCloseable(Protocol):
    """A resource with an asynchronous close method.

    Implementations live in host applications or optional infrastructure
    adapters. Ragnor only closes instances explicitly registered as owned.
    """

    async def aclose(self) -> None:
        """Release the resource; may be called only once by SDK lifecycle."""
        raise NotImplementedError
