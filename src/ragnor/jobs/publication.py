"""Provider-agnostic read-through gate for immutable staged vector generations.

Readers MUST use the authoritative active-generation manifest; querying all
staged vector keys defeats the atomic visibility/fencing guarantee.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from ragnor.core.exceptions import ScopeDeniedError

T_co = TypeVar("T_co", covariant=True)


class GenerationManifest(Protocol):
    """Durable manifest lookup, scoped to one authenticated tenant."""

    scope: str

    async def get_active_generation(self, document_id: str) -> str | None:
        """Read a committed active generation for this document."""
        raise NotImplementedError


class GenerationVectorReader(Protocol[T_co]):
    """Provider adapter that only reads one immutable generation key."""

    async def read_generation(self, generation_key: str) -> T_co:
        """Return content from exactly the specified immutable generation."""
        raise NotImplementedError


class ActiveGenerationReader:
    """Fail closed when scope or active generation is missing.

    Use this *instead of* direct unfiltered access to a staging provider.
    A custom provider must implement exact-generation filtering.
    """

    def __init__(
        self,
        manifest: GenerationManifest,
        vectors: GenerationVectorReader[tuple[str, ...]],
    ) -> None:
        self._manifest = manifest
        self._vectors = vectors

    async def read(self, scope: str, document_id: str) -> tuple[str, ...]:
        if scope != self._manifest.scope:
            raise ScopeDeniedError("scope does not match the manifest's authenticated tenant")
        key = await self._manifest.get_active_generation(document_id)
        if key is None:
            return ()
        return await self._vectors.read_generation(key)
