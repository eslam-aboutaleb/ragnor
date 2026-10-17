"""Manifest records and storage protocols for document-level sync tracking.

Manifest records track document identity, observed revisions, content hashes,
extractor/chunker/embedding fingerprints, active and staged generations,
lifecycle status, chunk IDs, and observation timestamps.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ManifestStatus(StrEnum):
    """Lifecycle status of a document in the manifest."""

    ACTIVE = "active"
    STAGED = "staged"
    TOMBSTONED = "tombstoned"
    FAILED = "failed"


@dataclass
class ManifestRecord:
    """Tracks document identity, fingerprints, generation, and status.

    Attributes:
        scope: Scoped namespace, collection, or source identifier.
        document_id: Document identity within the scope.
        observed_revision: Opaque revision hint (e.g. ETag, mtime, or version string).
            Treated strictly as a revision hint, not a universal content hash.
        content_hash: Cryptographic hash of normalized document content.
        extractor_fingerprint: Identity/version/options of the extractor.
        chunker_fingerprint: Configuration/version of the chunker.
        embedding_fingerprint: Model name/dimension of the embedding function.
        active_generation: Generation currently active and visible to readers.
        staged_generation: Generation currently staged and awaiting activation.
        status: Lifecycle status of the document.
        last_successful_observation: Epoch timestamp of last successful observation.
        chunk_ids: IDs of vector chunks active for this document.
        staged_chunk_ids: IDs of vector chunks staged for this document.
        metadata: Document metadata preserved through normalization.
        provenance: Original source provenance.
        fence_token: Monotonically increasing version used for compare-and-swap.
    """

    scope: str
    document_id: str
    observed_revision: str | None = None
    content_hash: str = ""
    extractor_fingerprint: str = ""
    chunker_fingerprint: str = ""
    embedding_fingerprint: str = ""
    active_generation: int = 0
    staged_generation: int = 0
    status: ManifestStatus | str = ManifestStatus.ACTIVE
    last_successful_observation: float | None = None
    chunk_ids: list[str] = field(default_factory=list)
    staged_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    fence_token: int = 0


@runtime_checkable
class ManifestStore(Protocol):
    """Storage seam for manifest records.

    Durable host adapters define persistence and concurrency semantics.
    """

    async def get(self, scope: str, document_id: str) -> ManifestRecord | None:
        """Fetch the manifest record for a document in a scope, or None."""
        ...

    async def list(
        self, scope: str, status: ManifestStatus | str | None = None
    ) -> list[ManifestRecord]:
        """List all manifest records in a scope, optionally filtered by status."""
        ...

    async def put(self, record: ManifestRecord) -> None:
        """Unconditionally put or overwrite a manifest record."""
        ...

    async def compare_and_swap(self, record: ManifestRecord, expected_fence: int) -> bool:
        """Atomically update a record if the stored fence_token matches expected_fence.

        If successful, increments record.fence_token. If expected_fence is 0,
        the record must not already exist.
        """
        ...

    async def delete(self, scope: str, document_id: str) -> bool:
        """Delete a record from the manifest store entirely."""
        ...


class InMemoryManifestStore:
    """Thread-safe and async-safe in-memory ManifestStore implementation."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ManifestRecord] = {}
        self._lock = asyncio.Lock()

    async def get(self, scope: str, document_id: str) -> ManifestRecord | None:
        async with self._lock:
            rec = self._records.get((scope, document_id))
            if rec is None:
                return None
            return ManifestRecord(**rec.__dict__)

    async def list(
        self, scope: str, status: ManifestStatus | str | None = None
    ) -> list[ManifestRecord]:
        async with self._lock:
            records = [
                ManifestRecord(**rec.__dict__)
                for (s, _), rec in self._records.items()
                if s == scope and (status is None or rec.status == status)
            ]
            return sorted(records, key=lambda r: r.document_id)

    async def put(self, record: ManifestRecord) -> None:
        async with self._lock:
            record.fence_token += 1
            self._records[(record.scope, record.document_id)] = ManifestRecord(**record.__dict__)

    async def compare_and_swap(self, record: ManifestRecord, expected_fence: int) -> bool:
        async with self._lock:
            key = (record.scope, record.document_id)
            existing = self._records.get(key)
            current_fence = existing.fence_token if existing is not None else 0
            if current_fence != expected_fence:
                return False
            record.fence_token = expected_fence + 1
            self._records[key] = ManifestRecord(**record.__dict__)
            return True

    async def delete(self, scope: str, document_id: str) -> bool:
        async with self._lock:
            key = (scope, document_id)
            if key in self._records:
                del self._records[key]
                return True
            return False
