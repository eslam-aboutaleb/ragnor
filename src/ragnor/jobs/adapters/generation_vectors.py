"""Immutable staged-generation adapter over an injected vector-store port.

This optional component has no vendor import: QdrantStore, pgvector adapters
and host-defined stores can implement the same `get`/`upsert` seam.
Read visibility is controlled separately by an active-generation manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ragnor.core.exceptions import ScopeDeniedError
from ragnor.jobs.persistence import GenerationReservations


class GenerationDocumentStore(Protocol):
    """Narrow read/write port supported by multiple concrete vector stores."""

    async def get(
        self,
        ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return documents with id, document, and selected payload metadata."""
        raise NotImplementedError

    async def upsert(
        self, documents: list[dict[str, Any]], session: Any = None, **kwargs: Any
    ) -> None:
        """Persist immutable generation-qualified document ids."""
        raise NotImplementedError


class ImmutableGenerationVectorAdapter:
    """Stage retry-safe, tenant-qualified vectors and read exactly one generation.

    Caller MUST first stage all chunks, then atomically publish the generation
    through an independent durable manifest. Do not expose `store` as the
    application's unfiltered reader. A durable, atomic reservation binds the
    entire generation contents before
    any provider read/write. Concurrent retries therefore write identical IDs
    and contents, even after a partial provider write or process crash.
    """

    def __init__(
        self,
        *,
        store: GenerationDocumentStore,
        scope: str,
        reservations: GenerationReservations,
    ) -> None:
        if not scope or not scope.strip():
            raise ValueError("scope must be nonempty")
        if getattr(reservations, "scope", None) != scope:
            raise ScopeDeniedError("generation reservations must use the vector adapter scope")
        if not callable(getattr(reservations, "reserve_generation", None)):
            raise ValueError("atomic generation reservations are required")
        self._store = store
        self._reservations = reservations
        self.scope = scope

    @staticmethod
    def _stable_id(scope: str, generation_key: str, source_id: str) -> str:
        raw = json.dumps([scope, generation_key, source_id], separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def stage(self, generation_key: str, documents: Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be nonempty")
        staged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for document in documents:
            source_id = document.get("id")
            contents = document.get("text")
            embedding = document.get("embedding")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("staged document id must be nonempty")
            if source_id in seen:
                raise ValueError("duplicate document id within one immutable generation")
            seen.add(source_id)
            if not isinstance(contents, str) or not contents:
                raise ValueError("staged document text must be nonempty")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("staged document embedding must be a nonempty list")
            meta = document.get("metadata", {})
            if not isinstance(meta, Mapping):
                raise ValueError("staged metadata must be a mapping")
            reserved = {"scope", "generation_key", "content_digest", "original_id"}
            if reserved.intersection(meta):
                raise ValueError("staged metadata cannot override generation identity")
            try:
                canonical = json.dumps(
                    {
                        "source_id": source_id,
                        "text": contents,
                        "embedding": embedding,
                        "metadata": dict(meta),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (ValueError, TypeError) as exc:
                raise ValueError("generation documents must be canonical JSON values") from exc
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            staged.append(
                {
                    "id": self._stable_id(self.scope, generation_key, source_id),
                    "text": contents,
                    "embedding": embedding,
                    "metadata": {
                        **meta,
                        "scope": self.scope,
                        "generation_key": generation_key,
                        "content_digest": digest,
                        "original_id": source_id,
                    },
                }
            )

        if not staged:
            raise ValueError("refusing to activate an empty generation")
        identity = sorted((item["id"], item["metadata"]["content_digest"]) for item in staged)
        batch_digest = hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # This durable CAS must commit independently of the external write. A
        # partial write can only be retried with the exact same complete batch.
        await self._reservations.reserve_generation(generation_key, batch_digest)
        prior = await self._store.get(
            ids=[x["id"] for x in staged],
            metadata_fields=["scope", "generation_key", "content_digest"],
        )
        by_id = {str(item["id"]): item for item in prior}
        to_upsert = []
        for item in staged:
            existing = by_id.get(item["id"])
            if existing is not None:
                actual = existing.get("metadata") or {}
                expected = item["metadata"]
                if any(
                    actual.get(key) != expected[key]
                    for key in ("scope", "generation_key", "content_digest")
                ):
                    raise ValueError("immutable generation content mismatch on replay")
            else:
                to_upsert.append(item)
        if to_upsert:
            await self._store.upsert(to_upsert)

    async def read_generation(self, generation_key: str) -> tuple[str, ...]:
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be nonempty")
        rows = await self._store.get(
            filters={"scope": self.scope, "generation_key": generation_key},
            metadata_fields=["scope", "generation_key", "original_id"],
        )
        for row in rows:
            meta = row.get("metadata") or {}
            if meta.get("scope") != self.scope or meta.get("generation_key") != generation_key:
                raise ScopeDeniedError("storage adapter returned an unscoped generation")
        ordered = sorted(rows, key=lambda x: str(x.get("metadata", {}).get("original_id", "")))
        return tuple(str(row["document"]) for row in ordered)
