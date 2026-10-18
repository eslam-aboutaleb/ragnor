"""Retrieval defense-in-depth with mandatory scope and tombstone enforcement.

Enforces zero-trust deletion privacy: even if physical cleanup in a vector
database is lagging, asynchronous, or retrying, tombstoned documents and
inactive chunk generations are filtered out before results are returned.
"""

from __future__ import annotations

import logging
from typing import Any

from ragit.stores.base import VectorStore
from ragit.sync.manifest import ManifestStatus, ManifestStore
from ragit.sync.state import SyncStateStore

logger = logging.getLogger(__name__)


class SyncRetriever:
    """Retriever wrapper enforcing active generation and tombstone checks.

    Readers only see active generations of permitted, non-tombstoned documents.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        manifest_store: ManifestStore,
        state_store: SyncStateStore | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._manifest_store = manifest_store
        self._state_store = state_store

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        scope: str,
        n_results: int = 10,
        threshold: float = 1.0,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search with mandatory active generation and tombstone filtering."""
        # Ensure mandatory metadata fields for generation and tombstone validation
        system_fields = ["doc_id", "scope", "generation", "active", *filters]
        if metadata_fields is None:
            effective_fields = system_fields
        else:
            effective_fields = sorted(set(metadata_fields) | set(system_fields))

        # Query underlying store filtering by scope and active=True
        results = await self._vector_store.hybrid_search(
            query=query,
            embedding=embedding,
            n_results=n_results * 2,  # Oversample to account for potential filtering
            threshold=threshold,
            text_field=text_field,
            metadata_fields=effective_fields,
            session=session,
            strict=strict,
            scope=scope,
            active=True,
            **filters,
        )

        # Read one manifest snapshot so a collection commit cannot be observed halfway.
        records = {record.document_id: record for record in await self._manifest_store.list(scope)}
        filtered: list[dict[str, Any]] = []
        for res in results:
            meta = res.get("metadata") or {}
            doc_id = meta.get("doc_id")
            if not doc_id or meta.get("scope") != scope or meta.get("active") is not True:
                continue
            rec = records.get(doc_id)
            if rec is None:
                continue

            # Strict deletion privacy: tombstoned, failed, or inactive records are dropped
            if rec.status != ManifestStatus.ACTIVE:
                continue

            # Ensure generation matches currently active generation
            chunk_gen = meta.get("generation")
            if rec.active_generation != chunk_gen:
                continue

            # Ensure chunk ID is actually in the active chunk list
            if res.get("id") not in rec.chunk_ids:
                continue

            # If caller didn't ask for system fields, project only what was requested
            if metadata_fields is not None:
                clean_meta = {k: v for k, v in meta.items() if k in metadata_fields}
                res["metadata"] = clean_meta

            filtered.append(res)
            if len(filtered) >= n_results:
                break

        return filtered
