"""Sync coordinator orchestrating 7-step manifest-driven sync algorithm.

Enforces:
1. Scoped leases and fencing tokens.
2. Bounded scan observation checkpoints.
3. Revision hint vs content hash planning.
4. Bounded batch embedding and chunk generation staging.
5. CAS/fencing activation at document or collection level.
6. Authoritative coverage tombstone enforcement.
7. Outbox deduplication, async GC, and lease release.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ragit.chunking.base import Chunker
from ragit.config import RagSettings
from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.dimensions import get_embedding_dimension
from ragit.ingestion.connector_core import LoadResult
from ragit.ingestion.sources import RawDocument
from ragit.stores.base import VectorStore
from ragit.sync.manifest import ManifestRecord, ManifestStatus, ManifestStore
from ragit.sync.state import SyncStateStore
from ragit.validation import validate_embedding

if TYPE_CHECKING:
    from ragit.jobs.outbox import EmbeddingJobStore

logger = logging.getLogger(__name__)


class SyncError(Exception):
    """Base error for sync operations."""


class LeaseAcquisitionError(SyncError):
    """Could not acquire lease for sync scope."""


class StaleWorkerError(SyncError):
    """Worker fencing token expired or superseded by another epoch."""


class FencingError(SyncError):
    """CAS update or fencing validation failed."""


class ActivationMode(StrEnum):
    """Activation granularity for staged documents."""

    DOCUMENT = "document"
    COLLECTION = "collection"


@dataclass
class SyncPlanItem:
    """An individual item in the sync plan."""

    document_id: str
    action: str  # "added", "changed", "unchanged", "removed", "metadata_only"
    raw_document: RawDocument | None = None
    content_hash: str = ""
    observed_revision: str | None = None


@dataclass
class SyncPlan:
    """Plan of document updates derived from scan comparison."""

    added: list[SyncPlanItem] = field(default_factory=list)
    changed: list[SyncPlanItem] = field(default_factory=list)
    unchanged: list[SyncPlanItem] = field(default_factory=list)
    removed: list[SyncPlanItem] = field(default_factory=list)
    metadata_only: list[SyncPlanItem] = field(default_factory=list)

    @property
    def total_mutations(self) -> int:
        return len(self.added) + len(self.changed) + len(self.removed) + len(self.metadata_only)


@dataclass
class SyncReport:
    """Summary report of a completed or failed sync execution."""

    scope: str
    success: bool
    added_count: int = 0
    changed_count: int = 0
    unchanged_count: int = 0
    removed_count: int = 0
    metadata_only_count: int = 0
    embedded_count: int = 0
    chunk_count: int = 0
    active_generation: int = 1
    fence_epoch: int = 0
    authoritative: bool = True
    errors: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class SyncCoordinator(Protocol):
    """Protocol for synchronizing documents within a scope."""

    async def run_sync(
        self,
        scope: str,
        source: Any,
        is_authoritative: bool | None = None,
        cursor: Any = None,
        limit: int | None = None,
    ) -> SyncReport:
        """Execute a full sync for the given scope and source."""
        ...

    async def plan(
        self,
        scope: str,
        documents: Sequence[RawDocument],
        is_authoritative: bool = True,
    ) -> SyncPlan:
        """Compute the sync plan for documents without applying mutations."""
        ...

    async def revoke_access(self, scope: str, document_id: str) -> bool:
        """Immediately tombstone and hide a document for privacy/access revocation."""
        ...

    async def garbage_collect(self, scope: str, retired_chunk_ids: list[str] | None = None) -> int:
        """Clean up retired chunk vectors from storage."""
        ...


class DefaultSyncCoordinator:
    """Standard implementation of the 7-step SyncCoordinator protocol."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        manifest_store: ManifestStore,
        state_store: SyncStateStore,
        vector_store: VectorStore,
        embedder: EmbeddingFunction,
        chunker: Chunker,
        settings: RagSettings,
        job_store: EmbeddingJobStore | None = None,
        worker_id: str = "worker-1",
        batch_size: int = 32,
        lease_ttl_seconds: float = 30.0,
        activation_mode: ActivationMode = ActivationMode.DOCUMENT,
    ) -> None:
        self._manifest_store = manifest_store
        self._state_store = state_store
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunker = chunker
        self._settings = settings
        self._job_store = job_store
        self._worker_id = worker_id
        self._batch_size = max(1, batch_size)
        self._lease_ttl_seconds = lease_ttl_seconds
        self._activation_mode = activation_mode

    def _chunker_fingerprint(self) -> str:
        name = self._chunker.__class__.__name__
        size = getattr(self._chunker, "chunk_size", "")
        overlap = getattr(self._chunker, "overlap", "")
        return f"{name}:{size}:{overlap}"

    def _embedding_fingerprint(self) -> str:
        model = self._settings.embedding_model
        dim = get_embedding_dimension(model)
        return f"{model}:{dim}"

    def _extractor_fingerprint(self, doc: RawDocument) -> str:
        return doc.extractor or "none"

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def plan(
        self,
        scope: str,
        documents: Sequence[RawDocument],
        is_authoritative: bool = True,
    ) -> SyncPlan:
        plan = SyncPlan()
        seen_ids: set[str] = set()
        c_fp = self._chunker_fingerprint()
        e_fp = self._embedding_fingerprint()

        for doc in documents:
            doc_id = doc.external_id or doc.source_name
            seen_ids.add(doc_id)
            chash = self._compute_hash(doc.content)
            rev = (
                doc.metadata.get("revision")
                or doc.metadata.get("etag")
                or doc.metadata.get("version")
            )
            existing = await self._manifest_store.get(scope, doc_id)

            if existing is None or existing.status == ManifestStatus.TOMBSTONED:
                plan.added.append(
                    SyncPlanItem(
                        document_id=doc_id,
                        action="added",
                        raw_document=doc,
                        content_hash=chash,
                        observed_revision=rev,
                    )
                )
                continue

            config_changed = (
                existing.chunker_fingerprint != c_fp
                or existing.embedding_fingerprint != e_fp
                or existing.extractor_fingerprint != self._extractor_fingerprint(doc)
            )

            if config_changed:
                plan.changed.append(
                    SyncPlanItem(
                        document_id=doc_id,
                        action="changed",
                        raw_document=doc,
                        content_hash=chash,
                        observed_revision=rev,
                    )
                )
            elif (
                rev is not None
                and existing.observed_revision is not None
                and rev == existing.observed_revision
            ) or existing.content_hash == chash:
                if existing.metadata != doc.metadata:
                    plan.metadata_only.append(
                        SyncPlanItem(
                            document_id=doc_id,
                            action="metadata_only",
                            raw_document=doc,
                            content_hash=chash,
                            observed_revision=rev,
                        )
                    )
                else:
                    plan.unchanged.append(
                        SyncPlanItem(
                            document_id=doc_id,
                            action="unchanged",
                            raw_document=doc,
                            content_hash=chash,
                            observed_revision=rev,
                        )
                    )
            else:
                plan.changed.append(
                    SyncPlanItem(
                        document_id=doc_id,
                        action="changed",
                        raw_document=doc,
                        content_hash=chash,
                        observed_revision=rev,
                    )
                )

        if is_authoritative:
            active_records = await self._manifest_store.list(scope, status=ManifestStatus.ACTIVE)
            for rec in active_records:
                if rec.document_id not in seen_ids:
                    plan.removed.append(
                        SyncPlanItem(
                            document_id=rec.document_id,
                            action="removed",
                            content_hash=rec.content_hash,
                            observed_revision=rec.observed_revision,
                        )
                    )

        return plan

    async def run_sync(  # noqa: PLR0912, PLR0915
        self,
        scope: str,
        source: Any,
        is_authoritative: bool | None = None,
        cursor: Any = None,
        limit: int | None = None,
    ) -> SyncReport:
        # Step 1: Acquire lease and fencing token
        token = await self._state_store.acquire_lease(
            scope, self._worker_id, ttl_seconds=self._lease_ttl_seconds
        )
        if token is None:
            raise LeaseAcquisitionError(
                f"Worker {self._worker_id} could not acquire lease for scope '{scope}'"
            )

        report = SyncReport(scope=scope, success=False, fence_epoch=token.epoch)
        retired_chunk_ids: list[str] = []

        try:
            # Step 2: Scan pages and checkpoint
            documents, authoritative_scan = await self._scan_source(
                source, cursor=cursor, limit=limit
            )
            if is_authoritative is not None:
                authoritative_scan = is_authoritative
            report.authoritative = authoritative_scan

            await self._state_store.save_checkpoint(
                scope,
                {"phase": "scan_complete", "count": len(documents), "cursor": cursor},
                token,
            )

            # Step 3: Plan updates
            plan = await self.plan(scope, documents, is_authoritative=authoritative_scan)
            report.added_count = len(plan.added)
            report.changed_count = len(plan.changed)
            report.unchanged_count = len(plan.unchanged)
            report.removed_count = len(plan.removed)
            report.metadata_only_count = len(plan.metadata_only)

            # Fast path: unchanged scan performs zero embedding calls
            if not plan.added and not plan.changed and not plan.removed and not plan.metadata_only:
                report.success = True
                report.active_generation = await self._state_store.get_active_generation(scope)
                await self._state_store.save_checkpoint(
                    scope, {"phase": "completed", "summary": "identical_scan"}, token
                )
                return report

            # Apply metadata-only updates
            for item in plan.metadata_only:
                doc = item.raw_document
                if doc is not None:
                    rec = await self._manifest_store.get(scope, item.document_id)
                    if rec is not None:
                        rec.metadata = dict(doc.metadata)
                        rec.provenance = dict(doc.provenance)
                        rec.last_successful_observation = time.time()
                        await self._manifest_store.compare_and_swap(rec, rec.fence_token)

            # Step 4: Load & validate changed artifacts, chunk, embed in bounded batches
            items_to_process = plan.added + plan.changed
            staged_docs_by_id: dict[str, list[dict[str, Any]]] = {}
            active_gen = await self._state_store.get_active_generation(scope)

            for item in items_to_process:
                doc = item.raw_document
                if doc is None:
                    continue
                try:
                    existing_rec = await self._manifest_store.get(scope, item.document_id)
                    if self._activation_mode == ActivationMode.COLLECTION:
                        staged_gen = active_gen + 1
                    else:
                        staged_gen = (existing_rec.active_generation if existing_rec else 0) + 1

                    chunks = self._chunker.chunk(doc.content)
                    if not chunks:
                        continue

                    # Bounded batch embedding
                    chunk_texts = [c.text for c in chunks]
                    embeddings: list[list[float]] = []
                    for i in range(0, len(chunk_texts), self._batch_size):
                        batch = chunk_texts[i : i + self._batch_size]
                        batch_embeddings = await self._embedder(batch)
                        embeddings.extend(batch_embeddings)

                    report.embedded_count += len(chunks)
                    expected_dim = get_embedding_dimension(self._settings.embedding_model)

                    staged_chunks: list[dict[str, Any]] = []
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                        validate_embedding(emb, expected_dim=expected_dim)
                        cid = f"{scope}:{item.document_id}:g{staged_gen}:{idx}"
                        meta = {
                            **doc.metadata,
                            **chunk.metadata,
                            "scope": scope,
                            "doc_id": item.document_id,
                            "generation": staged_gen,
                            "active": False,  # Staging chunks are invisible to active readers
                            "source": doc.source_name,
                        }
                        if doc.external_id:
                            meta["external_id"] = doc.external_id
                        if doc.provenance:
                            meta["provenance"] = dict(doc.provenance)
                        staged_chunks.append(
                            {
                                "id": cid,
                                "text": chunk.text,
                                "metadata": meta,
                                "embedding": emb,
                            }
                        )

                    # Stage into vector store
                    await self._vector_store.upsert(staged_chunks)
                    staged_docs_by_id[item.document_id] = staged_chunks

                    # Stage in manifest record
                    staged_rec = ManifestRecord(
                        scope=scope,
                        document_id=item.document_id,
                        observed_revision=item.observed_revision,
                        content_hash=item.content_hash,
                        extractor_fingerprint=self._extractor_fingerprint(doc),
                        chunker_fingerprint=self._chunker_fingerprint(),
                        embedding_fingerprint=self._embedding_fingerprint(),
                        active_generation=existing_rec.active_generation if existing_rec else 0,
                        staged_generation=staged_gen,
                        status=ManifestStatus.STAGED,
                        last_successful_observation=time.time(),
                        chunk_ids=existing_rec.chunk_ids if existing_rec else [],
                        staged_chunk_ids=[c["id"] for c in staged_chunks],
                        metadata=doc.metadata,
                        provenance=doc.provenance,
                        fence_token=existing_rec.fence_token if existing_rec else 0,
                    )
                    await self._manifest_store.put(staged_rec)

                    # Optional outbox integration for durable worker convergence
                    if self._job_store is not None:
                        await self._job_store.enqueue(
                            {
                                "idempotency_key": (
                                    f"{scope}:{item.document_id}:{staged_gen}:embed"
                                ),
                                "scope": scope,
                                "document_id": item.document_id,
                                "generation": staged_gen,
                                "status": "staged",
                            }
                        )
                except Exception as exc:
                    logger.warning("Failed processing document %s: %s", item.document_id, exc)
                    report.errors.append({"doc_id": item.document_id, "error": str(exc)})
                    if self._activation_mode == ActivationMode.COLLECTION:
                        # Collection mode aborts on any document failure
                        raise

            # Step 5: Verify fencing token and activate
            if not await self._state_store.validate_token(token):
                raise StaleWorkerError(
                    f"Fencing token {token.epoch} for worker {self._worker_id} expired or invalid."
                )

            if self._activation_mode == ActivationMode.COLLECTION:
                new_gen = await self._state_store.advance_generation(scope, active_gen)
                if new_gen < 0:
                    raise FencingError("Collection generation advance CAS failed.")
                report.active_generation = new_gen

                for doc_id, staged_chunks in staged_docs_by_id.items():
                    rec = await self._manifest_store.get(scope, doc_id)
                    if rec is not None:
                        retired_chunk_ids.extend(rec.chunk_ids)
                        rec.active_generation = new_gen
                        rec.chunk_ids = list(rec.staged_chunk_ids)
                        rec.staged_chunk_ids = []
                        rec.status = ManifestStatus.ACTIVE
                        await self._manifest_store.compare_and_swap(rec, rec.fence_token)

                        # Activate vector store chunks
                        activated = [
                            {**c, "metadata": {**c["metadata"], "active": True}}
                            for c in staged_chunks
                        ]
                        await self._vector_store.upsert(activated)
                        report.chunk_count += len(activated)
            else:
                # Document-level activation by default
                for doc_id, staged_chunks in staged_docs_by_id.items():
                    rec = await self._manifest_store.get(scope, doc_id)
                    if rec is not None and rec.status == ManifestStatus.STAGED:
                        old_chunk_ids = list(rec.chunk_ids)
                        rec.active_generation = rec.staged_generation
                        rec.chunk_ids = list(rec.staged_chunk_ids)
                        rec.staged_chunk_ids = []
                        rec.status = ManifestStatus.ACTIVE
                        cas_ok = await self._manifest_store.compare_and_swap(rec, rec.fence_token)
                        if cas_ok:
                            activated = [
                                {**c, "metadata": {**c["metadata"], "active": True}}
                                for c in staged_chunks
                            ]
                            await self._vector_store.upsert(activated)
                            retired_chunk_ids.extend(old_chunk_ids)
                            report.chunk_count += len(activated)

            # Step 6: Apply tombstones only within authoritative coverage
            if authoritative_scan and plan.removed:
                for item in plan.removed:
                    rec = await self._manifest_store.get(scope, item.document_id)
                    if rec is not None and rec.status != ManifestStatus.TOMBSTONED:
                        old_chunk_ids = list(rec.chunk_ids)
                        rec.status = ManifestStatus.TOMBSTONED
                        rec.active_generation = 0
                        rec.chunk_ids = []
                        await self._manifest_store.compare_and_swap(rec, rec.fence_token)
                        retired_chunk_ids.extend(old_chunk_ids)

            # Step 7: Persist checkpoint, release lease, and async GC
            await self._state_store.save_checkpoint(
                scope,
                {
                    "phase": "completed",
                    "added": report.added_count,
                    "changed": report.changed_count,
                    "removed": report.removed_count,
                },
                token,
            )

            if retired_chunk_ids:
                asyncio.create_task(self.garbage_collect(scope, retired_chunk_ids))

            report.success = True
            return report

        finally:
            await self._state_store.release_lease(token)

    async def _scan_source(
        self, source: Any, cursor: Any = None, limit: int | None = None
    ) -> tuple[list[RawDocument], bool]:
        """Scan documents from source, determining if coverage is complete and authoritative."""
        if hasattr(source, "load_page"):
            all_docs: list[RawDocument] = []
            authoritative = True
            cur = cursor
            while True:
                page = await source.load_page(cursor=cur, limit=limit)
                docs = page.documents if hasattr(page, "documents") else list(page)
                all_docs.extend(docs)
                if hasattr(page, "authoritative") and not page.authoritative:
                    authoritative = False
                if hasattr(page, "errors") and page.errors:
                    authoritative = False
                cur = getattr(page, "next_cursor", None)
                if not getattr(page, "has_more", False) or cur is None:
                    break
            return all_docs, authoritative

        if hasattr(source, "load"):
            res = await source.load(cursor=cursor, limit=limit)
            if isinstance(res, LoadResult):
                return list(res), res.authoritative
            docs = res.documents if hasattr(res, "documents") else list(res)
            return docs, True

        return list(source), True

    async def revoke_access(self, scope: str, document_id: str) -> bool:
        """Immediately tombstone and hide a document for privacy/access revocation."""
        rec = await self._manifest_store.get(scope, document_id)
        if rec is None:
            return False
        old_chunk_ids = list(rec.chunk_ids)
        rec.status = ManifestStatus.TOMBSTONED
        rec.active_generation = 0
        rec.chunk_ids = []
        await self._manifest_store.compare_and_swap(rec, rec.fence_token)
        if old_chunk_ids:
            try:
                await self._vector_store.delete(ids=old_chunk_ids)
            except Exception as exc:
                logger.warning("Async physical cleanup deferred for %s: %s", document_id, exc)
        return True

    async def garbage_collect(self, scope: str, retired_chunk_ids: list[str] | None = None) -> int:
        """Delete retired chunk vectors from the vector store."""
        if not retired_chunk_ids:
            return 0
        try:
            return await self._vector_store.delete(ids=retired_chunk_ids)
        except Exception as exc:
            logger.warning(
                "Garbage collection of retired chunks for scope %s failed: %s",
                scope,
                exc,
            )
            return 0
