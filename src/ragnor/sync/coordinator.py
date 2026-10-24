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
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
import inspect
import math
from contextlib import suppress
from types import SimpleNamespace
from uuid import uuid4
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ragnor.chunking.base import Chunker
from ragnor.core.config import RagSettings
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.dimensions import get_embedding_dimension
from ragnor.ingestion.connector_core import LoadResult, collect_pages, _error_record
from ragnor.embeddings._safe_config import safe_config
from ragnor.core.exceptions import IndexConsistencyError, RagnorError
from ragnor.ingestion.sources import RawDocument
from ragnor.stores.base import VectorStore
from ragnor.sync.manifest import ManifestRecord, ManifestStatus, ManifestStore
from ragnor.sync.state import SyncStateStore, FenceToken
from ragnor.telemetry.events import EventKind, EventSink, NullEventSink, TelemetryEvent
from ragnor.core.validation import validate_embedding

if TYPE_CHECKING:
    from ragnor.jobs.outbox import EmbeddingJobStore

logger = logging.getLogger(__name__)


class SyncError(RagnorError):
    """Base error for sync operations."""

    default_code = "ERR_SYNC"


class LeaseAcquisitionError(SyncError):
    """Could not acquire lease for sync scope."""

    default_code = "ERR_LEASE_ACQUISITION"


class StaleWorkerError(IndexConsistencyError, SyncError):
    """Worker fencing token expired or superseded by another epoch."""

    default_code = "ERR_STALE_WORKER"


class FencingError(IndexConsistencyError, SyncError):
    """CAS update or fencing validation failed."""

    default_code = "ERR_FENCING"


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
    artifacts: list[RawDocument] = field(default_factory=list)
    expected_fence: int = 0


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
    """Prepare immutable vectors, then publish their manifest under a fenced commit.

    Active records are never overwritten by staging. Readers gate immutable chunk
    IDs through the manifest, so prepublished vectors remain invisible until CAS.
    Collection mode requires an atomic manifest batch commit.
    """

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
        event_sink: EventSink | None = None,
    ) -> None:
        if not math.isfinite(lease_ttl_seconds) or lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive and finite")
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
        self._activation_mode = ActivationMode(activation_mode)
        self._event_sink = event_sink or NullEventSink()

    @staticmethod
    def _fingerprint(value: Any) -> str:
        canonical = json.dumps(safe_config(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _chunker_fingerprint(self) -> str:
        config = getattr(self._chunker, "config", None)
        if config is None:
            config = {key: getattr(self._chunker, key, None) for key in ("chunk_size", "overlap")}
        return self._fingerprint(
            {
                "type": type(self._chunker).__qualname__,
                "version": getattr(self._chunker, "version", None),
                "config": config,
            }
        )

    def _embedding_fingerprint(self) -> str:
        profile = getattr(self._embedder, "profile", None)
        if profile is not None:
            return profile.canonical_hash()
        model = self._settings.embedding_model
        return f"{model}:{get_embedding_dimension(model)}"

    def _extractor_fingerprint(self, doc: RawDocument) -> str:
        return self._fingerprint([doc.extractor, doc.metadata.get("extraction")])

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _group_documents(
        documents: Sequence[RawDocument],
    ) -> list[tuple[RawDocument, list[RawDocument]]]:
        groups: dict[str, list[RawDocument]] = {}
        for doc in documents:
            groups.setdefault(doc.external_id or doc.source_name, []).append(doc)
        grouped = []
        for artifacts in groups.values():
            if len(artifacts) == 1:
                grouped.append((artifacts[0], artifacts))
                continue
            locations = [json.dumps(safe_config(doc.location), sort_keys=True) for doc in artifacts]
            if any(doc.location is None for doc in artifacts) or len(set(locations)) != len(
                locations
            ):
                raise ValueError("Duplicate document artifacts require distinct locations")
            ordered_artifacts = sorted(
                artifacts,
                key=lambda doc: (
                    (0, doc.location["page"])
                    if isinstance(doc.location, dict) and isinstance(doc.location.get("page"), int)
                    else (1, json.dumps(safe_config(doc.location), sort_keys=True))
                ),
            )
            grouped.append(
                (
                    replace(
                        ordered_artifacts[0],
                        content="\n\n".join(doc.content for doc in ordered_artifacts),
                    ),
                    ordered_artifacts,
                )
            )
        return grouped

    async def plan(
        self, scope: str, documents: Sequence[RawDocument], is_authoritative: bool = True
    ) -> SyncPlan:
        plan = SyncPlan()
        seen_ids: set[str] = set()
        for doc, artifacts in self._group_documents(documents):
            doc_id = doc.external_id or doc.source_name
            seen_ids.add(doc_id)
            chash = (
                self._compute_hash(doc.content)
                if len(artifacts) == 1
                else self._fingerprint(
                    [[artifact.location, artifact.content] for artifact in artifacts]
                )
            )
            rev = (
                doc.metadata.get("revision")
                or doc.metadata.get("etag")
                or doc.metadata.get("version")
            )
            existing = await self._manifest_store.get(scope, doc_id)
            if existing is None or existing.status == ManifestStatus.TOMBSTONED:
                action = "added"
            elif (
                existing.status != ManifestStatus.ACTIVE
                or existing.chunker_fingerprint != self._chunker_fingerprint()
                or existing.embedding_fingerprint != self._embedding_fingerprint()
                or existing.extractor_fingerprint != self._extractor_fingerprint(doc)
                or existing.content_hash != chash
            ):
                action = "changed"
            elif existing.metadata != doc.metadata or existing.provenance != doc.provenance:
                action = "metadata_only"
            else:
                action = "unchanged"
            item = SyncPlanItem(
                doc_id, action, doc, chash, rev, artifacts, existing.fence_token if existing else 0
            )
            getattr(plan, action).append(item)
        if is_authoritative:
            for rec in await self._manifest_store.list(scope):
                if rec.status != ManifestStatus.TOMBSTONED and rec.document_id not in seen_ids:
                    plan.removed.append(
                        SyncPlanItem(rec.document_id, "removed", expected_fence=rec.fence_token)
                    )
        return plan

    def _emit(self, kind: EventKind, scope: str, token: FenceToken, **kwargs: Any) -> None:
        # An exporter outage must not abandon a lease or invalidate committed data.
        try:
            self._event_sink.emit(
                TelemetryEvent(
                    event_kind=kind,
                    stage="sync",
                    source_id=scope,
                    run_id=f"sync-{token.epoch}",
                    **kwargs,
                )
            )
        except Exception:
            logger.warning("Sync telemetry sink failed", exc_info=False)

    async def _heartbeat(self, token: FenceToken) -> None:
        while True:
            await asyncio.sleep(self._lease_ttl_seconds / 3)
            try:
                renewed = await self._state_store.renew_lease(token, self._lease_ttl_seconds)
            except Exception:
                logger.warning("Lease renewal failed; activation still requires a valid fence")
                return
            if renewed is None:
                return

    async def _prepare_item(
        self,
        scope: str,
        item: SyncPlanItem,
        generation: int,
        token: FenceToken,
        existing_vectors: list[list[float]] | None = None,
    ) -> tuple[ManifestRecord, list[dict[str, Any]]]:
        doc = item.raw_document
        assert doc is not None
        chunks = []
        for artifact in item.artifacts or [doc]:
            for chunk in self._chunker.chunk(artifact.content):
                metadata = {
                    **artifact.metadata,
                    **chunk.metadata,
                }
                if artifact.provenance:
                    metadata["provenance"] = dict(artifact.provenance)
                if artifact.location is not None:
                    metadata["location"] = artifact.location
                    if isinstance(artifact.location, dict):
                        metadata.update(artifact.location)
                # Location and source metadata cannot replace coordinator-owned fields.
                metadata.update(
                    scope=scope,
                    doc_id=item.document_id,
                    generation=generation,
                    active=False,
                    source=artifact.source_name,
                )
                if artifact.external_id is not None:
                    metadata["external_id"] = artifact.external_id
                chunks.append({"text": chunk.text, "metadata": metadata})
        if not chunks:
            raise SyncError("Document produced no chunks; retaining the previous active version")
        profile = getattr(self._embedder, "profile", None)
        dim = (
            profile.dimension
            if profile is not None
            else get_embedding_dimension(self._settings.embedding_model)
        )
        embeddings = list(existing_vectors or [])
        if existing_vectors is None:
            fn = self._embedder
            for offset in range(0, len(chunks), self._batch_size):
                batch = chunks[offset : offset + self._batch_size]
                vectors = await fn([chunk["text"] for chunk in batch])
                if len(vectors) != len(batch):
                    raise ValueError("Embedding response count does not match chunk count")
                embeddings.extend(vectors)
        if len(embeddings) != len(chunks):
            raise ValueError("Stored embedding count does not match unchanged document chunks")
        nonce = uuid4().hex
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            validate_embedding(vector, expected_dim=dim)
            chunk.update(
                id=f"{scope}:{item.document_id}:g{generation}:e{token.epoch}:{nonce}:{index}",
                embedding=vector,
            )
        record = ManifestRecord(
            scope=scope,
            document_id=item.document_id,
            observed_revision=item.observed_revision,
            content_hash=item.content_hash,
            extractor_fingerprint=self._extractor_fingerprint(doc),
            chunker_fingerprint=self._chunker_fingerprint(),
            embedding_fingerprint=self._embedding_fingerprint(),
            active_generation=generation,
            status=ManifestStatus.ACTIVE,
            chunk_ids=[chunk["id"] for chunk in chunks],
            metadata=dict(doc.metadata),
            provenance=dict(doc.provenance),
            last_successful_observation=time.time(),
            fence_token=item.expected_fence,
        )
        return record, chunks

    async def _checkpoint(self, report: SyncReport, token: FenceToken) -> None:
        checkpoint = {
            "phase": "completed" if report.success else "incomplete",
            "authoritative": report.authoritative,
            "last_error": "sync_incomplete" if not report.success else None,
            "added": report.added_count,
            "changed": report.changed_count,
            "removed": report.removed_count,
            "observed_at": time.time(),
        }
        previous = await self._state_store.get_checkpoint(report.scope) or {}
        checkpoint["last_successful_sync"] = (
            time.time() if report.success else previous.get("last_successful_sync")
        )
        if not await self._state_store.save_checkpoint(report.scope, checkpoint, token):
            raise StaleWorkerError("Cannot persist checkpoint after lease takeover")

    async def run_sync(  # noqa: PLR0915
        self,
        scope: str,
        source: Any,
        is_authoritative: bool | None = None,
        cursor: Any = None,
        limit: int | None = None,
    ) -> SyncReport:
        if not hasattr(self._state_store, "fenced"):
            raise FencingError("SyncStateStore must implement a fenced commit context")
        if not hasattr(self._manifest_store, "compare_and_swap_many"):
            raise FencingError("ManifestStore must implement atomic batch commits")
        token = await self._state_store.acquire_lease(
            scope, self._worker_id, self._lease_ttl_seconds
        )
        if token is None:
            raise LeaseAcquisitionError(f"Could not acquire lease for scope {scope!r}")
        heartbeat = asyncio.create_task(self._heartbeat(token))
        report = SyncReport(scope=scope, success=False, fence_epoch=token.epoch)
        uncommitted: set[str] = set()
        retired: list[str] = []
        self._emit(EventKind.SCAN_STARTED, scope, token)
        try:
            result = await self._scan_source(source, cursor, limit)
            report.authoritative = result.authoritative and is_authoritative is not False
            report.errors.extend(_error_record(error) for error in result.errors)
            plan = await self.plan(scope, result, report.authoritative)
            for action in ("added", "changed", "removed", "metadata_only", "unchanged"):
                setattr(report, f"{action}_count", len(getattr(plan, action)))
            for doc in result:
                self._emit(
                    EventKind.DOCUMENT_LOADED,
                    scope,
                    token,
                    document_id=doc.external_id or doc.source_name,
                )
            for item in plan.unchanged:
                self._emit(EventKind.DOCUMENT_SKIPPED, scope, token, document_id=item.document_id)
            active_gen = await self._state_store.get_active_generation(scope)
            prepared: list[tuple[SyncPlanItem, ManifestRecord, list[dict[str, Any]]]] = []
            for item in plan.added + plan.changed + plan.metadata_only:
                try:
                    existing = await self._manifest_store.get(scope, item.document_id)
                    generation = (
                        active_gen + 1
                        if self._activation_mode == ActivationMode.COLLECTION
                        else ((existing.active_generation if existing else 0) + 1)
                    )
                    existing_vectors = None
                    if item.action == "metadata_only":
                        assert existing is not None
                        by_id = await self._vector_store.get_embeddings(existing.chunk_ids)
                        existing_vectors = [by_id[chunk_id] for chunk_id in existing.chunk_ids]
                    record, chunks = await self._prepare_item(
                        scope, item, generation, token, existing_vectors
                    )
                    if not await self._state_store.validate_token(token):
                        raise StaleWorkerError("Lease lost during embedding")
                    uncommitted.update(record.chunk_ids)
                    await self._vector_store.upsert(chunks)
                    if existing_vectors is None:
                        report.embedded_count += len(chunks)
                    if self._job_store is not None:
                        await self._job_store.enqueue(
                            {
                                "idempotency_key": (
                                    f"{scope}:{item.document_id}:{generation}:e{token.epoch}:activate"
                                ),
                                "scope": scope,
                                "document_id": item.document_id,
                                "generation": generation,
                                "epoch": token.epoch,
                                "status": "pending",
                                "operation": "reconcile_generation",
                                "chunk_ids": record.chunk_ids,
                            }
                        )
                    prepared.append((item, record, chunks))
                except StaleWorkerError:
                    raise
                except Exception as exc:
                    report.errors.append({"doc_id": item.document_id, **_error_record(exc)})
                    if self._activation_mode == ActivationMode.COLLECTION:
                        raise
            if not await self._state_store.validate_token(token):
                raise StaleWorkerError("Lease expired before activation")
            try:
                async with self._state_store.fenced(token):
                    await self._activate(scope, plan, prepared, token, report, retired, uncommitted)
                    report.success = (
                        not report.errors
                        and result.complete
                        and result.status
                        in {
                            "ok",
                            "empty",
                            "success",
                            "empty_success",
                        }
                    )
                    await self._checkpoint(report, token)
            except ValueError as exc:
                if not await self._state_store.validate_token(token):
                    raise StaleWorkerError("Lease lost before fenced commit") from exc
                raise
            return report
        except Exception as exc:
            self._emit(EventKind.SYNC_FAILED, scope, token, error=exc)
            if await self._state_store.validate_token(token):
                await self._checkpoint(report, token)
            raise
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            await self._state_store.release_lease(token)
            await self.garbage_collect(scope, [*retired, *uncommitted])

    async def _activate(  # noqa: PLR0913, PLR0917
        self,
        scope: str,
        plan: SyncPlan,
        prepared: list[tuple[SyncPlanItem, ManifestRecord, list[dict[str, Any]]]],
        token: FenceToken,
        report: SyncReport,
        retired: list[str],
        uncommitted: set[str],
    ) -> None:
        updates: list[tuple[ManifestRecord, int]] = []
        old_ids: list[str] = []
        for item, record, chunks in prepared:
            current = await self._manifest_store.get(scope, item.document_id)
            if (current.fence_token if current else 0) != item.expected_fence:
                raise FencingError("Document changed while preparing its generation")
            if current:
                old_ids.extend(current.chunk_ids)
            # Publish vectors before changing visibility. The manifest still hides their IDs.
            await self._vector_store.upsert(
                [{**chunk, "metadata": {**chunk["metadata"], "active": True}} for chunk in chunks]
            )
            updates.append((record, item.expected_fence))
        for item in plan.removed:
            record = await self._manifest_store.get(scope, item.document_id)
            if record is None or record.fence_token != item.expected_fence:
                raise FencingError("Document changed before tombstoning")
            old_ids.extend(record.chunk_ids)
            updates.append(
                (
                    replace(
                        record, status=ManifestStatus.TOMBSTONED, active_generation=0, chunk_ids=[]
                    ),
                    item.expected_fence,
                )
            )
        if updates:
            if not await self._manifest_store.compare_and_swap_many(updates):
                raise FencingError("Manifest commit compare-and-swap failed")
            # From here these IDs are committed and must never be garbage-collected on failure.
            for _, record, chunks in prepared:
                uncommitted.difference_update(record.chunk_ids)
                report.chunk_count += len(chunks)
                self._emit(
                    EventKind.GENERATION_ACTIVATED,
                    scope,
                    token,
                    document_id=record.document_id,
                    generation=record.active_generation,
                    counts={"chunks": len(chunks)},
                )
            retired.extend(old_ids)
            if self._activation_mode == ActivationMode.COLLECTION:
                generation = await self._state_store.advance_generation(
                    scope, await self._state_store.get_active_generation(scope)
                )
                if generation < 0:
                    raise FencingError("Collection generation compare-and-swap failed")
                report.active_generation = generation
        if self._activation_mode == ActivationMode.DOCUMENT:
            report.active_generation = await self._state_store.get_active_generation(scope)

    async def _scan_source(
        self, source: Any, cursor: Any = None, limit: int | None = None
    ) -> LoadResult:
        if isinstance(source, LoadResult):
            return source
        if hasattr(source, "load_page"):
            first = True

            async def page(*, cursor=None, limit=None):
                nonlocal first
                actual_cursor = start_cursor if first else cursor
                first = False
                return await source.load_page(cursor=actual_cursor, limit=limit)

            start_cursor = cursor
            result = await collect_pages(
                SimpleNamespace(load_page=page),
                max_pages=getattr(source, "max_pages", None) or 10000,
                page_size=limit,
            )
            if start_cursor is not None:
                result.complete = False
            return result
        if hasattr(source, "load"):
            parameters = inspect.signature(source.load).parameters
            options = {
                key: value
                for key, value in {"cursor": cursor, "limit": limit}.items()
                if key in parameters
            }
            result = await source.load(**options)
            if isinstance(result, LoadResult):
                return result
            return LoadResult(list(result), complete=cursor is None)
        return LoadResult(list(source), complete=cursor is None)

    async def revoke_access(self, scope: str, document_id: str) -> bool:
        while True:
            record = await self._manifest_store.get(scope, document_id)
            if record is None:
                return False
            tombstone = replace(
                record, status=ManifestStatus.TOMBSTONED, active_generation=0, chunk_ids=[]
            )
            if await self._manifest_store.compare_and_swap(tombstone, record.fence_token):
                await self.garbage_collect(scope, record.chunk_ids)
                return True

    async def garbage_collect(self, scope: str, retired_chunk_ids: list[str] | None = None) -> int:
        if not retired_chunk_ids:
            return 0
        try:
            return await self._vector_store.delete(ids=retired_chunk_ids)
        except Exception:
            logger.warning("Physical cleanup deferred for scope %s", scope)
            return 0
