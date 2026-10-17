"""Health checking and status reporting for ragit subsystems.

Exposes HealthReport states ready, degraded, unavailable, and unknown with
per-component reasons. Checks configuration compatibility, backend
connectivity, active document/chunk counts, last sync, scan completeness,
job queues, and evaluation results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ragit.embeddings.dimensions import _DIMENSIONS, get_embedding_dimension
from ragit.sync.manifest import ManifestStatus

if TYPE_CHECKING:
    from ragit.embeddings.base import EmbeddingFunction
    from ragit.jobs.outbox import EmbeddingJobStore
    from ragit.stores.base import VectorStore
    from ragit.sync.manifest import ManifestStore
    from ragit.sync.state import SyncStateStore

logger = logging.getLogger(__name__)


class HealthState(StrEnum):
    """Subsystem or system-wide health classification."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


# Precedence order for state rollup: higher index = worse state
_STATE_SEVERITY: dict[HealthState, int] = {
    HealthState.READY: 0,
    HealthState.UNKNOWN: 1,
    HealthState.DEGRADED: 2,
    HealthState.UNAVAILABLE: 3,
}


@dataclass
class ComponentHealth:
    """Health status and diagnostic details of an individual component.

    Attributes:
        name: Identifying name of the subsystem (e.g. "vector_store").
        state: Subsystem health state.
        reason: Explanation when state is degraded or unavailable.
        latency_ms: Execution duration of the health probe in milliseconds.
        details: Diagnostic key-value measurements.
    """

    name: str
    state: HealthState
    reason: str = ""
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Rollup health report representing overall system readiness.

    Attributes:
        state: Worst-case aggregate HealthState across all components.
        components: Map of component names to ComponentHealth objects.
        reasons: Aggregated non-ready diagnostic messages.
        timestamp: Unix epoch timestamp when health was evaluated.
    """

    state: HealthState = HealthState.READY
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __init__(  # noqa: PLR0913
        self,
        state: HealthState | None = None,
        components: Sequence[ComponentHealth] | Mapping[str, ComponentHealth] | None = None,
        reasons: list[str] | None = None,
        timestamp: float | None = None,
        *,
        overall: HealthState | None = None,
    ) -> None:
        if isinstance(components, (list, tuple)):
            comp_map = {c.name: c for c in components}
        elif isinstance(components, (dict, Mapping)):
            comp_map = dict(components)
        else:
            comp_map = {}

        self.components = comp_map
        self.timestamp = timestamp if timestamp is not None else time.time()

        if overall is not None:
            self.state = overall
        elif state is not None:
            self.state = state
        elif comp_map:
            worst_severity = max(_STATE_SEVERITY[c.state] for c in comp_map.values())
            self.state = next(st for st, sev in _STATE_SEVERITY.items() if sev == worst_severity)
        else:
            self.state = HealthState.READY

        if reasons is not None:
            self.reasons = list(reasons)
        else:
            self.reasons = [
                f"[{c.name}] {c.reason}"
                for c in comp_map.values()
                if c.reason and c.state != HealthState.READY
            ]

    @property
    def is_healthy(self) -> bool:
        """Return True if all components are ready."""
        return self.state == HealthState.READY

    @property
    def is_ready(self) -> bool:
        """Return True if all components are healthy and ready."""
        return self.state == HealthState.READY

    @property
    def is_degraded(self) -> bool:
        """Return True if system is operational with warnings or stale components."""
        return self.state == HealthState.DEGRADED

    @property
    def is_unavailable(self) -> bool:
        """Return True if a critical subsystem is offline or incompatible."""
        return self.state == HealthState.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "state": str(self.state),
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "components": {
                name: {
                    "state": str(comp.state),
                    "reason": comp.reason,
                    "latency_ms": comp.latency_ms,
                    "details": comp.details,
                }
                for name, comp in self.components.items()
            },
        }


class HealthChecker:
    """Orchestrates comprehensive health checks across components."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        settings: Any | None = None,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingFunction | None = None,
        sync_state_store: SyncStateStore | None = None,
        state_store: SyncStateStore | None = None,
        manifest_store: ManifestStore | None = None,
        job_store: EmbeddingJobStore | None = None,
        scope: str = "default",
        allow_empty: bool = False,
        timeout_seconds: float = 2.0,
    ) -> None:
        """Initialize HealthChecker.

        Args:
            settings: Host configuration settings.
            vector_store: Storage backend for vector index operations.
            embedder: Embedding function callable.
            sync_state_store: Store holding sync checkpoints and epochs.
            state_store: Alias for sync_state_store.
            manifest_store: Manifest store tracking document records.
            job_store: Background embedding outbox / job queue store.
            scope: Scoped isolation namespace (tenant, collection).
            allow_empty: When True, 0 indexed chunks is treated as deliberately empty (READY).
                When False, 0 indexed chunks on a configured collection is DEGRADED.
            timeout_seconds: Strict bound for all network/database probes.
        """
        self.settings = settings
        self.vector_store = vector_store
        self.embedder = embedder
        self.sync_state_store = sync_state_store or state_store
        self.manifest_store = manifest_store
        self.job_store = job_store
        self.scope = scope
        self.allow_empty = allow_empty
        self.timeout_seconds = timeout_seconds

    def check_configuration(self) -> ComponentHealth:
        """Validate configuration settings and dimensional compatibility."""
        start = time.perf_counter()
        if self.settings is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="configuration",
                state=HealthState.UNKNOWN,
                reason="No settings configured",
                latency_ms=latency,
            )

        model = getattr(self.settings, "embedding_model", None)
        details: dict[str, Any] = {
            "embedding_model": model,
        }

        if model is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="configuration",
                state=HealthState.DEGRADED,
                reason="Embedding model not specified in settings",
                latency_ms=latency,
                details=details,
            )

        if model not in _DIMENSIONS:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="configuration",
                state=HealthState.DEGRADED,
                reason=(f"Unknown embedding model {model!r} (not in registered dimensions table)"),
                latency_ms=latency,
                details=details,
            )

        try:
            expected_dim = get_embedding_dimension(model)
            details["expected_dimension"] = expected_dim
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="configuration",
                state=HealthState.DEGRADED,
                reason=f"Unknown embedding model {model!r}: {exc}",
                latency_ms=latency,
                details=details,
            )

        # Check embedding dimension compatibility with vector store
        if self.vector_store is not None:
            store_dim = getattr(self.vector_store, "embedding_dim", None)
            details["store_dimension"] = store_dim

            if store_dim is not None and expected_dim != store_dim:
                latency = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    name="configuration",
                    state=HealthState.UNAVAILABLE,
                    reason=(
                        f"Dimension mismatch: store configured for {store_dim} "
                        f"but model {model!r} produces {expected_dim}"
                    ),
                    latency_ms=latency,
                    details=details,
                )

        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth(
            name="configuration",
            state=HealthState.READY,
            reason="Configuration compatible",
            latency_ms=latency,
            details=details,
        )

    async def check_backend_connectivity(self) -> ComponentHealth:
        """Probe storage backend connectivity and capabilities."""
        start = time.perf_counter()
        if self.vector_store is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="backend_connectivity",
                state=HealthState.UNKNOWN,
                reason="No vector store configured",
                latency_ms=latency,
            )

        try:
            # Bounded connectivity ping
            await asyncio.wait_for(
                self.vector_store.count(strict=True),
                timeout=self.timeout_seconds,
            )
            latency = (time.perf_counter() - start) * 1000

            caps = getattr(self.vector_store, "capabilities", None)
            details: dict[str, Any] = {
                "store_type": type(self.vector_store).__name__,
                "has_capabilities": caps is not None,
            }
            return ComponentHealth(
                name="backend_connectivity",
                state=HealthState.READY,
                latency_ms=latency,
                details=details,
            )
        except TimeoutError:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="backend_connectivity",
                state=HealthState.UNAVAILABLE,
                reason=f"Backend connectivity probe timed out after {self.timeout_seconds}s",
                latency_ms=latency,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="backend_connectivity",
                state=HealthState.UNAVAILABLE,
                reason=f"Backend connectivity failure: {exc}",
                latency_ms=latency,
            )

    async def check_counts(self, scope: str | None = None) -> ComponentHealth:
        """Inspect active indexed chunks distinguishing deliberate vs unexpected zero counts."""
        start = time.perf_counter()
        target_scope = scope or self.scope
        docs_count = 0
        chunk_count = 0

        if self.manifest_store is not None:
            try:
                if hasattr(self.manifest_store, "list"):
                    active_records = await asyncio.wait_for(
                        self.manifest_store.list(target_scope, status=ManifestStatus.ACTIVE),
                        timeout=self.timeout_seconds,
                    )
                else:
                    active_records = await asyncio.wait_for(
                        self.manifest_store.list_active(target_scope),
                        timeout=self.timeout_seconds,
                    )
                docs_count = len(active_records)
                chunk_count = sum(len(r.chunk_ids) for r in active_records)
            except Exception as exc:  # noqa: BLE001
                latency = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    name="active_counts",
                    state=HealthState.UNAVAILABLE,
                    reason=f"Failed to query manifest records: {exc}",
                    latency_ms=latency,
                )
        elif self.vector_store is not None:
            try:
                chunk_count = await asyncio.wait_for(
                    self.vector_store.count(strict=True),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                latency = (time.perf_counter() - start) * 1000
                return ComponentHealth(
                    name="active_counts",
                    state=HealthState.UNAVAILABLE,
                    reason=f"Failed to query chunk counts: {exc}",
                    latency_ms=latency,
                )
        else:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="active_counts",
                state=HealthState.UNKNOWN,
                reason="Neither manifest store nor vector store configured",
                latency_ms=latency,
            )

        latency = (time.perf_counter() - start) * 1000
        details = {
            "active_documents": docs_count,
            "active_chunks": chunk_count,
            "allow_empty": self.allow_empty,
        }

        if chunk_count > 0 or docs_count > 0:
            return ComponentHealth(
                name="active_counts",
                state=HealthState.READY,
                latency_ms=latency,
                details=details,
            )

        # 0 chunks / 0 docs
        if self.allow_empty:
            details["deliberate_empty"] = True
            return ComponentHealth(
                name="active_counts",
                state=HealthState.READY,
                reason="Collection is deliberately empty",
                latency_ms=latency,
                details=details,
            )

        return ComponentHealth(
            name="active_counts",
            state=HealthState.DEGRADED,
            reason="Configured collection unexpectedly indexing zero chunks",
            latency_ms=latency,
            details=details,
        )

    async def check_sync_status(  # noqa: PLR0911
        self, scope: str | None = None
    ) -> ComponentHealth:
        """Check last successful sync and scan completeness."""
        start = time.perf_counter()
        if self.sync_state_store is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="sync_status",
                state=HealthState.UNKNOWN,
                reason="No sync state store configured",
                latency_ms=latency,
            )

        target_scope = scope or self.scope
        try:
            checkpoint = None
            if hasattr(self.sync_state_store, "get_checkpoint"):
                checkpoint = await asyncio.wait_for(
                    self.sync_state_store.get_checkpoint(target_scope),
                    timeout=self.timeout_seconds,
                )
            elif hasattr(self.sync_state_store, "get_state"):
                state = await asyncio.wait_for(
                    self.sync_state_store.get_state(target_scope),
                    timeout=self.timeout_seconds,
                )
                if state is not None:
                    checkpoint = {
                        "phase": "completed" if state.last_sync_timestamp > 0 else "initial",
                        "authoritative": state.is_authoritative,
                        "last_error": state.last_error,
                    }

            latency = (time.perf_counter() - start) * 1000
            if checkpoint is None:
                if self.allow_empty:
                    return ComponentHealth(
                        name="sync_status",
                        state=HealthState.READY,
                        reason="Collection is deliberately empty; initial sync not yet run",
                        latency_ms=latency,
                    )
                return ComponentHealth(
                    name="sync_status",
                    state=HealthState.UNKNOWN,
                    reason=f"No sync checkpoint found for configured scope {target_scope!r}",
                    latency_ms=latency,
                )

            is_authoritative = checkpoint.get("authoritative", True)
            last_error = checkpoint.get("last_error")
            phase = checkpoint.get("phase", "")

            details: dict[str, Any] = {
                "checkpoint": checkpoint,
                "is_authoritative": is_authoritative,
            }

            if last_error:
                return ComponentHealth(
                    name="sync_status",
                    state=HealthState.DEGRADED,
                    reason=f"Last sync recorded failure: {last_error}",
                    latency_ms=latency,
                    details=details,
                )

            if not is_authoritative or phase not in ("completed", "scan_complete"):
                return ComponentHealth(
                    name="sync_status",
                    state=HealthState.DEGRADED,
                    reason="Last sync completed with incomplete or non-authoritative scan coverage",
                    latency_ms=latency,
                    details=details,
                )

            return ComponentHealth(
                name="sync_status",
                state=HealthState.READY,
                reason="Sync status healthy and authoritative",
                latency_ms=latency,
                details=details,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="sync_status",
                state=HealthState.UNAVAILABLE,
                reason=f"Failed to query sync state: {exc}",
                latency_ms=latency,
            )

    async def check_jobs(self, scope: str | None = None) -> ComponentHealth:
        """Inspect background embedding jobs for pending or dead-letter items."""
        start = time.perf_counter()
        if self.job_store is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="job_queue",
                state=HealthState.READY,
                reason="Job queue store not configured (in-process execution)",
                latency_ms=latency,
            )

        try:
            dead_letter_count = 0
            pending_count = 0
            details: dict[str, Any] = {}

            if hasattr(self.job_store, "get_status_counts"):
                counts = await asyncio.wait_for(
                    self.job_store.get_status_counts(), timeout=self.timeout_seconds
                )
                details = dict(counts)
                dead_letter_count = counts.get("dead_letter", 0) + counts.get("failed", 0)
                pending_count = counts.get("pending", 0)
            elif hasattr(self.job_store, "jobs"):
                for j in getattr(self.job_store, "jobs", []):
                    status = getattr(j, "status", "")
                    details[status] = details.get(status, 0) + 1
                    if status in ("dead_letter", "failed"):
                        dead_letter_count += 1
                    elif status == "pending":
                        pending_count += 1

            latency = (time.perf_counter() - start) * 1000

            if dead_letter_count > 0:
                return ComponentHealth(
                    name="job_queue",
                    state=HealthState.DEGRADED,
                    reason=(
                        f"Found {dead_letter_count} dead-letter embedding jobs "
                        "requiring intervention"
                    ),
                    latency_ms=latency,
                    details=details,
                )

            return ComponentHealth(
                name="job_queue",
                state=HealthState.READY,
                latency_ms=latency,
                details=details or {"pending_count": pending_count},
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="job_queue",
                state=HealthState.DEGRADED,
                reason=f"Could not inspect job queue: {exc}",
                latency_ms=latency,
            )

    def check_evaluations(
        self,
        last_eval_results: Mapping[str, Any] | None = None,
        thresholds: Mapping[str, float] | None = None,
    ) -> ComponentHealth:
        """Inspect recent evaluation or benchmark results."""
        if last_eval_results is None:
            return ComponentHealth(
                name="evaluation",
                state=HealthState.UNKNOWN,
                reason="No evaluation results recorded",
            )

        details = dict(last_eval_results)
        if last_eval_results.get("passed") is False:
            return ComponentHealth(
                name="evaluation",
                state=HealthState.DEGRADED,
                reason="Evaluation failed or below threshold",
                details=details,
            )

        thresh = thresholds or {"accuracy": 0.5, "recall": 0.5}
        for metric, min_val in thresh.items():
            val = last_eval_results.get(metric)
            if val is not None and float(val) < min_val:
                return ComponentHealth(
                    name="evaluation",
                    state=HealthState.DEGRADED,
                    reason=(
                        f"Evaluation metric {metric!r} ({val:.3f}) fell below "
                        f"threshold ({min_val:.3f})"
                    ),
                    details=details,
                )

        return ComponentHealth(
            name="evaluation",
            state=HealthState.READY,
            reason="Evaluation passed",
            details=details,
        )

    async def check_expensive_consistency(
        self, scope: str | None = None, limit: int = 100
    ) -> ComponentHealth:
        """Perform bounded, explicit consistency verification across manifest and store."""
        start = time.perf_counter()
        target_scope = scope or self.scope
        if self.manifest_store is None:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="expensive_consistency",
                state=HealthState.UNKNOWN,
                reason="Manifest store unavailable for consistency probe",
                latency_ms=latency,
            )

        try:

            async def _probe() -> tuple[int, int]:
                if hasattr(self.manifest_store, "list"):
                    records = await self.manifest_store.list(
                        target_scope, status=ManifestStatus.ACTIVE
                    )
                else:
                    records = await self.manifest_store.list_active(target_scope)
                active_count = len(records[:limit])
                store_count = 0
                if self.vector_store is not None:
                    store_count = await self.vector_store.count(strict=True)
                return active_count, store_count

            active_docs, store_chunks = await asyncio.wait_for(
                _probe(), timeout=self.timeout_seconds
            )
            latency = (time.perf_counter() - start) * 1000
            details = {
                "manifest_active_documents": active_docs,
                "vector_store_chunks": store_chunks,
                "bounded_limit": limit,
            }
            return ComponentHealth(
                name="expensive_consistency",
                state=HealthState.READY,
                latency_ms=latency,
                details=details,
            )
        except TimeoutError:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="expensive_consistency",
                state=HealthState.DEGRADED,
                reason=f"Consistency check timed out after {self.timeout_seconds}s",
                latency_ms=latency,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name="expensive_consistency",
                state=HealthState.DEGRADED,
                reason=f"Consistency check failed: {exc}",
                latency_ms=latency,
            )

    async def check_all(  # noqa: PLR0913
        self,
        scope: str | None = None,
        *,
        include_expensive: bool = False,
        expensive_limit: int = 100,
        last_evaluation_results: Mapping[str, Any] | None = None,
        last_eval_results: Mapping[str, Any] | None = None,
        eval_thresholds: Mapping[str, float] | None = None,
    ) -> HealthReport:
        """Run all health checks and return an aggregate HealthReport."""
        return await self.check_health(
            scope=scope,
            include_expensive=include_expensive,
            expensive_limit=expensive_limit,
            last_evaluation_results=last_evaluation_results or last_eval_results,
            eval_thresholds=eval_thresholds,
        )

    async def check_health(  # noqa: PLR0913
        self,
        scope: str | None = None,
        *,
        include_expensive: bool = False,
        expensive_limit: int = 100,
        last_evaluation_results: Mapping[str, Any] | None = None,
        last_eval_results: Mapping[str, Any] | None = None,
        eval_thresholds: Mapping[str, float] | None = None,
    ) -> HealthReport:
        """Run health checks across components and return an aggregate HealthReport.

        Args:
            scope: Optional target scope to evaluate.
            include_expensive: When True, executes bounded, deep consistency checks.
                Defaults to False to prevent unneeded I/O during routine liveness polling.
            expensive_limit: Bounded document count limit for expensive probe.
            last_evaluation_results: Optional recent benchmark or evaluation metrics.
            last_eval_results: Alias for last_evaluation_results.
            eval_thresholds: Optional minimum thresholds for evaluation metrics.

        Returns:
            :class:`HealthReport` with composite state and component breakdowns.
        """
        target_scope = scope or self.scope
        eval_res = last_evaluation_results or last_eval_results

        tasks = [
            self.check_backend_connectivity(),
            self.check_counts(scope=target_scope),
            self.check_sync_status(scope=target_scope),
            self.check_jobs(scope=target_scope),
        ]
        if include_expensive:
            tasks.append(
                self.check_expensive_consistency(scope=target_scope, limit=expensive_limit)
            )

        components_list = [self.check_configuration()]
        async_results = await asyncio.gather(*tasks)
        components_list.extend(async_results)

        if eval_res is not None:
            components_list.append(self.check_evaluations(eval_res, eval_thresholds))

        return HealthReport(components=components_list)
