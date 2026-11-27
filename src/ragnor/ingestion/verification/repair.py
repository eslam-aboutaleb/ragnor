"""Deterministic, bounded region repair state machine.

A repair action changes *how a region is parsed* (crop, resolution, serializer,
extractor), never what a value is. :class:`RepairAction` has no text or value
parameters, and the machine never edits an artifact: it either promotes a
re-extracted artifact that passes verification, or marks the region
``needs_review``. Attempts are written ahead of execution to the host's store,
so crashes, cancellation and replays cannot grant extra retries.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ragnor.ingestion.extraction_types import (
    BBox,
    ExtractionRegion,
    ExtractionTable,
    checked_bbox,
)
from ragnor.ingestion.verification.audit import AttemptRecord, AttemptStore
from ragnor.ingestion.verification.geometry import iou
from ragnor.ingestion.verification.report import Finding

logger = logging.getLogger("ragnor.ingestion.verification")
MAX_ATTEMPTS_CEILING = 3
_TOKEN = re.compile(r"^[a-z0-9_\-]{1,40}$")


class RepairActionKind(StrEnum):
    ADJUST_CROP = "adjust_crop"
    RESEGMENT = "resegment"
    RAISE_RESOLUTION = "raise_resolution"
    CHANGE_SERIALIZER = "change_serializer"
    TILE_DENSE_TABLE = "tile_dense_table"
    CHANGE_EXTRACTOR = "change_extractor"


_PARAM_KEYS: dict[RepairActionKind, frozenset[str]] = {
    RepairActionKind.ADJUST_CROP: frozenset({"margin"}),
    RepairActionKind.RESEGMENT: frozenset({"strategy"}),
    RepairActionKind.RAISE_RESOLUTION: frozenset({"dpi"}),
    RepairActionKind.CHANGE_SERIALIZER: frozenset({"serializer"}),
    RepairActionKind.TILE_DENSE_TABLE: frozenset({"tiles"}),
    RepairActionKind.CHANGE_EXTRACTOR: frozenset({"extractor"}),
}
# Which actions may ever be tried for which failing check.
ALLOWLIST: dict[str, frozenset[RepairActionKind]] = {
    "count": frozenset(
        {
            RepairActionKind.ADJUST_CROP,
            RepairActionKind.RESEGMENT,
            RepairActionKind.RAISE_RESOLUTION,
            RepairActionKind.CHANGE_EXTRACTOR,
        }
    ),
    "coverage": frozenset(
        {
            RepairActionKind.ADJUST_CROP,
            RepairActionKind.RESEGMENT,
            RepairActionKind.RAISE_RESOLUTION,
            RepairActionKind.CHANGE_EXTRACTOR,
        }
    ),
    "shape": frozenset(
        {
            RepairActionKind.CHANGE_EXTRACTOR,
            RepairActionKind.TILE_DENSE_TABLE,
            RepairActionKind.CHANGE_SERIALIZER,
            RepairActionKind.RAISE_RESOLUTION,
        }
    ),
    "arithmetic": frozenset(
        {
            RepairActionKind.RAISE_RESOLUTION,
            RepairActionKind.CHANGE_SERIALIZER,
            RepairActionKind.CHANGE_EXTRACTOR,
            RepairActionKind.TILE_DENSE_TABLE,
        }
    ),
}


@dataclass(frozen=True)
class RepairAction:
    kind: RepairActionKind
    params: tuple[tuple[str, int | float | str], ...] = ()

    def __post_init__(self) -> None:
        keys = [k for k, _ in self.params]
        if (
            len(set(keys)) != len(keys)
            or not set(keys) <= _PARAM_KEYS[self.kind]
            or list(self.params) != sorted(self.params)
        ):
            raise ValueError("Repair action parameters are not allowed for this action")
        for _, value in self.params:
            if isinstance(value, bool) or (isinstance(value, str) and not _TOKEN.match(value)):
                raise ValueError("Repair action parameters must be numbers or short tokens")
            if isinstance(value, int | float) and (not math.isfinite(value) or value <= 0):
                raise ValueError("Numeric repair parameters must be positive and finite")

    @classmethod
    def of(cls, kind: RepairActionKind, **params: int | float | str) -> RepairAction:
        return cls(kind, tuple(sorted(params.items())))

    @property
    def key(self) -> tuple[str, tuple[tuple[str, int | float | str], ...]]:
        return (self.kind.value, self.params)

    def param(self, name: str) -> int | float | str | None:
        return dict(self.params).get(name)


K = RepairActionKind
DEFAULT_LADDER: dict[str, tuple[RepairAction, ...]] = {
    "count": (
        RepairAction.of(K.ADJUST_CROP, margin=0.01),
        RepairAction.of(K.RESEGMENT, strategy="text"),
        RepairAction.of(K.RAISE_RESOLUTION, dpi=200),
    ),
    "coverage": (
        RepairAction.of(K.ADJUST_CROP, margin=0.02),
        RepairAction.of(K.RESEGMENT, strategy="lines"),
        RepairAction.of(K.RAISE_RESOLUTION, dpi=300),
    ),
    "shape": (
        RepairAction.of(K.CHANGE_EXTRACTOR, extractor="alternate"),
        RepairAction.of(K.TILE_DENSE_TABLE, tiles=2),
        RepairAction.of(K.CHANGE_SERIALIZER, serializer="html"),
    ),
    "arithmetic": (
        RepairAction.of(K.RAISE_RESOLUTION, dpi=300),
        RepairAction.of(K.CHANGE_SERIALIZER, serializer="html"),
        RepairAction.of(K.CHANGE_EXTRACTOR, extractor="alternate"),
    ),
}
_DEFAULT_COSTS = {K.RAISE_RESOLUTION: 2.0, K.CHANGE_EXTRACTOR: 2.0, K.TILE_DENSE_TABLE: 2.0}


@dataclass(frozen=True)
class RepairLimits:
    max_attempts_per_region: int = MAX_ATTEMPTS_CEILING
    max_total_attempts: int = 12
    max_cost: float = 10.0
    max_seconds: float = 60.0
    max_dpi: int = 300
    max_pixels: int = 25_000_000
    action_costs: Mapping[RepairActionKind, float] = field(
        default_factory=lambda: dict(_DEFAULT_COSTS)
    )

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts_per_region <= MAX_ATTEMPTS_CEILING:
            raise ValueError("max_attempts_per_region must be between 1 and 3")
        if any(
            type(v) is not int or v < 1
            for v in (self.max_total_attempts, self.max_dpi, self.max_pixels)
        ):
            raise ValueError("Repair limits must be positive integers")
        for value in (self.max_cost, self.max_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Repair cost and time limits must be positive and finite")

    def cost_of(self, action: RepairAction) -> float:
        return float(self.action_costs.get(action.kind, 1.0))


@dataclass(frozen=True)
class RegionRef:
    page: int
    region_id: str
    bbox: BBox
    failing_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        if not self.failing_checks:
            raise ValueError("A region to repair needs at least one failing check")


@dataclass(frozen=True)
class RegionArtifact:
    """Immutable re-extraction result for one region."""

    region_id: str
    bbox: BBox
    artifact_ref: str
    pixels: int = 0
    table: ExtractionTable | None = field(default=None, repr=False)
    regions: tuple[ExtractionRegion, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)


class RepairExecutor(Protocol):
    fingerprint: str

    def execute(
        self, region: RegionRef, action: RepairAction, *, max_pixels: int
    ) -> RegionArtifact:
        raise NotImplementedError


RegionVerifier = Callable[[RegionRef, RegionArtifact], Sequence[Finding]]


@dataclass(frozen=True)
class RegionResult:
    region_id: str
    status: str  # promoted | needs_review
    reason: str  # "" | attempts_exhausted | no_actions | budget | cancelled
    attempts: int
    artifact: RegionArtifact | None = None
    artifact_ref: str = ""
    final_bbox: BBox | None = None


@dataclass(frozen=True)
class RepairRunResult:
    run_id: str
    regions: tuple[RegionResult, ...]

    @property
    def promoted(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self.regions if r.status == "promoted")

    @property
    def needs_review(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self.regions if r.status == "needs_review")


class RepairStateMachine:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        executor: RepairExecutor,
        verifier: RegionVerifier,
        store: AttemptStore,
        limits: RepairLimits = RepairLimits(),
        ladder: Mapping[str, Sequence[RepairAction]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._executor = executor
        self._verifier = verifier
        self._store = store
        self.limits = limits
        self._clock = clock
        self._ladder = {k: tuple(v) for k, v in (ladder or DEFAULT_LADDER).items()}
        self._validate_ladder()

    def _validate_ladder(self) -> None:
        for check_id, actions in self._ladder.items():
            allowed = ALLOWLIST.get(check_id, frozenset())
            for action in actions:
                if action.kind not in allowed:
                    raise ValueError("Ladder action is not allowlisted for this check")
                dpi = action.param("dpi")
                if dpi is not None and float(dpi) > self.limits.max_dpi:
                    raise ValueError("Ladder DPI exceeds the configured maximum")

    def run(
        self,
        *,
        tenant_id: str,
        run_id: str,
        regions: Sequence[RegionRef],
        cancel: threading.Event | None = None,
    ) -> RepairRunResult:
        results = []
        for region in sorted(regions, key=lambda r: r.region_id):
            results.append(self._repair(tenant_id, run_id, region, cancel))
        return RepairRunResult(run_id, tuple(results))

    def _candidates(self, region: RegionRef) -> list[RepairAction]:
        seen: set[tuple[str, tuple[tuple[str, int | float | str], ...]]] = set()
        ordered = []
        for check_id in region.failing_checks:
            for action in self._ladder.get(check_id, ()):
                if action.key not in seen:
                    seen.add(action.key)
                    ordered.append(action)
        return ordered

    def _repair(  # noqa: PLR0911
        self,
        tenant: str,
        run_id: str,
        region: RegionRef,
        cancel: threading.Event | None,
    ) -> RegionResult:
        limits = self.limits
        history = self._store.list(tenant, run_id, region.region_id)
        done = next((r for r in history if r.outcome == "promoted"), None)
        if done is not None:
            return RegionResult(
                region.region_id, "promoted", "", len(history), artifact_ref=done.artifact_ref
            )
        # A crash mid-attempt leaves "started": it still consumed an attempt.
        for record in history:
            if record.outcome == "started":
                self._store.update(tenant, record.finished(outcome="interrupted"))
        history = self._store.list(tenant, run_id, region.region_id)
        tried = {(r.action_kind, r.action_params) for r in history}
        attempts = len(history)
        candidates = self._candidates(region)
        while attempts < limits.max_attempts_per_region:
            if cancel is not None and cancel.is_set():
                return RegionResult(region.region_id, "needs_review", "cancelled", attempts)
            action = next((a for a in candidates if a.key not in tried), None)
            if action is None:
                return RegionResult(region.region_id, "needs_review", "no_actions", attempts)
            run_records = self._store.list(tenant, run_id)
            spent = sum(r.cost for r in run_records)
            spent_ms = sum(r.duration_ms for r in run_records)
            if (
                len(run_records) >= limits.max_total_attempts
                or spent + limits.cost_of(action) > limits.max_cost
                or spent_ms / 1000.0 >= limits.max_seconds
            ):
                return RegionResult(region.region_id, "needs_review", "budget", attempts)
            attempts += 1
            tried.add(action.key)
            record = AttemptRecord(
                run_id,
                region.region_id,
                attempts,
                region.page,
                region.bbox,
                self._executor.fingerprint,
                action.kind.value,
                action.params,
                tuple(region.failing_checks),
                cost=limits.cost_of(action),
            )
            self._store.append(tenant, record)
            began = self._clock()
            artifact, outcome = self._attempt(region, action, cancel)
            self._store.update(
                tenant,
                record.finished(
                    outcome=outcome,
                    duration_ms=max(0, round((self._clock() - began) * 1000)),
                    artifact_ref=artifact.artifact_ref
                    if artifact and outcome == "promoted"
                    else "",
                ),
            )
            logger.info(
                "repair attempt region=%s n=%s outcome=%s", region.region_id, attempts, outcome
            )
            if outcome == "promoted" and artifact is not None:
                return RegionResult(
                    region.region_id,
                    "promoted",
                    "",
                    attempts,
                    artifact,
                    artifact.artifact_ref,
                    artifact.bbox,
                )
            if outcome == "cancelled":
                return RegionResult(region.region_id, "needs_review", "cancelled", attempts)
        return RegionResult(region.region_id, "needs_review", "attempts_exhausted", attempts)

    def _attempt(  # noqa: PLR0911
        self, region: RegionRef, action: RepairAction, cancel: threading.Event | None
    ) -> tuple[RegionArtifact | None, str]:
        try:
            artifact = self._executor.execute(region, action, max_pixels=self.limits.max_pixels)
        except Exception:  # noqa: BLE001 - executor errors never leak or publish partial output
            return None, "executor_failed"
        if cancel is not None and cancel.is_set():
            return None, "cancelled"
        if artifact.pixels > self.limits.max_pixels:
            return None, "resource_limit"
        if artifact.region_id != region.region_id or iou(artifact.bbox, region.bbox) <= 0:
            return None, "rebind_failed"
        try:
            findings = self._verifier(region, artifact)
        except Exception:  # noqa: BLE001
            return None, "verifier_failed"
        if any(f.severity == "error" for f in findings):
            return None, "still_failing"
        return artifact, "promoted"
