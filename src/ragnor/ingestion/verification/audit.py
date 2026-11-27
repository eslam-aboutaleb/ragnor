"""Bounded, sanitized repair-attempt audit records and tenant-scoped storage.

Records hold identifiers, geometry, check IDs and artifact references only.
Never source text, prompts, credentials or PII. The host owns persistence by
implementing :class:`AttemptStore`.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from ragnor.ingestion.extraction_types import BBox, checked_bbox

_IDENT = re.compile(r"^[A-Za-z0-9_.:\-]{1,120}$")
_REF = re.compile(r"^[A-Za-z0-9_.:/\-]{0,200}$")
OUTCOMES = frozenset(
    {
        "started",
        "promoted",
        "still_failing",
        "executor_failed",
        "verifier_failed",
        "resource_limit",
        "rebind_failed",
        "cancelled",
        "interrupted",
    }
)


@dataclass(frozen=True)
class AttemptRecord:
    run_id: str
    region_id: str
    attempt_no: int
    page: int
    bbox: BBox
    config_fingerprint: str
    action_kind: str
    action_params: tuple[tuple[str, int | float | str], ...]
    failed_checks: tuple[str, ...]
    outcome: str = "started"
    duration_ms: int = 0
    artifact_ref: str = ""
    cost: float = 0.0

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        for value in (self.run_id, self.region_id, self.config_fingerprint, self.action_kind):
            if not _IDENT.match(value):
                raise ValueError("Audit identifiers must be short opaque tokens")
        if not all(_IDENT.match(c) for c in self.failed_checks):
            raise ValueError("Audit check IDs must be short opaque tokens")
        if not _REF.match(self.artifact_ref) or ".." in self.artifact_ref:
            raise ValueError("Artifact references must be opaque store keys")
        if self.outcome not in OUTCOMES:
            raise ValueError("Unknown attempt outcome")
        if type(self.attempt_no) is not int or self.attempt_no < 1 or self.duration_ms < 0:
            raise ValueError("Invalid attempt counters")
        for key, value in self.action_params:
            if not _IDENT.match(key) or (isinstance(value, str) and not _IDENT.match(value)):
                raise ValueError("Audit action parameters must be opaque tokens")

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "region_id": self.region_id,
                "attempt_no": self.attempt_no,
                "page": self.page,
                "bbox": list(self.bbox),
                "config_fingerprint": self.config_fingerprint,
                "action": self.action_kind,
                "params": [list(p) for p in self.action_params],
                "failed_checks": list(self.failed_checks),
                "outcome": self.outcome,
                "duration_ms": self.duration_ms,
                "artifact_ref": self.artifact_ref,
                "cost": self.cost,
            },
            sort_keys=True,
        )

    def finished(self, **changes: object) -> AttemptRecord:
        return replace(self, **changes)  # type: ignore[arg-type]


class AttemptStore(Protocol):
    """Host-owned persistence. Every call is scoped by tenant."""

    def list(
        self, tenant_id: str, run_id: str, region_id: str | None = None
    ) -> list[AttemptRecord]:
        raise NotImplementedError

    def append(self, tenant_id: str, record: AttemptRecord) -> None:
        raise NotImplementedError

    def update(self, tenant_id: str, record: AttemptRecord) -> None:
        raise NotImplementedError


class InMemoryAttemptStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str, str, int], AttemptRecord] = {}

    @staticmethod
    def _tenant(tenant_id: str) -> str:
        if not _IDENT.match(tenant_id):
            raise ValueError("Tenant IDs must be short opaque tokens")
        return tenant_id

    def list(
        self, tenant_id: str, run_id: str, region_id: str | None = None
    ) -> list[AttemptRecord]:
        tenant = self._tenant(tenant_id)
        with self._lock:
            rows = [
                row
                for (t, run, region, _), row in self._rows.items()
                if t == tenant and run == run_id and (region_id is None or region == region_id)
            ]
        return sorted(rows, key=lambda r: (r.region_id, r.attempt_no))

    def append(self, tenant_id: str, record: AttemptRecord) -> None:
        key = (self._tenant(tenant_id), record.run_id, record.region_id, record.attempt_no)
        with self._lock:
            if key in self._rows:
                raise ValueError("Attempt already recorded")
            self._rows[key] = record

    def update(self, tenant_id: str, record: AttemptRecord) -> None:
        key = (self._tenant(tenant_id), record.run_id, record.region_id, record.attempt_no)
        with self._lock:
            if key not in self._rows:
                raise KeyError("Unknown attempt")
            self._rows[key] = record


class AuditReader:
    """Access control for audit reads: a principal sees only its granted tenants."""

    def __init__(self, store: AttemptStore, grants: Mapping[str, frozenset[str]]) -> None:
        self._store = store
        self._grants = dict(grants)

    def read(self, principal: str, tenant_id: str, run_id: str) -> list[AttemptRecord]:
        if tenant_id not in self._grants.get(principal, frozenset()):
            raise PermissionError("Not authorized for this tenant's audit records")
        return self._store.list(tenant_id, run_id)
