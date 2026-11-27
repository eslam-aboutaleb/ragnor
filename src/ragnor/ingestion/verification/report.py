"""Typed verification findings and the versioned ``VerificationReport``."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal

SEVERITIES = ("info", "warning", "error")
DISPOSITIONS = ("none", "repair_candidate", "needs_review")
STATUSES = ("verified", "failed", "needs_review")


@dataclass(frozen=True)
class Finding:
    """One observation. ``observed``/``expected`` are short summaries, never page text."""

    page: int
    region_id: str
    check_id: str
    severity: str
    observed: str
    expected: str
    source: str
    disposition: str = "none"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES or self.disposition not in DISPOSITIONS:
            raise ValueError("Unknown finding severity or disposition")
        if not self.check_id or not self.source or type(self.page) is not int or self.page < 1:
            raise ValueError("Findings need a page, check_id and source")
        if len(self.observed) > 200 or len(self.expected) > 200:
            raise ValueError("Finding summaries are bounded to 200 characters")


@dataclass(frozen=True)
class VerifierSettings:
    """Versioned verifier configuration; its fingerprint identifies a policy."""

    version: str = "1"
    mandatory: frozenset[str] = frozenset({"completeness"})
    match_iou: float = 0.3
    coverage_min: float = 0.6
    arithmetic_tolerance: Decimal = Decimal(0)
    fail_on_warning: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.match_iou <= 1 or not 0 < self.coverage_min <= 1:
            raise ValueError("match_iou and coverage_min must be in (0, 1]")
        if not self.arithmetic_tolerance.is_finite() or self.arithmetic_tolerance < 0:
            raise ValueError("Arithmetic tolerance must be a finite non-negative Decimal")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mandatory": sorted(self.mandatory),
            "match_iou": self.match_iou,
            "coverage_min": self.coverage_min,
            "arithmetic_tolerance": str(self.arithmetic_tolerance),
            "fail_on_warning": self.fail_on_warning,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    page: int
    status: str
    findings: tuple[Finding, ...]
    checks_run: tuple[str, ...]
    settings_version: str
    settings_fingerprint: str
    extraction_status: str = ""
    render_back_regions: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("Unknown verification status")

    @property
    def failing_region_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                f.region_id
                for f in self.findings
                if f.severity == "error" and f.disposition == "repair_candidate"
            )
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f"{f.check_id}:{f.severity}" for f in self.findings))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=sorted)
