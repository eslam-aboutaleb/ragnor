# ruff: noqa: PLR0913, PLR0917
"""Versioned document-family extraction recipes and a guarded fast path (EX-09).

A recipe records how a verified run extracted one repeated PDF layout (same
vendor invoice, lab-report template, ...) so later documents of that family can
skip full layout detection. The fast path is an optimisation only: the
verification step ALWAYS runs on a recipe hit, and any doubt (weak or ambiguous
family match, stale engine fingerprint, failed verification, provider error)
falls back once to the normal full flow. Nothing here has a third-party
dependency and nothing is stored globally: persistence is a host-injected
:class:`RecipeStore` (in-memory, JSON filesystem and stdlib SQLite reference
implementations are included).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ragnor.ingestion.extraction_types import (
    BBox,
    PageExtractionOutcome,
    PageVerifier,
    checked_bbox,
)

RECIPE_SCHEMA_VERSION = 1
_ID_MAX = 200


class RecipeError(RuntimeError):
    """Content-free recipe failure; ``code`` is a fixed identifier."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"Extraction recipe failure: {code}")


class ExtractionCancelled(RecipeError):
    """Raised when the host cancels processing; never retried or swallowed."""

    def __init__(self) -> None:
        super().__init__("cancelled")


class RecipeLearningRefused(RecipeError):
    """A recipe was not stored because the learning guards were not met."""


def _check_id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _ID_MAX or "\x00" in value:
        raise ValueError(f"{what} must be a non-empty string of at most {_ID_MAX} characters")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


class RecipeStatus(StrEnum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    DEACTIVATED = "deactivated"


@dataclass(frozen=True)
class RecipeAnchor:
    """Expected first-page text at (approximately) a normalized position."""

    text: str
    bbox: BBox
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        _check_id(self.text, "Anchor text")
        checked_bbox(self.bbox)
        if not (math.isfinite(self.tolerance) and 0 <= self.tolerance <= 0.5):
            raise ValueError("Anchor tolerance must be between 0 and 0.5")


@dataclass(frozen=True)
class FamilyCriteria:
    """Strong-match criteria: exact metadata, first-page anchors, page count, rotation."""

    family_id: str
    metadata: tuple[tuple[str, str], ...] = ()
    anchors: tuple[RecipeAnchor, ...] = ()
    page_count: int | None = None
    rotation: int = 0

    def __post_init__(self) -> None:
        _check_id(self.family_id, "family_id")
        if not self.metadata and not self.anchors:
            raise ValueError("A family needs metadata or anchor criteria")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270")
        if self.page_count is not None and (
            type(self.page_count) is not int or self.page_count < 1
        ):
            raise ValueError("page_count must be a positive integer")


@dataclass(frozen=True)
class RecipeRegion:
    """One saved extraction crop with its own provider choice."""

    name: str
    page: int
    bbox: BBox
    provider: str
    kind: str = "text"
    dpi: int = 200
    margin: float = 0.0

    def __post_init__(self) -> None:
        _check_id(self.name, "Region name")
        _check_id(self.provider, "Region provider")
        checked_bbox(self.bbox)
        if type(self.page) is not int or self.page < 1:
            raise ValueError("Region page must be a positive integer")
        if self.kind not in {"text", "table", "figure"}:
            raise ValueError("Region kind must be text, table or figure")
        if type(self.dpi) is not int or not 50 <= self.dpi <= 600:
            raise ValueError("Region dpi must be between 50 and 600")
        if not (math.isfinite(self.margin) and 0 <= self.margin < 0.25):
            raise ValueError("Region margin must be between 0 and 0.25")


@dataclass(frozen=True)
class QualityThresholds:
    min_region_coverage: float = 1.0
    min_text_chars: int = 1

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min_region_coverage) and 0 < self.min_region_coverage <= 1):
            raise ValueError("min_region_coverage must be in (0, 1]")
        if type(self.min_text_chars) is not int or self.min_text_chars < 0:
            raise ValueError("min_text_chars must be a non-negative integer")


@dataclass(frozen=True)
class RecipeValidity:
    """Everything that invalidates a recipe when it changes."""

    extractor: str
    extractor_version: str
    model: str
    layout_schema: str
    options_sha256: str
    rotation: int
    template_version: str

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if key != "rotation":
                _check_id(value, key)
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270")

    @classmethod
    def for_options(cls, options: Mapping[str, Any], **fields: Any) -> RecipeValidity:
        return cls(options_sha256=_digest(dict(options)), **fields)

    @property
    def fingerprint(self) -> str:
        return _digest(vars(self))


@dataclass(frozen=True)
class RecipeProvenance:
    created_by: str
    created_at: str
    source_document_sha256: str
    verifier_id: str
    verifier_version: str
    approval: str = "automatic"

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            _check_id(value, key)


@dataclass(frozen=True)
class ExtractionRecipe:
    """A typed, versioned, tenant-scoped extraction plan."""

    tenant: str
    criteria: FamilyCriteria
    regions: tuple[RecipeRegion, ...]
    validity: RecipeValidity
    provenance: RecipeProvenance
    expected_counts: tuple[tuple[str, int, int], ...] = ()
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    serializer: str = "plain-text"
    artifact_policy: str = "none"
    revision: int = 1
    status: RecipeStatus = RecipeStatus.ACTIVE
    status_reason: str = ""
    schema_version: int = RECIPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_id(self.tenant, "tenant")
        _check_id(self.serializer, "serializer")
        if self.artifact_policy not in {"none", "crops", "crops-encrypted"}:
            raise ValueError("Unknown artifact policy")
        if self.schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError("Unsupported recipe schema version")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if not self.regions:
            raise ValueError("A recipe needs at least one region")
        names = [region.name for region in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("Region names must be unique")
        known = set(names)
        for name, low, high in self.expected_counts:
            if name not in known or not (type(low) is int and type(high) is int):
                raise ValueError("Expected counts must reference recipe regions")
            if not 0 <= low <= high:
                raise ValueError("Expected count bounds are invalid")
        if self.validity.rotation != self.criteria.rotation:
            raise ValueError("Recipe rotation must match its family criteria")

    @property
    def family_id(self) -> str:
        return self.criteria.family_id

    @property
    def active(self) -> bool:
        return self.status is RecipeStatus.ACTIVE

    @property
    def extraction_fingerprint(self) -> str:
        """Changes whenever the engine, options, template or recipe plan change."""
        return _digest(
            {
                "validity": vars(self.validity),
                "plan": self.to_dict()["regions"],
                "serializer": self.serializer,
                "thresholds": vars(self.thresholds),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tenant": self.tenant,
            "revision": self.revision,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "serializer": self.serializer,
            "artifact_policy": self.artifact_policy,
            "criteria": {
                "family_id": self.criteria.family_id,
                "metadata": [list(item) for item in self.criteria.metadata],
                "anchors": [
                    {"text": a.text, "bbox": list(a.bbox), "tolerance": a.tolerance}
                    for a in self.criteria.anchors
                ],
                "page_count": self.criteria.page_count,
                "rotation": self.criteria.rotation,
            },
            "regions": [{**vars(region), "bbox": list(region.bbox)} for region in self.regions],
            "validity": vars(self.validity).copy(),
            "provenance": vars(self.provenance).copy(),
            "expected_counts": [list(item) for item in self.expected_counts],
            "thresholds": vars(self.thresholds).copy(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExtractionRecipe:
        """Strictly rebuild a recipe; any malformed field raises ``RecipeError``."""
        try:
            crit = data["criteria"]
            return cls(
                tenant=data["tenant"],
                revision=data["revision"],
                status=RecipeStatus(data["status"]),
                status_reason=data.get("status_reason", ""),
                serializer=data["serializer"],
                artifact_policy=data["artifact_policy"],
                schema_version=data["schema_version"],
                criteria=FamilyCriteria(
                    family_id=crit["family_id"],
                    metadata=tuple((str(k), str(v)) for k, v in crit["metadata"]),
                    anchors=tuple(
                        RecipeAnchor(a["text"], tuple(a["bbox"]), a["tolerance"])
                        for a in crit["anchors"]
                    ),
                    page_count=crit["page_count"],
                    rotation=crit["rotation"],
                ),
                regions=tuple(
                    RecipeRegion(**{**r, "bbox": tuple(r["bbox"])}) for r in data["regions"]
                ),
                validity=RecipeValidity(**data["validity"]),
                provenance=RecipeProvenance(**data["provenance"]),
                expected_counts=tuple(
                    (str(n), int(lo), int(hi)) for n, lo, hi in data["expected_counts"]
                ),
                thresholds=QualityThresholds(**data["thresholds"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RecipeError("corrupt_recipe") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> ExtractionRecipe:
        try:
            data = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise RecipeError("corrupt_recipe") from exc
        if not isinstance(data, dict):
            raise RecipeError("corrupt_recipe")
        return cls.from_dict(data)

    @property
    def content_sha256(self) -> str:
        return _digest(self.to_dict())


# --------------------------------------------------------------------------- stores


class RecipeStore(Protocol):
    """Host-owned persistence. All reads and writes are tenant scoped.

    Revisions are immutable and append-only, which makes every change auditable;
    ``put_if_absent`` and ``compare_and_set`` are atomic, so concurrent writers
    cannot both succeed for the same revision.
    """

    def get(self, tenant: str, family_id: str) -> ExtractionRecipe | None:
        """Latest revision for the family, whatever its status."""
        raise NotImplementedError

    def put_if_absent(self, recipe: ExtractionRecipe) -> bool:
        """Store revision 1; False when the family already has any revision."""
        raise NotImplementedError

    def compare_and_set(self, expected_revision: int, recipe: ExtractionRecipe) -> bool:
        """Append ``recipe`` iff the latest revision is ``expected_revision``."""
        raise NotImplementedError

    def deactivate(
        self,
        tenant: str,
        family_id: str,
        expected_revision: int,
        reason: str,
        *,
        quarantine: bool = False,
    ) -> bool:
        """Append a non-active revision (CAS on ``expected_revision``)."""
        raise NotImplementedError

    def list_revisions(self, tenant: str, family_id: str) -> list[ExtractionRecipe]:
        raise NotImplementedError


def _next_revision(recipe: ExtractionRecipe, status: RecipeStatus, reason: str) -> ExtractionRecipe:
    return replace(recipe, revision=recipe.revision + 1, status=status, status_reason=reason[:200])


class _StoreBase:
    def deactivate(
        self,
        tenant: str,
        family_id: str,
        expected_revision: int,
        reason: str,
        *,
        quarantine: bool = False,
    ) -> bool:
        current = self.get(tenant, family_id)  # type: ignore[attr-defined]
        if current is None or current.revision != expected_revision:
            return False
        status = RecipeStatus.QUARANTINED if quarantine else RecipeStatus.DEACTIVATED
        return self.compare_and_set(  # type: ignore[attr-defined]
            expected_revision, _next_revision(current, status, reason)
        )


class InMemoryRecipeStore(_StoreBase):
    """Per-instance store; instances never share state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], list[ExtractionRecipe]] = {}

    def get(self, tenant: str, family_id: str) -> ExtractionRecipe | None:
        with self._lock:
            history = self._data.get((tenant, family_id))
            return history[-1] if history else None

    def put_if_absent(self, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != 1:
            raise ValueError("put_if_absent stores revision 1")
        key = (recipe.tenant, recipe.family_id)
        with self._lock:
            if key in self._data:
                return False
            self._data[key] = [recipe]
            return True

    def compare_and_set(self, expected_revision: int, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != expected_revision + 1:
            raise ValueError("Recipe revision must follow the expected revision")
        with self._lock:
            history = self._data.get((recipe.tenant, recipe.family_id))
            if not history or history[-1].revision != expected_revision:
                return False
            history.append(recipe)
            return True

    def list_revisions(self, tenant: str, family_id: str) -> list[ExtractionRecipe]:
        with self._lock:
            return list(self._data.get((tenant, family_id), ()))


def _slug(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


class JsonFileRecipeStore(_StoreBase):
    """Filesystem store: ``root/<tenant>/<family>/rev-NNNNNNNN.json``.

    Revision files are created with a hard-link publish, which fails when the
    target exists, so concurrent writers (threads or processes) get an atomic CAS
    without locks. Tenant and family names are hashed for the path, so hostile
    ids cannot traverse directories. Corrupt files raise ``corrupt_recipe``.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _dir(self, tenant: str, family_id: str) -> Path:
        return (
            self._root / _slug(_check_id(tenant, "tenant")) / _slug(_check_id(family_id, "family"))
        )

    @staticmethod
    def _revisions(directory: Path) -> list[int]:
        if not directory.is_dir():
            return []
        found = []
        for item in directory.glob("rev-*.json"):
            try:
                found.append(int(item.stem.removeprefix("rev-")))
            except ValueError:
                continue
        return sorted(found)

    def _read(self, directory: Path, revision: int) -> ExtractionRecipe:
        path = directory / f"rev-{revision:08d}.json"
        try:
            recipe = ExtractionRecipe.from_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise RecipeError("corrupt_recipe") from exc
        if recipe.revision != revision:
            raise RecipeError("corrupt_recipe")
        return recipe

    def get(self, tenant: str, family_id: str) -> ExtractionRecipe | None:
        directory = self._dir(tenant, family_id)
        revisions = self._revisions(directory)
        if not revisions:
            return None
        recipe = self._read(directory, revisions[-1])
        if recipe.tenant != tenant or recipe.family_id != family_id:
            raise RecipeError("corrupt_recipe")
        return recipe

    def _publish(self, recipe: ExtractionRecipe, expected_latest: int) -> bool:
        directory = self._dir(recipe.tenant, recipe.family_id)
        directory.mkdir(parents=True, exist_ok=True)
        revisions = self._revisions(directory)
        if (revisions[-1] if revisions else 0) != expected_latest:
            return False
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(recipe.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, directory / f"rev-{recipe.revision:08d}.json")
            except FileExistsError:
                return False
            return True
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    def put_if_absent(self, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != 1:
            raise ValueError("put_if_absent stores revision 1")
        return self._publish(recipe, 0)

    def compare_and_set(self, expected_revision: int, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != expected_revision + 1:
            raise ValueError("Recipe revision must follow the expected revision")
        return self._publish(recipe, expected_revision)

    def list_revisions(self, tenant: str, family_id: str) -> list[ExtractionRecipe]:
        directory = self._dir(tenant, family_id)
        return [self._read(directory, n) for n in self._revisions(directory)]


class SQLiteRecipeStore(_StoreBase):
    """Reference SQL store on stdlib ``sqlite3``; hosts port the same schema.

    ``(tenant, family_id, revision)`` is the primary key, so a losing concurrent
    writer hits an integrity error and gets ``False`` instead of overwriting.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS ragnor_extraction_recipes ("
        " tenant TEXT NOT NULL, family_id TEXT NOT NULL, revision INTEGER NOT NULL,"
        " status TEXT NOT NULL, body TEXT NOT NULL,"
        " PRIMARY KEY (tenant, family_id, revision))"
    )

    _BASE = "SELECT body FROM ragnor_extraction_recipes WHERE tenant=? AND family_id=?"
    _LATEST = _BASE + " ORDER BY revision DESC LIMIT 1"
    _ALL = _BASE + " ORDER BY revision ASC"

    def __init__(self, database: str | os.PathLike[str] = ":memory:") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(database), check_same_thread=False, isolation_level=None)
        self._conn.execute(self._SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _rows(self, tenant: str, family_id: str, latest_only: bool) -> list[ExtractionRecipe]:
        sql = self._LATEST if latest_only else self._ALL
        with self._lock:
            rows = self._conn.execute(sql, (tenant, family_id)).fetchall()
        return [ExtractionRecipe.from_json(row[0]) for row in rows]

    def get(self, tenant: str, family_id: str) -> ExtractionRecipe | None:
        rows = self._rows(tenant, family_id, True)
        return rows[0] if rows else None

    def _insert(self, recipe: ExtractionRecipe) -> bool:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO ragnor_extraction_recipes VALUES (?,?,?,?,?)",
                    (
                        recipe.tenant,
                        recipe.family_id,
                        recipe.revision,
                        recipe.status.value,
                        recipe.to_json(),
                    ),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def put_if_absent(self, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != 1:
            raise ValueError("put_if_absent stores revision 1")
        return self._insert(recipe)

    def compare_and_set(self, expected_revision: int, recipe: ExtractionRecipe) -> bool:
        if recipe.revision != expected_revision + 1:
            raise ValueError("Recipe revision must follow the expected revision")
        current = self.get(recipe.tenant, recipe.family_id)
        if current is None or current.revision != expected_revision:
            return False
        return self._insert(recipe)

    def list_revisions(self, tenant: str, family_id: str) -> list[ExtractionRecipe]:
        return self._rows(tenant, family_id, False)


# ------------------------------------------------------------------- classification


@dataclass(frozen=True)
class DocumentFingerprint:
    """What the classifier may see: metadata and first-page text anchors."""

    metadata: Mapping[str, str] = field(default_factory=dict)
    first_page_words: tuple[tuple[str, BBox], ...] = ()
    page_count: int | None = None
    rotation: int = 0


@dataclass(frozen=True)
class FamilyDecision:
    family_id: str | None
    score: float
    reason: str


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _anchor_hit(anchor: RecipeAnchor, words: Sequence[tuple[str, BBox]]) -> bool:
    want = _norm(anchor.text)
    for text, box in words:
        if _norm(text) != want:
            continue
        if all(abs(a - b) <= anchor.tolerance for a, b in zip(box, anchor.bbox, strict=True)):
            return True
    return False


class FamilyClassifier:
    """Strong-match classifier; ambiguity, weak matches and drift return no family."""

    def __init__(
        self,
        families: Iterable[FamilyCriteria],
        *,
        min_score: float = 0.99,
        min_margin: float = 0.05,
    ) -> None:
        self._families = tuple(families)
        if not (0 < min_score <= 1) or not (0 <= min_margin < 1):
            raise ValueError("Classifier thresholds are out of range")
        ids = [f.family_id for f in self._families]
        if len(set(ids)) != len(ids):
            raise ValueError("Family ids must be unique")
        self.min_score = min_score
        self.min_margin = min_margin

    def score(self, criteria: FamilyCriteria, doc: DocumentFingerprint) -> float:
        if criteria.rotation != doc.rotation:
            return 0.0
        if criteria.page_count is not None and doc.page_count != criteria.page_count:
            return 0.0
        for key, value in criteria.metadata:
            if doc.metadata.get(key) != value:
                return 0.0
        if not criteria.anchors:
            return 1.0
        hits = sum(_anchor_hit(a, doc.first_page_words) for a in criteria.anchors)
        return hits / len(criteria.anchors)

    def classify(self, doc: DocumentFingerprint) -> FamilyDecision:
        scored = sorted(((self.score(f, doc), f.family_id) for f in self._families), reverse=True)
        if not scored or scored[0][0] < self.min_score:
            return FamilyDecision(None, scored[0][0] if scored else 0.0, "no_strong_match")
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.min_margin:
            return FamilyDecision(None, scored[0][0], "ambiguous_match")
        return FamilyDecision(scored[0][1], scored[0][0], "matched")


# ---------------------------------------------------------------------- fast path


@dataclass(frozen=True)
class ExtractedRegion:
    name: str
    text: str = field(repr=False)
    object_count: int = 0
    shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    reasons: tuple[str, ...] = ()


class RegionExtractor(Protocol):
    """Applies one saved region (crop + provider) to a document."""

    def extract_region(self, document: Any, region: RecipeRegion) -> ExtractedRegion:
        raise NotImplementedError


class RecipeVerifier(Protocol):
    """Count, coverage and shape verification; mandatory on every recipe hit."""

    id: str
    version: str

    def verify(
        self, recipe: ExtractionRecipe, regions: Sequence[ExtractedRegion]
    ) -> VerificationReport:
        raise NotImplementedError


class CountCoverageVerifier:
    """Default verifier: every region present, counts in range, text and shape sane."""

    id = "recipe-count-coverage"
    version = "1"

    def __init__(self, expected_shapes: Mapping[str, tuple[int, int]] | None = None) -> None:
        self._shapes = dict(expected_shapes or {})

    def verify(
        self, recipe: ExtractionRecipe, regions: Sequence[ExtractedRegion]
    ) -> VerificationReport:
        reasons: list[str] = []
        got = {r.name: r for r in regions}
        if len(got) != len(regions):
            reasons.append("duplicate_region")
        wanted = {r.name for r in recipe.regions}
        extra = set(got) - wanted
        if extra:
            reasons.append("unexpected_region")
        present = [n for n in wanted if n in got]
        coverage = len(present) / len(wanted)
        if coverage < recipe.thresholds.min_region_coverage:
            reasons.append("region_coverage")
        for name, low, high in recipe.expected_counts:
            region = got.get(name)
            if region is not None and not low <= region.object_count <= high:
                reasons.append(f"count_out_of_range:{name}")
        for name, shape in self._shapes.items():
            region = got.get(name)
            if region is not None and region.shape != shape:
                reasons.append(f"shape_mismatch:{name}")
        for region in got.values():
            if len(region.text.strip()) < recipe.thresholds.min_text_chars:
                reasons.append(f"empty_region:{region.name}")
        return VerificationReport(not reasons, tuple(sorted(set(reasons))))


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class RecipeResult:
    source: str  # "recipe" or "full"
    regions: tuple[ExtractedRegion, ...]
    family_id: str | None
    recipe_revision: int | None
    extraction_fingerprint: str | None
    fallback_reason: str = ""
    provider_calls: int = 0


class RecipeTelemetry:
    """Counters per tenant; contains no document content."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, Counter[str]] = {}

    def record(self, tenant: str, event: str, amount: int = 1) -> None:
        with self._lock:
            self._counts.setdefault(tenant, Counter())[event] += amount

    def snapshot(self, tenant: str | None = None) -> dict[str, int]:
        with self._lock:
            if tenant is not None:
                return dict(self._counts.get(tenant, {}))
            total: Counter[str] = Counter()
            for counter in self._counts.values():
                total.update(counter)
            return dict(total)

    def rates(self, tenant: str | None = None) -> dict[str, float]:
        snap = self.snapshot(tenant)
        hits, false_hits = snap.get("recipe_hit", 0), snap.get("false_hit", 0)
        attempts = hits + false_hits
        docs = snap.get("documents", 0)
        return {
            "match_precision": hits / attempts if attempts else 1.0,
            "false_hit_rate": false_hits / attempts if attempts else 0.0,
            "fallback_rate": snap.get("fallback", 0) / docs if docs else 0.0,
            "cache_hit_rate": hits / docs if docs else 0.0,
        }


class RecipeFastPath:
    """classify -> look up -> apply -> VERIFY -> publish; otherwise one full run.

    The verifier is a required constructor argument: there is no code path that
    publishes recipe output without it. After a failed check the recipe is
    quarantined and the document goes through ``full_extractor`` exactly once.
    """

    def __init__(
        self,
        *,
        classifier: FamilyClassifier,
        store: RecipeStore,
        region_extractor: RegionExtractor,
        verifier: RecipeVerifier,
        full_extractor: Callable[[Any], Sequence[ExtractedRegion]],
        engine_validity: RecipeValidity,
        telemetry: RecipeTelemetry | None = None,
        full_cost_calls: int = 1,
        quarantine_after: int = 1,
    ) -> None:
        if verifier is None or not callable(getattr(verifier, "verify", None)):
            raise ValueError("A recipe verifier is required")
        self._classifier = classifier
        self._store = store
        self._regions = region_extractor
        self._verifier = verifier
        self._full = full_extractor
        self._engine = engine_validity
        self.telemetry = telemetry or RecipeTelemetry()
        self._full_cost = full_cost_calls
        if type(quarantine_after) is not int or quarantine_after < 1:
            raise ValueError("quarantine_after must be a positive integer")
        self._quarantine_after = quarantine_after
        self._failures: dict[tuple[str, str], int] = {}

    def _full_run(
        self,
        tenant: str,
        document: Any,
        family: str | None,
        reason: str,
        token: CancellationToken | None,
    ) -> RecipeResult:
        self._check(token)
        regions = tuple(self._full(document))
        self.telemetry.record(tenant, "fallback" if reason != "no_family" else "full_no_family")
        self.telemetry.record(tenant, "full_runs")
        return RecipeResult("full", regions, family, None, None, reason)

    @staticmethod
    def _check(token: CancellationToken | None) -> None:
        if token is not None and token.cancelled:
            raise ExtractionCancelled

    def process(
        self,
        tenant: str,
        document: Any,
        fingerprint: DocumentFingerprint,
        *,
        cancel: CancellationToken | None = None,
    ) -> RecipeResult:
        _check_id(tenant, "tenant")
        self._check(cancel)
        self.telemetry.record(tenant, "documents")
        decision = self._classifier.classify(fingerprint)
        if decision.family_id is None:
            reason = "no_family" if decision.reason == "no_strong_match" else decision.reason
            return self._full_run(tenant, document, None, reason, cancel)
        family = decision.family_id
        recipe = self._store.get(tenant, family)
        if recipe is None or recipe.tenant != tenant or recipe.family_id != family:
            return self._full_run(tenant, document, family, "no_recipe", cancel)
        if not recipe.active:
            return self._full_run(tenant, document, family, f"recipe_{recipe.status.value}", cancel)
        if recipe.validity != self._engine:
            self._store.deactivate(tenant, family, recipe.revision, "stale_validity")
            self.telemetry.record(tenant, "stale_invalidation")
            return self._full_run(tenant, document, family, "stale_validity", cancel)
        return self._apply(tenant, document, recipe, cancel)

    def _apply(
        self, tenant: str, document: Any, recipe: ExtractionRecipe, cancel: CancellationToken | None
    ) -> RecipeResult:
        extracted: list[ExtractedRegion] = []
        calls = 0
        try:
            for region in recipe.regions:
                self._check(cancel)
                extracted.append(self._regions.extract_region(document, region))
                calls += 1
            report = self._verifier.verify(recipe, tuple(extracted))
        except ExtractionCancelled:
            raise
        except Exception:  # noqa: BLE001 - provider details must not leak; fail closed
            report = VerificationReport(False, ("recipe_apply_failed",))
        key = (tenant, recipe.family_id)
        if not report.ok:
            self._failures[key] = self._failures.get(key, 0) + 1
            if self._failures[key] >= self._quarantine_after:
                self._store.deactivate(
                    tenant,
                    recipe.family_id,
                    recipe.revision,
                    "verification_failed",
                    quarantine=True,
                )
                self._failures.pop(key, None)
            self.telemetry.record(tenant, "false_hit")
            self.telemetry.record(tenant, "wasted_region_calls", calls)
            return self._full_run(tenant, document, recipe.family_id, "verification_failed", cancel)
        self._failures.pop(key, None)
        self.telemetry.record(tenant, "recipe_hit")
        self.telemetry.record(tenant, "region_calls", calls)
        self.telemetry.record(tenant, "calls_saved", max(self._full_cost - calls, 0))
        return RecipeResult(
            "recipe",
            tuple(extracted),
            recipe.family_id,
            recipe.revision,
            recipe.extraction_fingerprint,
            provider_calls=calls,
        )


# ---------------------------------------------------------------------- learning


class RecipeLearner:
    """Stores a recipe only from a run that already passed the quality gate.

    Never learns from unverified VLM guesses: every page of the source run must
    be complete and accepted by the host's :class:`PageVerifier`, a sample must be
    confirmed by the host, and in ``approval_required`` mode an explicit host
    approval callback must return True. Failures raise ``RecipeLearningRefused``
    with a fixed code and store nothing.
    """

    def __init__(
        self,
        store: RecipeStore,
        page_verifier: PageVerifier,
        *,
        sample_confirmer: Callable[[ExtractionRecipe], bool],
        approval_required: bool = True,
        approver: Callable[[ExtractionRecipe], bool] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if approval_required and approver is None:
            raise ValueError("approval_required needs an approver callback")
        self._store = store
        self._verifier = page_verifier
        self._confirm = sample_confirmer
        self._approval_required = approval_required
        self._approver = approver
        self._clock = clock

    def learn(
        self, draft: ExtractionRecipe, verified_pages: Sequence[PageExtractionOutcome]
    ) -> ExtractionRecipe:
        if not verified_pages:
            raise RecipeLearningRefused("no_verified_run")
        for page in verified_pages:
            if not page.complete or not self._verifier.verify(page):
                raise RecipeLearningRefused("unverified_run")
        if not self._confirm(draft):
            raise RecipeLearningRefused("sample_not_confirmed")
        approval = "automatic"
        if self._approval_required:
            assert self._approver is not None  # noqa: S101 - guarded in __init__
            if not self._approver(draft):
                raise RecipeLearningRefused("not_approved")
            approval = "host-approved"
        provenance = replace(
            draft.provenance,
            verifier_id=self._verifier.id,
            verifier_version=self._verifier.version,
            approval=approval,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock())),
        )
        current = self._store.get(draft.tenant, draft.family_id)
        if current is None:
            recipe = replace(
                draft,
                provenance=provenance,
                revision=1,
                status=RecipeStatus.ACTIVE,
                status_reason="",
            )
            stored = self._store.put_if_absent(recipe)
        else:
            recipe = replace(
                draft,
                provenance=provenance,
                revision=current.revision + 1,
                status=RecipeStatus.ACTIVE,
                status_reason="",
            )
            stored = self._store.compare_and_set(current.revision, recipe)
        if not stored:
            raise RecipeLearningRefused("concurrent_update")
        return recipe


__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "CancellationToken",
    "CountCoverageVerifier",
    "DocumentFingerprint",
    "ExtractedRegion",
    "ExtractionCancelled",
    "ExtractionRecipe",
    "FamilyClassifier",
    "FamilyCriteria",
    "FamilyDecision",
    "InMemoryRecipeStore",
    "JsonFileRecipeStore",
    "QualityThresholds",
    "RecipeAnchor",
    "RecipeError",
    "RecipeFastPath",
    "RecipeLearner",
    "RecipeLearningRefused",
    "RecipeProvenance",
    "RecipeRegion",
    "RecipeResult",
    "RecipeStatus",
    "RecipeStore",
    "RecipeTelemetry",
    "RecipeValidity",
    "RecipeVerifier",
    "RegionExtractor",
    "SQLiteRecipeStore",
    "VerificationReport",
]
