"""Deterministic reconciliation of a proposed layout with a page inventory.

The result is a versioned :class:`LayoutPlan`. Plans only describe region
geometry and decisions. They carry no text, so applying one cannot change a
source value. Nothing is ever dropped: regions the inventory does not confirm
stay in the plan marked ``uncertain``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ragnor.ingestion.extraction_types import BBox, checked_bbox
from ragnor.ingestion.layout.inventory import InventoryRegion, InventoryTrace, PageInventory
from ragnor.ingestion.layout.regions import (
    LayoutRegion,
    ProposedLayout,
    area,
    coverage,
    expand_box,
    iou,
    stable_region_id,
)

PLAN_VERSION = 1
KEEP = "keep"
SPLIT = "split"
MERGE = "merge"
ADD = "add"
TAG_ARTIFACT = "tag_artifact"
UNCERTAIN = "uncertain"


class PlanInvariantError(ValueError):
    """A plan violated a reconciliation invariant."""


@dataclass(frozen=True)
class ReconcilePolicy:
    match_iou: float = 0.5
    contain_coverage: float = 0.8
    split_union_coverage: float = 0.7
    merge_gap: float = 0.08
    merge_x_overlap: float = 0.9
    missing_iou: float = 0.3
    crop_margin: float = 0.01
    max_uncertainty: float = 0.5


@dataclass(frozen=True)
class PlanRegion:
    region_id: str
    kind: str
    bbox: BBox
    crop_box: BBox
    action: str
    reason_codes: tuple[str, ...]
    source_region_ids: tuple[str, ...] = ()
    source_bboxes: tuple[BBox, ...] = ()
    inventory_region_ids: tuple[str, ...] = ()
    reading_order: int | None = None
    caption_target: str | None = None

    @property
    def changed(self) -> bool:
        """True when the region must be re-extracted (new or re-cut geometry)."""
        return self.action in {SPLIT, MERGE, ADD}


@dataclass(frozen=True)
class LayoutPlan:
    page: int
    page_fingerprint: str
    status: str  # unchanged | revised | needs_review | baseline
    regions: tuple[PlanRegion, ...]
    before_region_ids: tuple[str, ...]
    suspicion_codes: tuple[str, ...] = ()
    trace: InventoryTrace | None = None
    failure_code: str = ""
    version: int = PLAN_VERSION

    @property
    def changed_region_ids(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self.regions if r.changed)

    @property
    def needs_review(self) -> bool:
        return self.status in {"needs_review", "baseline"} or any(
            r.action == UNCERTAIN for r in self.regions
        )

    def validate_invariants(self) -> None:
        ids = [r.region_id for r in self.regions]
        if len(set(ids)) != len(ids):
            raise PlanInvariantError("duplicate region IDs")
        covered = set(ids)
        for region in self.regions:
            checked_bbox(region.bbox)
            checked_bbox(region.crop_box)
            cb, bb = region.crop_box, region.bbox
            if not (cb[0] <= bb[0] and cb[1] <= bb[1] and cb[2] >= bb[2] and cb[3] >= bb[3]):
                raise PlanInvariantError("crop box must contain region box")
            if not region.source_bboxes and not region.inventory_region_ids:
                raise PlanInvariantError("region has no provenance")
            if region.action in {SPLIT, MERGE, ADD} and not region.inventory_region_ids:
                raise PlanInvariantError("changed region lacks inventory decision")
            if region.action in {KEEP, TAG_ARTIFACT, UNCERTAIN} and (
                region.source_bboxes != (region.bbox,)
            ):
                raise PlanInvariantError("kept region geometry diverged from source")
            covered.update(region.source_region_ids)
        missing = set(self.before_region_ids) - covered
        if missing:
            raise PlanInvariantError("source region dropped without a decision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page": self.page,
            "page_fingerprint": self.page_fingerprint,
            "status": self.status,
            "failure_code": self.failure_code,
            "suspicion_codes": list(self.suspicion_codes),
            "trace": None
            if self.trace is None
            else {
                "provider": self.trace.provider,
                "model": self.trace.model,
                "request_id": self.trace.request_id,
            },
            "before_region_ids": list(self.before_region_ids),
            "regions": [
                {
                    "region_id": r.region_id,
                    "kind": r.kind,
                    "bbox": list(r.bbox),
                    "crop_box": list(r.crop_box),
                    "action": r.action,
                    "reason_codes": list(r.reason_codes),
                    "source_region_ids": list(r.source_region_ids),
                    "source_bboxes": [list(b) for b in r.source_bboxes],
                    "inventory_region_ids": list(r.inventory_region_ids),
                    "reading_order": r.reading_order,
                    "caption_target": r.caption_target,
                }
                for r in self.regions
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def baseline_plan(  # noqa: PLR0913
    layout: ProposedLayout,
    *,
    status: str,
    reason: str,
    suspicion_codes: tuple[str, ...] = (),
    failure_code: str = "",
    trace: InventoryTrace | None = None,
    crop_margin: float = 0.01,
) -> LayoutPlan:
    """Every proposed region kept as-is; used when no inventory decision exists."""
    regions = tuple(
        PlanRegion(
            r.region_id,
            r.kind,
            r.bbox,
            expand_box(r.bbox, crop_margin),
            KEEP,
            (reason,),
            (r.region_id,),
            (r.bbox,),
            reading_order=r.reading_order,
            caption_target=r.caption_target,
        )
        for r in _ordered(layout.regions)
    )
    return LayoutPlan(
        layout.page,
        layout.page_fingerprint,
        status,
        regions,
        tuple(sorted(r.region_id for r in layout.regions)),
        suspicion_codes,
        trace,
        failure_code,
    )


def _ordered(regions):  # type: ignore[no-untyped-def]
    return sorted(
        regions,
        key=lambda r: (
            r.reading_order is None,
            r.reading_order if r.reading_order is not None else 0,
            r.bbox[1],
            r.bbox[0],
            r.region_id,
        ),
    )


def _x_overlap(a: BBox, b: BBox) -> float:
    inter = min(a[2], b[2]) - max(a[0], b[0])
    narrow = min(a[2] - a[0], b[2] - b[0])
    return max(0.0, inter) / narrow if narrow > 0 else 0.0


class _Reconciler:
    def __init__(self, layout: ProposedLayout, inventory: PageInventory, policy: ReconcilePolicy):
        self.layout = layout
        self.inventory = inventory
        self.policy = policy
        self.proposed = {r.region_id: r for r in layout.regions}
        self.inv = sorted(inventory.regions, key=lambda r: (r.kind, r.bbox, r.region_id))
        self.inv_tables = [r for r in self.inv if r.kind == "table"]
        self.tables = sorted(
            (r for r in layout.regions if r.kind == "table"),
            key=lambda r: (r.bbox[1], r.bbox[0], r.region_id),
        )
        self.consumed_p: set[str] = set()
        self.consumed_i: set[str] = set()
        self.out: list[PlanRegion] = []
        self.inv_to_out: dict[str, str] = {}
        self.used_ids: set[str] = set()

    def unique(self, base: str) -> str:
        candidate, n = base, 1
        while candidate in self.used_ids:
            n += 1
            candidate = f"{base}.{n}"
        self.used_ids.add(candidate)
        return candidate

    def emit(self, **kw: Any) -> None:
        self.out.append(PlanRegion(crop_box=expand_box(kw["bbox"], self.policy.crop_margin), **kw))

    def run(self, suspicion_codes: tuple[str, ...]) -> LayoutPlan:
        self.split_tables()
        self.merge_tables()
        self.match_rest()
        self.add_missing()
        out = _link_captions(self.out, self.inventory.regions, self.inv_to_out)
        ordered = tuple(_ordered(out))
        meaningful = any(
            r.action in {SPLIT, MERGE, ADD, TAG_ARTIFACT}
            or "reading_order_updated" in r.reason_codes
            or "caption_linked" in r.reason_codes
            for r in ordered
        )
        uncertain = any(r.action == UNCERTAIN for r in ordered)
        status = "revised" if meaningful else ("needs_review" if uncertain else "unchanged")
        plan = LayoutPlan(
            self.layout.page,
            self.layout.page_fingerprint,
            status,
            ordered,
            tuple(sorted(self.proposed)),
            suspicion_codes,
            self.inventory.trace,
        )
        plan.validate_invariants()
        return plan

    def split_tables(self) -> None:
        """One proposed box that holds several inventory tables becomes several regions."""
        policy, page = self.policy, self.layout.page
        for table in self.tables:
            children = [
                c
                for c in self.inv_tables
                if c.region_id not in self.consumed_i
                and coverage(c.bbox, table.bbox) >= policy.contain_coverage
            ]
            if len(children) < 2:
                continue
            if any(
                iou(a.bbox, b.bbox) >= 0.1
                for i, a in enumerate(children)
                for b in children[i + 1 :]
            ):
                continue
            if sum(area(c.bbox) for c in children) < policy.split_union_coverage * area(table.bbox):
                continue
            self.consumed_p.add(table.region_id)
            for child in children:
                self.consumed_i.add(child.region_id)
                region_id = self.unique(
                    stable_region_id(page, "table", child.bbox, (table.region_id,))
                )
                self.inv_to_out[child.region_id] = region_id
                self.emit(
                    region_id=region_id,
                    kind="table",
                    bbox=child.bbox,
                    action=SPLIT,
                    reason_codes=("split_merged_table",),
                    source_region_ids=(table.region_id,),
                    source_bboxes=(table.bbox,),
                    inventory_region_ids=(child.region_id,),
                    reading_order=child.reading_order,
                )

    def merge_tables(self) -> None:
        """Pieces of one inventory table merge only when stacked and grid-compatible."""
        policy, page = self.policy, self.layout.page
        for big in self.inv_tables:
            if big.region_id in self.consumed_i:
                continue
            members = [
                t
                for t in self.tables
                if t.region_id not in self.consumed_p
                and coverage(t.bbox, big.bbox) >= policy.contain_coverage
            ]
            if len(members) < 2 or any(
                other.region_id != big.region_id
                and coverage(m.bbox, other.bbox) >= policy.contain_coverage
                for m in members
                for other in self.inv_tables
            ):
                continue
            ordered = sorted(members, key=lambda r: r.bbox[1])
            stacked = all(
                _x_overlap(a.bbox, b.bbox) >= policy.merge_x_overlap
                and -0.01 <= b.bbox[1] - a.bbox[3] <= policy.merge_gap
                for a, b in zip(ordered, ordered[1:], strict=False)
            )
            compatible = stacked and len({m.grid[1] for m in members if m.grid is not None}) <= 1
            self.consumed_i.add(big.region_id)
            self.consumed_p.update(m.region_id for m in members)
            if compatible:
                parents = tuple(m.region_id for m in ordered)
                region_id = self.unique(stable_region_id(page, "table", big.bbox, parents))
                self.inv_to_out[big.region_id] = region_id
                self.emit(
                    region_id=region_id,
                    kind="table",
                    bbox=big.bbox,
                    action=MERGE,
                    reason_codes=("merge_split_table",),
                    source_region_ids=parents,
                    source_bboxes=tuple(m.bbox for m in ordered),
                    inventory_region_ids=(big.region_id,),
                    reading_order=big.reading_order,
                )
                continue
            for member in ordered:
                self.used_ids.add(member.region_id)
                self.emit(
                    region_id=member.region_id,
                    kind=member.kind,
                    bbox=member.bbox,
                    action=UNCERTAIN,
                    reason_codes=("merge_incompatible",),
                    source_region_ids=(member.region_id,),
                    source_bboxes=(member.bbox,),
                    inventory_region_ids=(big.region_id,),
                    reading_order=member.reading_order,
                    caption_target=member.caption_target,
                )

    def match_rest(self) -> None:
        """Match one-to-one; an unconfirmed region is flagged, never deleted."""
        threshold = self.policy.match_iou
        for region in _ordered(self.layout.regions):
            if region.region_id in self.consumed_p:
                continue
            self.used_ids.add(region.region_id)
            same = _best(region, self.inv, self.consumed_i, threshold, same_kind=True)
            other = (
                None
                if same
                else _best(region, self.inv, self.consumed_i, threshold, same_kind=False)
            )
            common: dict[str, Any] = {
                "region_id": region.region_id,
                "bbox": region.bbox,
                "source_region_ids": (region.region_id,),
                "source_bboxes": (region.bbox,),
                "caption_target": region.caption_target,
            }
            if same is not None:
                self._matched(region, same, common)
                continue
            kind, action, reason = region.kind, UNCERTAIN, "unconfirmed_by_inventory"
            if other is not None:
                self.consumed_i.add(other.region_id)
                self.inv_to_out[other.region_id] = region.region_id
                if other.kind == "repeated_header":
                    kind, action, reason = "repeated_header", TAG_ARTIFACT, "repeated_header_tagged"
                else:
                    reason = "type_conflict"
            self.emit(
                kind=kind,
                action=action,
                reason_codes=(reason,),
                inventory_region_ids=() if other is None else (other.region_id,),
                reading_order=region.reading_order,
                **common,
            )

    def _matched(self, region: LayoutRegion, same: InventoryRegion, common: dict[str, Any]) -> None:
        self.consumed_i.add(same.region_id)
        self.inv_to_out[same.region_id] = region.region_id
        reasons = ["matched"]
        order = region.reading_order
        if same.reading_order is not None and same.reading_order != order:
            order = same.reading_order
            reasons.append("reading_order_updated")
        self.emit(
            kind=region.kind,
            action=KEEP,
            reason_codes=tuple(reasons),
            inventory_region_ids=(same.region_id,),
            reading_order=order,
            **common,
        )

    def add_missing(self) -> None:
        """Objects the inventory sees that the baseline missed."""
        for item in self.inv:
            if item.region_id in self.consumed_i:
                continue
            if any(
                iou(item.bbox, r.bbox) >= self.policy.missing_iou for r in self.proposed.values()
            ):
                continue
            region_id = self.unique(stable_region_id(self.layout.page, item.kind, item.bbox))
            self.inv_to_out[item.region_id] = region_id
            self.emit(
                region_id=region_id,
                kind=item.kind,
                bbox=item.bbox,
                action=ADD,
                reason_codes=("missing_object",),
                source_region_ids=(),
                source_bboxes=(),
                inventory_region_ids=(item.region_id,),
                reading_order=item.reading_order,
            )


def reconcile(
    layout: ProposedLayout,
    inventory: PageInventory,
    *,
    policy: ReconcilePolicy = ReconcilePolicy(),
    suspicion_codes: tuple[str, ...] = (),
) -> LayoutPlan:
    if inventory.uncertainty > policy.max_uncertainty:
        return baseline_plan(
            layout,
            status="needs_review",
            reason="inventory_uncertain",
            suspicion_codes=suspicion_codes,
            trace=inventory.trace,
            crop_margin=policy.crop_margin,
        )
    return _Reconciler(layout, inventory, policy).run(suspicion_codes)


def _best(
    region: LayoutRegion,
    inventory: list[InventoryRegion],
    consumed: set[str],
    threshold: float,
    *,
    same_kind: bool,
) -> InventoryRegion | None:
    best: tuple[float, InventoryRegion] | None = None
    for item in inventory:
        if item.region_id in consumed or (item.kind == region.kind) != same_kind:
            continue
        score = iou(region.bbox, item.bbox)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, item)
    return None if best is None else best[1]


def _link_captions(
    regions: list[PlanRegion],
    inventory: tuple[InventoryRegion, ...],
    inv_to_out: Mapping[str, str],
) -> list[PlanRegion]:
    targets = {r.region_id: r.caption_target for r in inventory if r.caption_target}
    linked: list[PlanRegion] = []
    for region in regions:
        target = None
        if region.kind == "caption" and region.caption_target is None:
            for inv_id in region.inventory_region_ids:
                if inv_id in targets and targets[inv_id] in inv_to_out:
                    target = inv_to_out[targets[inv_id]]
        if target is None:
            linked.append(region)
        else:
            linked.append(
                PlanRegion(
                    **{
                        **{f: getattr(region, f) for f in region.__dataclass_fields__},
                        "caption_target": target,
                        "reason_codes": (*region.reason_codes, "caption_linked"),
                    }
                )
            )
    return linked


RegionExtractor = Callable[[PlanRegion], object]


def apply_plan(plan: LayoutPlan, extract: RegionExtractor) -> dict[str, object]:
    """Re-run extraction only for regions whose geometry changed."""
    plan.validate_invariants()
    changed = set(plan.changed_region_ids)
    return {r.region_id: extract(r) for r in plan.regions if r.region_id in changed}


__all__ = [
    "ADD",
    "KEEP",
    "LayoutPlan",
    "MERGE",
    "PLAN_VERSION",
    "PlanInvariantError",
    "PlanRegion",
    "ReconcilePolicy",
    "SPLIT",
    "TAG_ARTIFACT",
    "UNCERTAIN",
    "apply_plan",
    "baseline_plan",
    "reconcile",
]
