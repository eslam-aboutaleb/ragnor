"""Deterministic suspicion signals that decide whether a page needs an inventory."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ragnor.ingestion.layout.regions import LayoutRegion, ProposedLayout, area, iou


@dataclass(frozen=True)
class SuspicionPolicy:
    min_confidence: float = 0.6
    table_overlap_iou: float = 0.1
    adjacent_gap: float = 0.015
    adjacent_x_overlap: float = 0.8
    caption_distance: float = 0.08
    unclaimed_area: float = 0.05
    # Absent confidence is never "high confidence". It is reported, but it only
    # triggers an inventory when the host opts in.
    trigger_on_unreported_confidence: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.min_confidence,
            self.table_overlap_iou,
            self.adjacent_gap,
            self.adjacent_x_overlap,
            self.caption_distance,
            self.unclaimed_area,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Suspicion thresholds must be between zero and one")


@dataclass(frozen=True)
class Signal:
    code: str
    region_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuspicionReport:
    signals: tuple[Signal, ...]
    confidence_reported: bool

    @property
    def suspicious(self) -> bool:
        return bool(self.signals)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.code for s in self.signals))


def _x_overlap(a: LayoutRegion, b: LayoutRegion) -> float:
    inter = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
    narrow = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
    return max(0.0, inter) / narrow if narrow > 0 else 0.0


def assess(layout: ProposedLayout, policy: SuspicionPolicy = SuspicionPolicy()) -> SuspicionReport:
    signals: list[Signal] = []
    regions = sorted(layout.regions, key=lambda r: (r.bbox[1], r.bbox[0], r.region_id))
    low = [
        r.region_id
        for r in regions
        if r.confidence is not None and r.confidence < policy.min_confidence
    ]
    if low:
        signals.append(Signal("low_confidence", tuple(low)))
    unreported = [r.region_id for r in regions if r.confidence is None]
    if unreported and policy.trigger_on_unreported_confidence:
        signals.append(Signal("confidence_unreported", tuple(unreported)))

    tables = [r for r in regions if r.kind == "table"]
    for i, first in enumerate(tables):
        for second in tables[i + 1 :]:
            pair = tuple(sorted((first.region_id, second.region_id)))
            if iou(first.bbox, second.bbox) > policy.table_overlap_iou:
                signals.append(Signal("overlapping_tables", pair))
                continue
            top, bottom = sorted((first, second), key=lambda r: r.bbox[1])
            gap = bottom.bbox[1] - top.bbox[3]
            if (
                0 <= gap <= policy.adjacent_gap
                and _x_overlap(first, second) >= policy.adjacent_x_overlap
            ):
                signals.append(Signal("adjacent_tables", pair))

    targets = [r for r in regions if r.kind in {"table", "figure"}]
    for caption in (r for r in regions if r.kind == "caption"):
        near = any(
            min(abs(caption.bbox[1] - t.bbox[3]), abs(t.bbox[1] - caption.bbox[3]))
            <= policy.caption_distance
            and _x_overlap(caption, t) > 0
            for t in targets
        )
        if caption.caption_target is None and not near:
            signals.append(Signal("orphan_caption", (caption.region_id,)))

    seen: dict[int, str] = {}
    for region in regions:
        if region.reading_order is None:
            continue
        if region.reading_order in seen:
            signals.append(
                Signal("reading_order_conflict", (seen[region.reading_order], region.region_id))
            )
        else:
            seen[region.reading_order] = region.region_id

    if any(area(box) >= policy.unclaimed_area for box in layout.unclaimed_ink):
        signals.append(Signal("unclaimed_ink"))
    if layout.expected_counts is not None and layout.expected_counts != layout.counts():
        signals.append(Signal("count_mismatch"))
    if layout.parser_errors:
        signals.append(Signal("parser_error"))
    return SuspicionReport(tuple(signals), confidence_reported=not unreported)
