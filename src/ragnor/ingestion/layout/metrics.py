"""Region-layout disagreement metrics, reported alongside the EX-02 benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ragnor.ingestion.extraction_types import BBox
from ragnor.ingestion.layout.reconcile import (
    ADD,
    MERGE,
    SPLIT,
    TAG_ARTIFACT,
    UNCERTAIN,
    LayoutPlan,
)
from ragnor.ingestion.layout.regions import iou


def layout_disagreement(plans: Iterable[LayoutPlan]) -> dict[str, float | int]:
    """Aggregate how often the inventory disagreed with the baseline layout."""
    totals: Counter[str] = Counter()
    for plan in plans:
        totals["pages"] += 1
        totals["pages_disagreed"] += any(
            r.action in {SPLIT, MERGE, ADD, TAG_ARTIFACT, UNCERTAIN} for r in plan.regions
        )
        totals["baseline_regions"] += len(plan.before_region_ids)
        totals["split"] += sum(r.action == SPLIT for r in plan.regions)
        totals["merge"] += sum(r.action == MERGE for r in plan.regions)
        totals["added"] += sum(r.action == ADD for r in plan.regions)
        totals["tagged_artifacts"] += sum(r.action == TAG_ARTIFACT for r in plan.regions)
        totals["uncertain"] += sum(r.action == UNCERTAIN for r in plan.regions)
        totals["fallbacks"] += bool(plan.failure_code)
    pages = totals["pages"]
    return {
        **{
            k: totals[k]
            for k in (
                "pages",
                "pages_disagreed",
                "baseline_regions",
                "split",
                "merge",
                "added",
                "tagged_artifacts",
                "uncertain",
                "fallbacks",
            )
        },
        "page_disagreement_rate": totals["pages_disagreed"] / pages if pages else 0.0,
    }


def segmentation_f1(
    truth: Iterable[BBox], predicted: Iterable[BBox], *, min_iou: float = 0.7
) -> float:
    """Greedy one-to-one box F1 used to compare baseline and reconciled layouts."""
    truth_left, scored = list(truth), 0
    predicted = list(predicted)
    for box in predicted:
        best = max(truth_left, key=lambda t: iou(t, box), default=None)
        if best is not None and iou(best, box) >= min_iou:
            truth_left.remove(best)
            scored += 1
    total_truth = scored + len(truth_left)
    if not predicted and not total_truth:
        return 1.0
    precision = scored / len(predicted) if predicted else 0.0
    recall = scored / total_truth if total_truth else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
