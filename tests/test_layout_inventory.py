"""EX-07: page inventory schema, suspicion, reconciliation and cost policy."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import replace

import pytest

from ragnor.ingestion import PageExtractionOutcome, PageExtractionStatus
from ragnor.ingestion.extraction_types import ExtractionCell, ExtractionRegion, ExtractionTable
from ragnor.ingestion.layout import (
    InventoryBudget,
    InventoryError,
    InventoryLimits,
    LayoutInventoryPlanner,
    LayoutInventoryPolicy,
    LayoutRegion,
    PageImage,
    PageInventory,
    PlanInvariantError,
    ProposedLayout,
    ReconcilePolicy,
    SuspicionPolicy,
    apply_plan,
    assess,
    baseline_plan,
    proposed_from_outcome,
    reconcile,
    stable_region_id,
)
from ragnor.ingestion.layout.metrics import layout_disagreement, segmentation_f1
from ragnor.ingestion.layout.regions import (
    area,
    expand_box,
    from_canonical,
    iou,
    to_canonical,
    union_box,
)

FP = "page-fp-1"


def region(rid, kind, box, **kw):
    return LayoutRegion(rid, kind, box, **kw)


def layout(*regions, **kw):
    return ProposedLayout(1, tuple(regions), page_fingerprint=FP, **kw)


def inv_json(*items, uncertainty=0.1, fp=FP, width=1000, height=1400, **extra):
    regions = []
    for item in items:
        rid, kind, box, *rest = item
        entry = {"id": rid, "type": kind, "bbox": list(box)}
        if rest and rest[0] is not None:
            entry["reading_order"] = rest[0]
        if len(rest) > 1 and rest[1] is not None:
            entry["caption_target"] = rest[1]
        regions.append(entry)
    return {
        "schema_version": 1,
        "page_fingerprint": fp,
        "image": {"width": width, "height": height},
        "regions": regions,
        "uncertainty": uncertainty,
        "trace": {"provider": "fake-vlm", "model": "m1", "request_id": "r1"},
        **extra,
    }


def parse(*items, **kw):
    return PageInventory.from_json(inv_json(*items, **kw))


def actions(plan):
    return sorted((r.action, r.kind) for r in plan.regions)


# ---------------------------------------------------------------- fixtures
def two_tables_as_one():
    base = layout(region("t", "table", (0.1, 0.1, 0.9, 0.6), confidence=0.4))
    inv = parse(
        ("a", "table", (0.1, 0.1, 0.9, 0.3), 0),
        ("b", "table", (0.1, 0.35, 0.9, 0.6), 1),
    )
    return base, inv


def table_split_in_three():
    boxes = [(0.1, 0.1, 0.9, 0.25), (0.1, 0.26, 0.9, 0.4), (0.1, 0.41, 0.9, 0.55)]
    base = layout(*[region(f"p{i}", "table", b, grid=(4, 3)) for i, b in enumerate(boxes)])
    inv = parse(("whole", "table", (0.1, 0.1, 0.9, 0.55), 0))
    return base, inv


def orphan_caption_missing_figure():
    base = layout(region("c", "caption", (0.1, 0.7, 0.5, 0.74), confidence=0.9))
    inv = parse(
        ("cap", "caption", (0.1, 0.7, 0.5, 0.74), 1, "fig"),
        ("fig", "figure", (0.1, 0.4, 0.5, 0.68), 0),
    )
    return base, inv


def test_two_tables_in_one_box_split_without_touching_values():
    base, inv = two_tables_as_one()
    plan = reconcile(base, inv)
    assert plan.status == "revised"
    assert actions(plan) == [("split", "table"), ("split", "table")]
    assert {r.bbox for r in plan.regions} == {(0.1, 0.1, 0.9, 0.3), (0.1, 0.35, 0.9, 0.6)}
    assert all(
        r.source_region_ids == ("t",) and r.reason_codes == ("split_merged_table",)
        for r in plan.regions
    )
    assert all(r.crop_box[0] <= r.bbox[0] and r.crop_box[3] >= r.bbox[3] for r in plan.regions)
    assert plan.changed_region_ids == tuple(r.region_id for r in plan.regions)
    plan.validate_invariants()
    assert "text" not in plan.to_json()


def test_table_split_into_three_merges_when_compatible():
    base, inv = table_split_in_three()
    plan = reconcile(base, inv)
    assert actions(plan) == [("merge", "table")]
    merged = plan.regions[0]
    assert merged.source_region_ids == ("p0", "p1", "p2")
    assert merged.bbox == (0.1, 0.1, 0.9, 0.55) and merged.inventory_region_ids == ("whole",)


def test_split_pieces_with_different_column_counts_are_not_merged():
    base, inv = table_split_in_three()
    base = layout(
        region("p0", "table", (0.1, 0.1, 0.9, 0.25), grid=(4, 3)),
        region("p1", "table", (0.1, 0.26, 0.9, 0.4), grid=(4, 5)),
    )
    inv = parse(("whole", "table", (0.1, 0.1, 0.9, 0.4), 0))
    plan = reconcile(base, inv)
    assert {r.action for r in plan.regions} == {"uncertain"}
    assert {r.region_id for r in plan.regions} == {"p0", "p1"}
    assert plan.status == "needs_review" and plan.needs_review


def test_side_by_side_or_distant_tables_are_not_merged():
    base = layout(
        region("l", "table", (0.05, 0.1, 0.45, 0.5)),
        region("r", "table", (0.55, 0.1, 0.95, 0.5)),
    )
    inv = parse(("one", "table", (0.05, 0.1, 0.95, 0.5)))
    plan = reconcile(base, inv)
    assert {r.reason_codes for r in plan.regions} == {("merge_incompatible",)}


def test_orphan_caption_is_linked_and_missing_figure_proposed():
    base, inv = orphan_caption_missing_figure()
    assert "orphan_caption" in assess(base).codes
    plan = reconcile(base, inv)
    by_kind = {r.kind: r for r in plan.regions}
    assert by_kind["figure"].action == "add" and by_kind["figure"].source_bboxes == ()
    assert by_kind["figure"].inventory_region_ids == ("fig",)
    assert by_kind["caption"].caption_target == by_kind["figure"].region_id
    assert "caption_linked" in by_kind["caption"].reason_codes
    assert by_kind["caption"].region_id == "c"
    assert plan.changed_region_ids == (by_kind["figure"].region_id,)


def test_sidebar_matched_and_unconfirmed_region_is_kept_uncertain_never_deleted():
    base = layout(
        region("main", "text", (0.05, 0.1, 0.65, 0.9), confidence=0.3),
        region("side", "sidebar", (0.7, 0.1, 0.95, 0.9)),
        region("ghost", "text", (0.3, 0.92, 0.6, 0.96)),
    )
    inv = parse(
        ("m", "text", (0.05, 0.1, 0.65, 0.9), 0),
        ("s", "sidebar", (0.7, 0.1, 0.95, 0.9), 1),
    )
    plan = reconcile(base, inv)
    kept = {r.region_id: r for r in plan.regions}
    assert set(kept) == {"main", "side", "ghost"}
    assert kept["ghost"].action == "uncertain"
    assert kept["ghost"].reason_codes == ("unconfirmed_by_inventory",)
    assert kept["main"].action == "keep" and kept["side"].action == "keep"


def test_rtl_two_column_reading_order_follows_inventory():
    base = layout(
        region("left", "text", (0.05, 0.1, 0.48, 0.9), reading_order=0, confidence=0.4),
        region("right", "text", (0.52, 0.1, 0.95, 0.9), reading_order=1),
    )
    inv = parse(
        ("r", "text", (0.52, 0.1, 0.95, 0.9), 0),
        ("l", "text", (0.05, 0.1, 0.48, 0.9), 1),
    )
    plan = reconcile(base, inv)
    assert [r.region_id for r in plan.regions] == ["right", "left"]
    assert plan.status == "revised"
    assert all("reading_order_updated" in r.reason_codes for r in plan.regions)
    assert plan.changed_region_ids == ()


def test_repeated_header_is_tagged_not_dropped():
    base = layout(region("h", "header", (0.1, 0.02, 0.9, 0.06), confidence=0.3))
    inv = parse(("rh", "repeated_header", (0.1, 0.02, 0.9, 0.06)))
    plan = reconcile(base, inv)
    only = plan.regions[0]
    assert (only.region_id, only.kind, only.action) == ("h", "repeated_header", "tag_artifact")
    assert only.bbox == (0.1, 0.02, 0.9, 0.06) and not only.changed


def test_type_conflict_is_uncertain_and_keeps_baseline_kind():
    base = layout(region("x", "text", (0.1, 0.1, 0.5, 0.3), confidence=0.2))
    inv = parse(("y", "figure", (0.1, 0.1, 0.5, 0.3)))
    plan = reconcile(base, inv)
    assert plan.regions[0].kind == "text" and plan.regions[0].reason_codes == ("type_conflict",)


def test_hallucinated_empty_inventory_never_deletes_baseline_regions():
    base = layout(
        region("a", "table", (0.1, 0.1, 0.9, 0.4), confidence=0.2),
        region("b", "text", (0.1, 0.5, 0.9, 0.9)),
    )
    plan = reconcile(base, parse())
    assert {r.region_id for r in plan.regions} == {"a", "b"}
    assert all(r.action == "uncertain" for r in plan.regions)


def test_uncertain_inventory_changes_nothing():
    base, inv = two_tables_as_one()
    inv = parse(
        ("a", "table", (0.1, 0.1, 0.9, 0.3)), ("b", "table", (0.1, 0.35, 0.9, 0.6)), uncertainty=0.9
    )
    plan = reconcile(base, inv)
    assert plan.status == "needs_review"
    assert [(r.region_id, r.action) for r in plan.regions] == [("t", "keep")]
    assert plan.regions[0].reason_codes == ("inventory_uncertain",)


def test_stable_ids_and_provider_ordering_do_not_change_plan():
    base, _ = two_tables_as_one()
    items = [("a", "table", (0.1, 0.1, 0.9, 0.3), 0), ("b", "table", (0.1, 0.35, 0.9, 0.6), 1)]
    first = reconcile(base, parse(*items))
    second = reconcile(base, parse(*reversed(items)))
    assert first.to_json() == second.to_json()
    ids = {r.region_id for r in first.regions}
    assert ids == {r.region_id for r in reconcile(base, parse(*items)).regions}
    assert stable_region_id(1, "table", (0.1, 0.1, 0.9, 0.3), ("t",)) in ids
    # a different parent set gives a different ID
    assert stable_region_id(1, "table", (0.1, 0.1, 0.9, 0.3), ("u",)) not in ids


def test_apply_plan_reextracts_changed_regions_only():
    base, inv = table_split_in_three()
    plan = reconcile(base, inv)
    calls = []
    result = apply_plan(plan, lambda r: calls.append(r.region_id) or r.crop_box)
    assert list(result) == [plan.regions[0].region_id] == calls
    keep_plan = reconcile(
        layout(region("k", "text", (0.1, 0.1, 0.5, 0.3))),
        parse(("i", "text", (0.1, 0.1, 0.5, 0.3))),
    )
    assert apply_plan(keep_plan, lambda r: pytest.fail("unchanged region re-extracted")) == {}


def test_plan_invariants_reject_dropped_or_untraceable_regions():
    base, inv = two_tables_as_one()
    plan = reconcile(base, inv)
    with pytest.raises(PlanInvariantError, match="dropped"):
        replace(
            plan, regions=plan.regions[:0], before_region_ids=("t", "zzz")
        ).validate_invariants()
    dup = replace(plan, regions=(plan.regions[0], plan.regions[0]))
    with pytest.raises(PlanInvariantError, match="duplicate"):
        dup.validate_invariants()
    orphan = replace(plan.regions[0], source_bboxes=(), inventory_region_ids=())
    with pytest.raises(PlanInvariantError, match="provenance"):
        replace(plan, regions=(orphan,)).validate_invariants()
    lost = replace(plan.regions[0], inventory_region_ids=())
    with pytest.raises(PlanInvariantError, match="inventory decision"):
        replace(plan, regions=(lost,)).validate_invariants()
    moved = replace(
        baseline_plan(base, status="unchanged", reason="r").regions[0], bbox=(0.2, 0.2, 0.3, 0.3)
    )
    with pytest.raises(PlanInvariantError, match="diverged"):
        replace(plan, regions=(moved,)).validate_invariants()
    cropless = replace(plan.regions[0], crop_box=(0.5, 0.5, 0.6, 0.6))
    with pytest.raises(PlanInvariantError, match="crop"):
        replace(plan, regions=(cropless,)).validate_invariants()


def test_crop_margin_respects_page_bounds():
    assert expand_box((0.0, 0.0, 1.0, 0.5), 0.05) == (0.0, 0.0, 1.0, 0.55)
    with pytest.raises(ValueError):
        expand_box((0, 0, 1, 1), 0.5)
    base, inv = two_tables_as_one()
    plan = reconcile(
        layout(region("t", "table", (0.0, 0.0, 1.0, 0.6))),
        parse(("a", "table", (0.0, 0.0, 1.0, 0.3)), ("b", "table", (0.0, 0.35, 1.0, 0.6))),
        policy=ReconcilePolicy(crop_margin=0.02),
    )
    assert all(0 <= v <= 1 for r in plan.regions for v in r.crop_box)


# ---------------------------------------------------------------- measured gain
def test_segmentation_failures_improve_without_extra_calls_on_clean_pages():
    truth_a = [(0.1, 0.1, 0.9, 0.3), (0.1, 0.35, 0.9, 0.6)]
    truth_b = [(0.1, 0.1, 0.9, 0.55)]
    cases = [(two_tables_as_one(), truth_a), (table_split_in_three(), truth_b)]
    for (base, inv), truth in cases:
        before = segmentation_f1(truth, [r.bbox for r in base.regions if r.kind == "table"])
        plan = reconcile(base, inv)
        after = segmentation_f1(truth, [r.bbox for r in plan.regions if r.kind == "table"])
        assert before < 1.0 and after == 1.0
    assert segmentation_f1([], []) == 1.0 and segmentation_f1([], [(0, 0, 1, 1)]) == 0.0
    assert segmentation_f1([(0, 0, 1, 1)], []) == 0.0
    metrics = layout_disagreement(reconcile(b, i) for (b, i), _ in cases)
    assert metrics["pages"] == 2 and metrics["split"] == 2 and metrics["merge"] == 1
    assert metrics["page_disagreement_rate"] == 1.0
    assert layout_disagreement([])["page_disagreement_rate"] == 0.0


# ---------------------------------------------------------------- schema
GOOD = inv_json(("a", "table", (0.1, 0.1, 0.5, 0.3), 0, None))


def bad(**patch):
    data = json.loads(json.dumps(GOOD))
    data.update(patch)
    return data


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("not json", "invalid_schema"),
        ("[]", "invalid_schema"),
        (bad(schema_version=2), "invalid_schema"),
        (bad(extra=1), "invalid_schema"),
        (bad(regions="x"), "invalid_schema"),
        (bad(page_fingerprint=""), "invalid_schema"),
        (bad(image={"width": 0, "height": 5}), "invalid_schema"),
        (bad(image={"width": 5}), "invalid_schema"),
        (bad(uncertainty=1.5), "invalid_schema"),
        (bad(uncertainty=True), "invalid_schema"),
        (bad(trace="x"), "invalid_schema"),
        (bad(trace={"provider": 5}), "invalid_schema"),
        (bad(counts={"table": 9}), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "table"}]), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "wat", "bbox": [0, 0, 1, 1]}]), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, 1.5, 1]}]), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, 1]}]), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, True, 1]}]), "invalid_schema"),
        (bad(regions=[{"id": "a", "type": "table", "bbox": [0.6, 0, 0.4, 1]}]), "invalid_schema"),
        (
            bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, 1, 1], "text": "7"}]),
            "invalid_schema",
        ),
        (
            bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, 1, 1], "reading_order": -1}]),
            "invalid_schema",
        ),
        (
            bad(regions=[{"id": "a", "type": "table", "bbox": [0, 0, 1, 1], "caption_target": 3}]),
            "invalid_schema",
        ),
        (
            bad(
                regions=[
                    {"id": "a", "type": "caption", "bbox": [0, 0, 1, 1], "caption_target": "ghost"}
                ]
            ),
            "invalid_schema",
        ),
        (bad(regions=[{"id": "", "type": "table", "bbox": [0, 0, 1, 1]}]), "invalid_schema"),
        (bad(regions=["x"]), "invalid_schema"),
        (bad(regions=[GOOD["regions"][0], GOOD["regions"][0]]), "conflicting_regions"),
        (
            bad(
                regions=[
                    {"id": "a", "type": "table", "bbox": [0.1, 0.1, 0.5, 0.5]},
                    {"id": "b", "type": "table", "bbox": [0.1, 0.1, 0.5, 0.52]},
                ]
            ),
            "conflicting_regions",
        ),
        (
            bad(
                regions=[
                    {"id": "a", "type": "table", "bbox": [0, 0, 0.2, 0.2], "reading_order": 0},
                    {"id": "b", "type": "text", "bbox": [0.5, 0.5, 0.7, 0.7], "reading_order": 0},
                ]
            ),
            "conflicting_regions",
        ),
    ],
)
def test_malformed_inventory_is_a_typed_failure(raw, code):
    payload = json.dumps(raw) if isinstance(raw, dict) else raw
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(payload)
    assert err.value.code == code and "secret" not in str(err.value)


def test_nan_and_oversized_and_mismatched_inventories():
    nan_json = json.dumps(GOOD).replace("0.5, 0.3]", "NaN, 0.3]")
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(nan_json)
    assert err.value.code == "invalid_schema"
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(json.dumps(GOOD), limits=InventoryLimits(max_output_bytes=20))
    assert err.value.code == "output_limit"
    many = inv_json(*[(f"r{i}", "text", (0.0, i / 50, 0.5, i / 50 + 0.01)) for i in range(40)])
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(many, limits=InventoryLimits(max_objects=10))
    assert err.value.code == "object_limit"
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(GOOD, expected_fingerprint="other")
    assert err.value.code == "fingerprint_mismatch"
    with pytest.raises(InventoryError) as err:
        PageInventory.from_json(GOOD, image=PageImage(b"x", 10, 10, 72))
    assert err.value.code == "image_mismatch"
    assert InventoryError("not-a-code").code == "provider_failed"
    with pytest.raises(ValueError):
        InventoryLimits(max_objects=0)


def test_inventory_json_roundtrip_and_counts():
    inv = PageInventory.from_json(
        inv_json(("a", "table", (0.1, 0.1, 0.5, 0.3), 0), ("b", "figure", (0.6, 0.1, 0.9, 0.3), 1))
    )
    assert inv.counts() == {"figure": 1, "table": 1}
    again = PageInventory.from_json(inv.to_json())
    assert again == inv and again.trace.provider == "fake-vlm"
    with_counts = dict(json.loads(inv.to_json()))
    assert PageInventory.from_json(with_counts) == inv


def test_rotation_maps_boxes_to_the_upright_page():
    box = (0.1, 0.2, 0.4, 0.3)
    for rot in (0, 90, 180, 270):
        assert to_canonical(from_canonical(box, rot), rot) == pytest.approx(box)
    inv = PageInventory.from_json(inv_json(("f", "figure", from_canonical(box, 90))), rotation=90)
    assert inv.regions[0].bbox == pytest.approx(box)
    with pytest.raises(ValueError):
        to_canonical(box, 45)
    with pytest.raises(ValueError):
        from_canonical(box, 45)
    assert union_box([(0, 0, 0.2, 0.2), (0.5, 0.5, 0.9, 0.9)]) == (0, 0, 0.9, 0.9)
    with pytest.raises(ValueError):
        union_box([])
    assert iou((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)) == 0 and area((0.2, 0.2, 0.1, 0.1)) == 0


# ---------------------------------------------------------------- suspicion
def test_clean_confident_layout_has_no_signals_and_missing_confidence_is_not_trusted():
    clean = layout(
        region("t", "table", (0.1, 0.1, 0.9, 0.4), confidence=0.95, reading_order=0),
        region("p", "text", (0.1, 0.5, 0.9, 0.9), confidence=0.9, reading_order=1),
    )
    report = assess(clean)
    assert not report.suspicious and report.confidence_reported
    unknown = layout(region("t", "table", (0.1, 0.1, 0.9, 0.4)))
    quiet = assess(unknown)
    assert not quiet.suspicious and not quiet.confidence_reported
    strict = assess(unknown, SuspicionPolicy(trigger_on_unreported_confidence=True))
    assert strict.codes == ("confidence_unreported",)


def test_each_suspicion_signal_fires():
    overlapping = layout(
        region("a", "table", (0.1, 0.1, 0.9, 0.5), confidence=0.9),
        region("b", "table", (0.1, 0.3, 0.9, 0.7), confidence=0.9),
    )
    assert assess(overlapping).codes == ("overlapping_tables",)
    adjacent = layout(
        region("a", "table", (0.1, 0.1, 0.9, 0.4), confidence=0.9),
        region("b", "table", (0.1, 0.405, 0.9, 0.6), confidence=0.9),
    )
    assert assess(adjacent).codes == ("adjacent_tables",)
    dup_order = layout(
        region("a", "text", (0.1, 0.1, 0.4, 0.2), confidence=0.9, reading_order=1),
        region("b", "text", (0.5, 0.1, 0.9, 0.2), confidence=0.9, reading_order=1),
    )
    assert assess(dup_order).codes == ("reading_order_conflict",)
    odd = layout(
        region("a", "text", (0.1, 0.1, 0.4, 0.2), confidence=0.9),
        unclaimed_ink=((0.1, 0.5, 0.7, 0.9),),
        expected_counts={"table": 2},
        parser_errors=("E1",),
    )
    assert assess(odd).codes == ("unclaimed_ink", "count_mismatch", "parser_error")
    ok_caption = layout(
        region("f", "figure", (0.1, 0.1, 0.5, 0.4), confidence=0.9),
        region("c", "caption", (0.1, 0.41, 0.5, 0.45), confidence=0.9),
    )
    assert not assess(ok_caption).suspicious
    with pytest.raises(ValueError):
        SuspicionPolicy(min_confidence=2)


def test_region_and_layout_validation():
    for kw in (
        {"kind": "bogus"},
        {"confidence": 2},
        {"reading_order": -1},
        {"region_id": ""},
    ):
        args = {"region_id": "x", "kind": "text", "bbox": (0, 0, 1, 1), **kw}
        with pytest.raises(ValueError):
            LayoutRegion(**args)
    with pytest.raises(ValueError):
        layout(region("x", "text", (0, 0, 1, 1)), region("x", "text", (0, 0, 1, 1)))
    with pytest.raises(ValueError):
        ProposedLayout(0, ())
    with pytest.raises(ValueError):
        PageImage(b"", 0, 1, 72)
    with pytest.raises(ValueError):
        PageImage(b"", 1, 1, 72, rotation=45)
    assert layout(region("a", "text", (0, 0, 1, 1))).counts() == {"text": 1}


def test_proposed_layout_from_page_evidence_has_stable_ids():
    table = ExtractionTable(
        (0.1, 0.1, 0.9, 0.5), 1, 1, (ExtractionCell(0, 0, "secret value", (0.1, 0.1, 0.9, 0.5)),)
    )
    outcome = PageExtractionOutcome(
        2,
        PageExtractionStatus.EXTRACTED,
        tables=(table,),
        regions=(
            ExtractionRegion("image", (0.1, 0.6, 0.5, 0.9)),
            ExtractionRegion("paragraph", (0.1, 0.55, 0.5, 0.59)),
            ExtractionRegion("mystery", (0.5, 0.5, 0.6, 0.6)),
            ExtractionRegion("image", (0.1, 0.6, 0.5, 0.9)),
        ),
    )
    one = proposed_from_outcome(outcome, page_fingerprint="fp")
    two = proposed_from_outcome(outcome, page_fingerprint="fp")
    assert one == two and one.page == 2
    assert one.counts() == {"figure": 2, "table": 1, "text": 1}
    assert len({r.region_id for r in one.regions}) == 4
    assert "secret value" not in repr(one)


# ---------------------------------------------------------------- planner
class Images:
    def __init__(self, rotation=0, dpi=100, width=1000, height=1400):
        self.calls = 0
        self.rotation, self.dpi, self.width, self.height = rotation, dpi, width, height

    def render(self, page, *, max_dpi, max_pixels):
        self.calls += 1
        return PageImage(b"png", self.width, self.height, self.dpi, self.rotation)


class Provider:
    id, version, remote, cost_per_call = "fake-vlm", "1", False, 0.25

    def __init__(self, payload=None, *, fail=0, sleep=0.0, error=None):
        self.payload, self.fail, self.sleep, self.error = payload, fail, sleep, error
        self.calls = 0
        self.live = self.peak = 0
        self._lock = threading.Lock()

    def inventory(self, page_image, proposed_layout, budget):
        with self._lock:
            self.calls += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
            n = self.calls
        try:
            if self.sleep:
                time.sleep(self.sleep)
            if self.error or n <= self.fail:
                raise RuntimeError(self.error or "SECRET-PROVIDER-DETAIL")
            return self.payload
        finally:
            with self._lock:
                self.live -= 1


def enabled(**kw):
    return LayoutInventoryPolicy(enabled=True, **kw)


def suspicious_layout(page=1):
    return ProposedLayout(
        page,
        (region("t", "table", (0.1, 0.1, 0.9, 0.6), confidence=0.4),),
        page_fingerprint=FP,
    )


SPLIT_PAYLOAD = inv_json(
    ("a", "table", (0.1, 0.1, 0.9, 0.3), 0), ("b", "table", (0.1, 0.35, 0.9, 0.6), 1)
)


def test_clean_page_makes_zero_provider_and_render_calls():
    provider, images = Provider(SPLIT_PAYLOAD), Images()
    planner = LayoutInventoryPlanner(provider, images, enabled())
    clean = layout(region("t", "table", (0.1, 0.1, 0.9, 0.6), confidence=0.97))
    result = planner.plan_page(clean)
    assert (provider.calls, images.calls, result.provider_calls) == (0, 0, 0)
    assert result.skipped_reason == "clean_page" and result.plan.status == "unchanged"
    assert result.plan.regions[0].reason_codes == ("clean_page",)


def test_default_policy_is_opt_in_and_makes_no_calls():
    provider, images = Provider(SPLIT_PAYLOAD), Images()
    result = LayoutInventoryPlanner(provider, images).plan_page(suspicious_layout())
    assert result.skipped_reason == "inventory_disabled" and provider.calls == 0
    none = LayoutInventoryPlanner(None, None, enabled()).plan_page(suspicious_layout())
    assert none.skipped_reason == "inventory_disabled"


def test_low_confidence_page_makes_exactly_one_bounded_call():
    provider, images = Provider(SPLIT_PAYLOAD), Images(dpi=120)
    planner = LayoutInventoryPlanner(provider, images, enabled())
    result = planner.plan_page(suspicious_layout())
    assert (provider.calls, images.calls, result.provider_calls) == (1, 1, 1)
    assert result.plan.status == "revised" and result.suspicion.codes == ("low_confidence",)
    assert result.plan.suspicion_codes == ("low_confidence",)
    assert result.plan.trace.provider == "fake-vlm"
    assert result.failure_code == ""


def test_rotated_page_inventory_is_mapped_to_canonical_coordinates():
    target = (0.1, 0.65, 0.5, 0.9)
    payload = inv_json(
        ("a", "table", (0.1, 0.1, 0.9, 0.3), 0),
        ("b", "table", (0.1, 0.35, 0.9, 0.6), 1),
        ("fig", "figure", from_canonical(target, 90)),
    )
    payload["regions"][0]["bbox"] = list(from_canonical((0.1, 0.1, 0.9, 0.3), 90))
    payload["regions"][1]["bbox"] = list(from_canonical((0.1, 0.35, 0.9, 0.6), 90))
    provider, images = Provider(payload), Images(rotation=90)
    result = LayoutInventoryPlanner(provider, images, enabled()).plan_page(suspicious_layout())
    figure = next(r for r in result.plan.regions if r.kind == "figure")
    assert figure.bbox == pytest.approx(target)
    assert sum(r.action == "split" for r in result.plan.regions) == 2


def test_denied_external_upload_is_safe_failure_before_render():
    provider, images = Provider(SPLIT_PAYLOAD), Images()
    provider.remote = True
    result = LayoutInventoryPlanner(provider, images, enabled()).plan_page(suspicious_layout())
    assert result.failure_code == "external_upload_denied"
    assert (provider.calls, images.calls) == (0, 0)
    assert result.plan.status == "baseline" and result.plan.regions[0].action == "keep"
    allowed = LayoutInventoryPlanner(provider, images, enabled(allow_remote_upload=True))
    assert allowed.plan_page(suspicious_layout()).plan.status == "revised"


def test_timeout_retries_exhausted_and_needs_review_policy():
    slow = Provider(SPLIT_PAYLOAD, sleep=0.3)
    budget = InventoryBudget(timeout_seconds=0.05, max_retries=1)
    result = LayoutInventoryPlanner(
        slow, Images(), enabled(budget=budget, on_failure="needs_review")
    ).plan_page(suspicious_layout())
    assert result.failure_code == "timeout" and result.provider_calls == 2
    assert result.plan.status == "needs_review" and result.plan.failure_code == "timeout"
    boom = Provider(error="SECRET-PROVIDER-DETAIL")
    result = LayoutInventoryPlanner(
        boom, Images(), enabled(budget=InventoryBudget(max_retries=2))
    ).plan_page(suspicious_layout())
    assert result.failure_code == "provider_failed" and boom.calls == 3
    assert "SECRET" not in result.plan.to_json()


def test_transient_failure_is_retried_then_succeeds():
    flaky = Provider(SPLIT_PAYLOAD, fail=1)
    result = LayoutInventoryPlanner(flaky, Images(), enabled()).plan_page(suspicious_layout())
    assert result.plan.status == "revised" and flaky.calls == 2 and result.provider_calls == 2


def test_budget_overflow_is_refused_and_recorded():
    boom = Provider(error="x")
    planner = LayoutInventoryPlanner(
        boom, Images(), enabled(budget=InventoryBudget(max_calls=1, max_retries=3))
    )
    ledger = planner.new_ledger()
    result = planner.plan_page(suspicious_layout(), ledger=ledger)
    assert result.failure_code == "budget_exceeded" and boom.calls == 1
    assert ledger.refusals == 1 and ledger.calls == 1
    pricey = LayoutInventoryPlanner(
        Provider(SPLIT_PAYLOAD), Images(), enabled(budget=InventoryBudget(max_cost=0.1))
    )
    assert pricey.plan_page(suspicious_layout()).failure_code == "budget_exceeded"
    zero = LayoutInventoryPlanner(
        Provider(SPLIT_PAYLOAD), Images(), enabled(budget=InventoryBudget(max_calls=0))
    )
    assert zero.plan_page(suspicious_layout()).failure_code == "budget_exceeded"


def test_cancellation_stops_before_any_call():
    provider = Provider(SPLIT_PAYLOAD)
    cancel = threading.Event()
    cancel.set()
    result = LayoutInventoryPlanner(provider, Images(), enabled()).plan_page(
        suspicious_layout(), cancel=cancel
    )
    assert result.failure_code == "cancelled" and provider.calls == 0
    mid = threading.Event()

    class Cancelling(Provider):
        def inventory(self, *a):
            mid.set()
            raise RuntimeError("x")

    p = Cancelling()
    result = LayoutInventoryPlanner(
        p, Images(), enabled(budget=InventoryBudget(max_retries=5))
    ).plan_page(suspicious_layout(), cancel=mid)
    assert result.failure_code == "cancelled" and result.provider_calls == 1


def test_oversized_image_and_bad_provider_output_fail_safely():
    result = LayoutInventoryPlanner(Provider(SPLIT_PAYLOAD), Images(dpi=300), enabled()).plan_page(
        suspicious_layout()
    )
    assert result.failure_code == "image_limit"
    result = LayoutInventoryPlanner(
        Provider(SPLIT_PAYLOAD), Images(width=5000, height=5000), enabled()
    ).plan_page(suspicious_layout())
    assert result.failure_code == "image_limit"
    for payload, code in (
        ("{bad", "invalid_schema"),
        (inv_json(fp="other"), "fingerprint_mismatch"),
    ):
        result = LayoutInventoryPlanner(Provider(payload), Images(), enabled()).plan_page(
            suspicious_layout()
        )
        assert result.failure_code == code and result.plan.status == "baseline"


def test_concurrency_is_bounded_results_ordered_and_budget_shared():
    provider = Provider(SPLIT_PAYLOAD, sleep=0.03)
    planner = LayoutInventoryPlanner(
        provider,
        Images(),
        enabled(max_concurrency=2, budget=InventoryBudget(max_calls=4, max_cost=10)),
    )
    pages = [suspicious_layout(p) for p in range(1, 7)]
    results = planner.plan_pages(pages)
    assert [r.plan.page for r in results] == [1, 2, 3, 4, 5, 6]
    assert provider.peak <= 2
    assert sum(r.failure_code == "budget_exceeded" for r in results) == 2
    assert provider.calls == 4
    with pytest.raises(ValueError):
        LayoutInventoryPolicy(max_concurrency=0)
    with pytest.raises(ValueError):
        LayoutInventoryPolicy(on_failure="x")
    with pytest.raises(ValueError):
        InventoryBudget(max_dpi=1)
    with pytest.raises(ValueError):
        InventoryBudget(timeout_seconds=0)
    with pytest.raises(ValueError):
        InventoryBudget(max_cost=float("nan"))


def test_logs_are_sanitized(caplog):
    caplog.set_level(logging.DEBUG, logger="ragnor.ingestion.layout")
    boom = Provider(error="SECRET-PROVIDER-DETAIL")
    LayoutInventoryPlanner(boom, Images(), enabled()).plan_page(suspicious_layout())
    LayoutInventoryPlanner(Provider(SPLIT_PAYLOAD), Images(), enabled()).plan_page(
        suspicious_layout()
    )
    text = caplog.text
    assert "inventory failed" in text and "code=provider_failed" in text
    assert "SECRET" not in text and "png" not in text


def test_base_install_has_no_network_or_model_dependency():
    script = (
        "import builtins\n"
        "orig = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.split('.')[0] in {'litellm','httpx','requests','PIL','openai','mcp'}:\n"
        "        raise ImportError(name)\n"
        "    return orig(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import ragnor.ingestion.layout as l\n"
        "assert l.LayoutInventoryPlanner\n"
        "assert not l.LayoutInventoryPolicy().enabled\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
