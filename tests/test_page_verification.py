"""EX-08: verification checks, bounded repair state machine, publication."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import threading
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragnor.ingestion import PageExtractionOutcome, PageExtractionStatus
from ragnor.ingestion.extraction_types import ExtractionCell, ExtractionRegion, ExtractionTable
from ragnor.ingestion.verification import (
    ALLOWLIST,
    ArithmeticRelation,
    AttemptRecord,
    AuditReader,
    CheckUnavailableError,
    ExpectedPage,
    ExpectedRegion,
    Finding,
    InMemoryAttemptStore,
    InMemoryRevisionStore,
    PageVerificationPipeline,
    PipelinePageVerifier,
    PublishPolicy,
    RegionArtifact,
    RegionRef,
    RenderBackResult,
    RepairAction,
    RepairActionKind,
    RepairLimits,
    RepairStateMachine,
    VerifierSettings,
    default_checks,
    parse_decimal,
    publish_verified,
)
from ragnor.ingestion.verification.geometry import covered_fraction, iou
from ragnor.ingestion.verification.repair import DEFAULT_LADDER

K = RepairActionKind


def table(rows, bbox=(0.1, 0.1, 0.9, 0.5)):
    cells = tuple(
        ExtractionCell(r, c, text, (0.1, 0.1, 0.2, 0.2))
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    )
    return ExtractionTable(bbox, len(rows), len(rows[0]), cells)


INVOICE = [
    ["Item", "Amount"],
    ["A", "$1,000.10"],
    ["B", "250.20"],
    ["C", "(50.30)"],
    ["Total", "1,200.00"],
]
REL = (ArithmeticRelation(0, 1, 1, 3, 4),)


def outcome(tables=(), regions=(), status=PageExtractionStatus.EXTRACTED, page=1):
    return PageExtractionOutcome(
        page, status, text="x", tables=tuple(tables), regions=tuple(regions)
    )


def verify(out, **kw):
    return PageVerificationPipeline().verify(out, **kw)


# ---------------------------------------------------------------- arithmetic
def test_decimal_safe_totals_pass_exactly():
    report = verify(outcome([table(INVOICE)]), relations=REL)
    assert report.status == "verified" and "arithmetic" in report.checks_run
    assert parse_decimal("(50.30)") == Decimal("-50.30")
    assert Decimal("1000.10") + Decimal("250.20") - Decimal("50.30") == Decimal("1200.00")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[2].__setitem__(1, "251.20"),  # wrong row
        lambda rows: rows.insert(3, list(rows[2])),  # duplicated row
        lambda rows: rows.pop(2),  # missing row
    ],
)
def test_wrong_duplicated_or_missing_row_is_caught_and_amount_never_edited(mutate):
    rows = [list(r) for r in INVOICE]
    mutate(rows)
    t = table(rows)
    before = [c.text for c in t.cells]
    report = verify(outcome([t]), relations=REL)
    assert report.status == "failed"
    assert (
        report.findings[0].check_id == "arithmetic"
        and report.findings[0].disposition == "repair_candidate"
    )
    assert [c.text for c in t.cells] == before
    assert report.failing_region_ids


def test_unrecognized_or_unparseable_amounts_warn_and_never_fail():
    report = verify(outcome([table(INVOICE)]))
    assert report.status == "verified"
    assert [f.severity for f in report.findings] == ["warning"]
    assert report.findings[0].check_id == "arithmetic"
    rows = [list(r) for r in INVOICE]
    rows[1][1] = "twelve"
    report = verify(outcome([table(rows)]), relations=REL)
    assert report.status == "verified" and report.findings[0].severity == "warning"
    strict = PageVerificationPipeline(VerifierSettings(fail_on_warning=True))
    assert strict.verify(outcome([table(rows)]), relations=REL).status == "needs_review"


def test_arithmetic_relation_bounds_and_tolerance():
    assert (
        verify(outcome([table(INVOICE)]), relations=(ArithmeticRelation(3, 1, 1, 3, 4),)).status
        == "failed"
    )
    assert (
        verify(outcome([table(INVOICE)]), relations=(ArithmeticRelation(0, 1, 1, 9, 4),)).status
        == "failed"
    )
    rows = [list(r) for r in INVOICE]
    rows[4][1] = "1,200.01"
    loose = PageVerificationPipeline(VerifierSettings(arithmetic_tolerance=Decimal("0.02")))
    assert loose.verify(outcome([table(rows)]), relations=REL).status == "verified"
    assert verify(outcome([table(rows)]), relations=REL).status == "failed"
    with pytest.raises(ValueError):
        VerifierSettings(arithmetic_tolerance=Decimal("-1"))
    with pytest.raises(ValueError):
        VerifierSettings(coverage_min=0)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1,234.50", "1234.50"),
        ("$ 5", "5"),
        ("€7.25", "7.25"),
        ("(12)", "-12"),
        ("-3", "-3"),
        ("٣٤٫", None),
        ("١٢", "12"),
        ("1,23", None),
        ("12abc", None),
        ("", None),
        ("1.2.3", None),
    ],
)
def test_parse_decimal(text, value):
    assert parse_decimal(text) == (None if value is None else Decimal(value))


@given(
    st.lists(
        st.decimals(min_value=-(10**6), max_value=10**6, places=2, allow_nan=False),
        min_size=1,
        max_size=8,
    )
)
def test_sum_check_matches_decimal_arithmetic_for_any_amounts(amounts):
    rows = [["h", "Amount"], *[[f"r{i}", f"{a:.2f}"] for i, a in enumerate(amounts)]]
    rows.append(["Total", f"{sum(amounts, Decimal(0)):.2f}"])
    rel = (ArithmeticRelation(0, 1, 1, len(amounts), len(amounts) + 1),)
    report = verify(outcome([table(rows)]), relations=rel)
    assert report.status == "verified"


# ---------------------------------------------------------------- count, coverage
def test_count_catches_omitted_table_and_flags_unexpected_regions():
    expected = ExpectedPage(
        (
            ExpectedRegion("T1", "table", (0.1, 0.1, 0.9, 0.3)),
            ExpectedRegion("T2", "table", (0.1, 0.5, 0.9, 0.7)),
            ExpectedRegion("F1", "figure", (0.1, 0.75, 0.5, 0.95)),
            ExpectedRegion("H1", "heading", (0.1, 0.02, 0.5, 0.06)),
        )
    )
    out = outcome(
        [table(INVOICE, (0.1, 0.1, 0.9, 0.3))],
        [
            ExtractionRegion("image", (0.1, 0.75, 0.5, 0.95)),
            ExtractionRegion("text", (0.1, 0.96, 0.2, 0.99)),
        ],
    )
    report = verify(out, expected=expected)
    missing = {f.region_id for f in report.findings if f.severity == "error"}
    assert missing == {"T2", "H1"} and report.status == "failed"
    assert report.failing_region_ids == ("H1", "T2")
    assert any(f.severity == "warning" and f.observed == "unexpected text" for f in report.findings)


def test_coverage_uses_content_boxes_not_page_area():
    content = ExpectedPage(content_boxes=((0.1, 0.1, 0.9, 0.3), (0.1, 0.6, 0.9, 0.8)))
    covered = outcome(
        regions=[
            ExtractionRegion("text", (0.1, 0.1, 0.9, 0.3)),
            ExtractionRegion("text", (0.1, 0.6, 0.9, 0.8)),
        ]
    )
    report = verify(covered, expected=content)
    # whitespace is not a false positive
    assert report.status == "verified"
    assert not [f for f in report.findings if f.severity == "error"]
    gap = outcome(regions=[ExtractionRegion("text", (0.1, 0.1, 0.9, 0.3))])
    failed = verify(gap, expected=content)
    assert failed.status == "failed"
    assert [f.region_id for f in failed.findings if f.severity == "error"] == ["content-1"]
    # a huge extracted box does not prove coverage of content it does not touch
    assert covered_fraction((0, 0, 0.1, 0.1), [(0.5, 0.5, 1, 1)]) == 0.0
    assert covered_fraction((0, 0, 0, 0), [(0, 0, 1, 1)]) == 0.0
    assert covered_fraction((0, 0, 1, 1), [(0, 0, 0.6, 1), (0.4, 0, 1, 1)]) == pytest.approx(1.0)
    assert covered_fraction((0, 0, 1, 1), [(0, 0, 0.5, 0.5), (0, 0.25, 0.5, 1)]) == pytest.approx(
        0.5
    )
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0


# ---------------------------------------------------------------- shape
def test_shape_compares_with_independent_parser_when_present():
    t = table(INVOICE)
    same = table(INVOICE)
    assert verify(outcome([t]), alternate_tables=(same,)).status == "verified"
    other = table(INVOICE[:-1])
    report = verify(outcome([t]), alternate_tables=(other,))
    assert report.status == "failed" and report.findings[0].observed == "5x2"
    merged = ExtractionTable(
        t.bbox, 5, 2, (ExtractionCell(0, 0, "H", (0.1, 0.1, 0.2, 0.2), colspan=2),)
    )
    spans = verify(outcome([t]), alternate_tables=(merged,))
    assert spans.findings[0].observed == "merged spans differ"
    far = table(INVOICE, (0.1, 0.8, 0.9, 0.95))
    assert (
        verify(outcome([t]), alternate_tables=(far,)).findings[0].observed == "no independent table"
    )
    assert verify(outcome([t])).checks_run == ("completeness", "arithmetic")


# ---------------------------------------------------------------- pipeline policy
def test_incomplete_page_never_verifies_and_mandatory_checks_cannot_be_skipped():
    blank = outcome(status=PageExtractionStatus.NEEDS_OCR)
    report = verify(blank)
    assert report.status == "needs_review" and report.extraction_status == "needs_ocr"
    pipeline = PageVerificationPipeline(
        VerifierSettings(mandatory=frozenset({"completeness", "count"}))
    )
    report = pipeline.verify(outcome())
    assert report.status == "needs_review" and "count" not in report.checks_run
    assert report.findings[0].observed == "required input unavailable"


def test_crashing_check_is_reported_not_skipped():
    class Boom:
        check_id = "boom"

        def run(self, ctx):
            raise RuntimeError("SECRET-PARSER-DETAIL")

    report = PageVerificationPipeline(checks=[*default_checks(), Boom()]).verify(outcome())
    assert report.status == "needs_review" and "boom" not in report.checks_run
    assert "SECRET" not in report.to_json()
    with pytest.raises(ValueError):
        PageVerificationPipeline(checks=[Boom(), Boom()])


def test_render_back_runs_only_for_targeted_failing_regions():
    calls = []

    class RB:
        def __init__(self, sim, boom=False):
            self.sim, self.boom = sim, boom

        def compare(self, page, region_id, bbox):
            calls.append(region_id)
            if self.boom:
                raise RuntimeError("x")
            return RenderBackResult(self.sim, "artifacts/r1")

    expected = ExpectedPage((ExpectedRegion("T9", "table", (0.1, 0.5, 0.9, 0.7)),))
    clean = PageVerificationPipeline(render_back=RB(1.0)).verify(outcome(), expected=ExpectedPage())
    assert calls == [] and clean.render_back_regions == ()
    p = PageVerificationPipeline(render_back=RB(1.0), render_back_limit=1)
    report = p.verify(outcome(), expected=expected)
    assert calls == ["T9"] and report.render_back_regions == ("T9",)
    assert report.findings[-1].check_id == "render_back" and report.findings[-1].severity == "info"
    low = PageVerificationPipeline(render_back=RB(0.2)).verify(outcome(), expected=expected)
    assert low.status == "needs_review"
    boom = PageVerificationPipeline(render_back=RB(0, True)).verify(outcome(), expected=expected)
    assert boom.findings[-1].observed == "render-back raised"


def test_report_is_versioned_and_deterministic():
    a = verify(outcome([table(INVOICE)]), relations=REL)
    b = verify(outcome([table(INVOICE)]), relations=REL)
    assert a.to_json() == b.to_json()
    data = json.loads(a.to_json())
    assert data["settings_version"] == "1" and len(data["settings_fingerprint"]) == 64
    assert VerifierSettings(version="2").fingerprint != VerifierSettings().fingerprint
    with pytest.raises(ValueError):
        Finding(1, "r", "c", "bad", "o", "e", "s")
    with pytest.raises(ValueError):
        Finding(1, "r", "c", "error", "o" * 201, "e", "s")
    with pytest.raises(ValueError):
        Finding(0, "r", "c", "error", "o", "e", "s")


def test_page_verifier_adapter_gates_extractor_acceptance():
    pipeline = PageVerificationPipeline()
    adapter = PipelinePageVerifier(pipeline, lambda page: {"relations": REL})
    good = outcome([table(INVOICE)])
    bad_rows = [list(r) for r in INVOICE]
    bad_rows[2][1] = "1.00"
    assert adapter.verify(good) is True and adapter.report(1).status == "verified"
    assert adapter.verify(outcome([table(bad_rows)], page=2)) is False
    assert adapter.report(2).status == "failed" and adapter.report(9) is None
    assert adapter.options["settings_sha256"] == pipeline.settings.fingerprint
    assert adapter.id and adapter.version == "1"

    def explode(page):
        raise RuntimeError("context failed")

    with pytest.raises(RuntimeError):
        PipelinePageVerifier(pipeline, explode).verify(good)
    assert PipelinePageVerifier(pipeline).verify(good) is True


# ---------------------------------------------------------------- repair
class Executor:
    fingerprint = "fp-test"

    def __init__(self, succeed_at=None, *, crash_at=(), pixels=0, bbox=None, region_id=None):
        self.succeed_at, self.crash_at = succeed_at, set(crash_at)
        self.pixels, self.bbox, self.region_id = pixels, bbox, region_id
        self.calls: list[tuple[str, RepairAction]] = []

    def execute(self, region, action, *, max_pixels):
        self.calls.append((region.region_id, action))
        n = len(self.calls)
        if n in self.crash_at:
            raise RuntimeError("SECRET-OCR-CRASH")
        ok = self.succeed_at is not None and n >= self.succeed_at
        return RegionArtifact(
            self.region_id or region.region_id,
            self.bbox or region.bbox,
            f"artifacts/{region.region_id}/{n}" + ("" if ok else "-bad"),
            pixels=self.pixels,
        )


def judge(region, artifact):
    if artifact.artifact_ref.endswith("-bad"):
        return [
            Finding(
                1,
                region.region_id,
                "count",
                "error",
                "still missing",
                "present",
                "v",
                "repair_candidate",
            )
        ]
    return []


def ref(rid="R1", checks=("count",), bbox=(0.1, 0.1, 0.5, 0.3)):
    return RegionRef(1, rid, bbox, checks)


class Clock:
    def __init__(self, step=0.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def machine(  # noqa: PLR0913
    executor, store=None, *, verifier=judge, limits=RepairLimits(), clock=None, ladder=None
):
    return RepairStateMachine(
        executor, verifier, store or InMemoryAttemptStore(), limits, ladder, clock or Clock()
    )


def test_repair_retries_only_failed_regions_and_never_more_than_three():
    ex = Executor(succeed_at=None)
    store = InMemoryAttemptStore()
    result = machine(ex, store).run(tenant_id="t1", run_id="run", regions=[ref("R1"), ref("R2")])
    assert [r.attempts for r in result.regions] == [3, 3] and result.needs_review == ("R1", "R2")
    assert {r.reason for r in result.regions} == {"attempts_exhausted"}
    for rid in ("R1", "R2"):
        keys = [a.key for r, a in ex.calls if r == rid]
        assert len(keys) == 3 == len(set(keys))  # no repeated action or params
    assert len(store.list("t1", "run")) == 6


def test_successful_repair_promotes_and_rebinds_crop():
    ex = Executor(succeed_at=2, bbox=(0.08, 0.08, 0.52, 0.32))
    result = machine(ex).run(tenant_id="t", run_id="r", regions=[ref()])
    only = result.regions[0]
    assert (
        only.status == "promoted"
        and only.attempts == 2
        and only.final_bbox == (0.08, 0.08, 0.52, 0.32)
    )
    assert only.artifact_ref == "artifacts/R1/2" and result.promoted == ("R1",)


def test_attempt_limit_is_configurable_but_capped_at_three():
    ex = Executor()
    result = machine(ex, limits=RepairLimits(max_attempts_per_region=1)).run(
        tenant_id="t", run_id="r", regions=[ref()]
    )
    assert result.regions[0].attempts == 1
    for bad in (0, 4):
        with pytest.raises(ValueError):
            RepairLimits(max_attempts_per_region=bad)
    for kw in ({"max_cost": 0}, {"max_seconds": float("inf")}, {"max_dpi": 0}):
        with pytest.raises(ValueError):
            RepairLimits(**kw)


def test_cost_and_total_attempt_budgets_are_enforced():
    ladder = {
        "count": (
            RepairAction.of(K.RAISE_RESOLUTION, dpi=200),
            RepairAction.of(K.RAISE_RESOLUTION, dpi=300),
            RepairAction.of(K.ADJUST_CROP, margin=0.01),
        )
    }
    ex = Executor()
    result = machine(ex, limits=RepairLimits(max_cost=3.0), ladder=ladder).run(
        tenant_id="t", run_id="r", regions=[ref()]
    )
    assert result.regions[0].reason == "budget" and len(ex.calls) == 1  # 2.0 spent, next costs 2.0
    ex = Executor()
    result = machine(ex, limits=RepairLimits(max_total_attempts=4)).run(
        tenant_id="t", run_id="r", regions=[ref("A"), ref("B")]
    )
    assert len(ex.calls) == 4 and [r.reason for r in result.regions] == [
        "attempts_exhausted",
        "budget",
    ]


def test_time_budget_is_cumulative_across_replays():
    ex = Executor()
    store = InMemoryAttemptStore()
    limits = RepairLimits(max_seconds=1.0)
    first = machine(ex, store, limits=limits, clock=Clock(step=0.6)).run(
        tenant_id="t", run_id="r", regions=[ref()]
    )
    assert first.regions[0].reason == "budget" and len(ex.calls) == 2
    again = machine(Executor(), store, limits=limits, clock=Clock(step=0.0)).run(
        tenant_id="t", run_id="r", regions=[ref()]
    )
    assert again.regions[0].reason == "budget"


def test_cancellation_stops_before_and_after_execution():
    cancel = threading.Event()
    cancel.set()
    ex = Executor()
    result = machine(ex).run(tenant_id="t", run_id="r", regions=[ref()], cancel=cancel)
    assert result.regions[0].reason == "cancelled" and ex.calls == []

    cancel2 = threading.Event()

    class Cancelling(Executor):
        def execute(self, region, action, **kw):
            cancel2.set()
            return super().execute(region, action, **kw)

    store = InMemoryAttemptStore()
    result = machine(Cancelling(succeed_at=1), store).run(
        tenant_id="t", run_id="r", regions=[ref()], cancel=cancel2
    )
    assert result.regions[0].status == "needs_review" and result.regions[0].artifact is None
    assert store.list("t", "r")[0].outcome == "cancelled"


def test_replay_never_grants_extra_attempts_or_repeats_actions():
    store = InMemoryAttemptStore()
    ex1 = Executor()
    machine(ex1, store, limits=RepairLimits(max_attempts_per_region=2)).run(
        tenant_id="t", run_id="r", regions=[ref()]
    )
    ex2 = Executor()
    replay = machine(ex2, store).run(tenant_id="t", run_id="r", regions=[ref()])
    assert len(ex2.calls) == 1  # 3 allowed, 2 already used
    all_keys = [(a.action_kind, a.action_params) for a in store.list("t", "r")]
    assert len(all_keys) == len(set(all_keys)) == 3 and replay.regions[0].attempts == 3
    # a run that already promoted is idempotent and does not re-execute
    ok_store = InMemoryAttemptStore()
    machine(Executor(succeed_at=1), ok_store).run(tenant_id="t", run_id="r", regions=[ref()])
    ex3 = Executor()
    done = machine(ex3, ok_store).run(tenant_id="t", run_id="r", regions=[ref()])
    assert ex3.calls == [] and done.regions[0].status == "promoted" and done.regions[0].artifact_ref


def test_crash_mid_attempt_still_counts_as_an_attempt():
    store = InMemoryAttemptStore()
    store.append(
        "t",
        AttemptRecord(
            "r",
            "R1",
            1,
            1,
            (0.1, 0.1, 0.5, 0.3),
            "fp",
            "adjust_crop",
            (("margin", 0.01),),
            ("count",),
        ),
    )
    ex = Executor()
    result = machine(ex, store).run(tenant_id="t", run_id="r", regions=[ref()])
    assert len(ex.calls) == 2 and result.regions[0].attempts == 3
    assert store.list("t", "r")[0].outcome == "interrupted"
    assert ("adjust_crop", (("margin", 0.01),)) not in [
        (a.kind.value, a.params) for _, a in ex.calls
    ]


def test_fault_injection_never_publishes_partial_or_leaks_details():
    store = InMemoryAttemptStore()
    ex = Executor(succeed_at=3, crash_at=(2,))
    result = machine(ex, store).run(tenant_id="t", run_id="r", regions=[ref()])
    assert result.regions[0].status == "promoted"
    outcomes = [r.outcome for r in store.list("t", "r")]
    assert outcomes == ["still_failing", "executor_failed", "promoted"]
    assert all("SECRET" not in r.to_json() for r in store.list("t", "r"))

    def verifier_crash(region, artifact):
        raise RuntimeError("verifier down")

    r2 = machine(Executor(succeed_at=1), verifier=verifier_crash).run(
        tenant_id="t", run_id="x", regions=[ref()]
    )
    assert r2.regions[0].status == "needs_review" and r2.regions[0].artifact is None
    huge = machine(Executor(succeed_at=1, pixels=10**9), limits=RepairLimits(max_pixels=1000)).run(
        tenant_id="t", run_id="y", regions=[ref()]
    )
    assert huge.regions[0].status == "needs_review"
    wrong = machine(Executor(succeed_at=1, region_id="OTHER")).run(
        tenant_id="t", run_id="z", regions=[ref()]
    )
    assert wrong.regions[0].status == "needs_review"
    away = machine(Executor(succeed_at=1, bbox=(0.7, 0.7, 0.9, 0.9))).run(
        tenant_id="t", run_id="w", regions=[ref()]
    )
    assert away.regions[0].status == "needs_review"


def test_no_candidate_actions_means_needs_review():
    result = machine(Executor()).run(
        tenant_id="t", run_id="r", regions=[ref(checks=("unknown_check",))]
    )
    assert result.regions[0].reason == "no_actions"
    with pytest.raises(ValueError):
        RegionRef(1, "R", (0, 0, 1, 1), ())


def test_actions_cannot_carry_values_and_ladders_respect_the_allowlist():
    with pytest.raises(ValueError):
        RepairAction.of(K.ADJUST_CROP, value="1200.00")
    with pytest.raises(ValueError):
        RepairAction.of(K.RESEGMENT, strategy="has spaces")
    with pytest.raises(ValueError):
        RepairAction.of(K.RAISE_RESOLUTION, dpi=0)
    with pytest.raises(ValueError):
        RepairAction.of(K.RAISE_RESOLUTION, dpi=True)
    with pytest.raises(ValueError):
        RepairAction(K.ADJUST_CROP, (("margin", 1), ("margin", 2)))
    assert RepairAction.of(K.TILE_DENSE_TABLE, tiles=2).param("tiles") == 2
    bad = {"arithmetic": (RepairAction.of(K.ADJUST_CROP, margin=0.01),)}
    with pytest.raises(ValueError, match="allowlisted"):
        machine(Executor(), ladder=bad)
    dpi = {"arithmetic": (RepairAction.of(K.RAISE_RESOLUTION, dpi=600),)}
    with pytest.raises(ValueError, match="DPI"):
        machine(Executor(), ladder=dpi)
    for check_id, actions in DEFAULT_LADDER.items():
        assert all(a.kind in ALLOWLIST[check_id] for a in actions) and len(actions) <= 3


# ---------------------------------------------------------------- audit
def test_audit_records_are_sanitized_deterministic_and_tenant_scoped():
    base: dict = {
        "run_id": "r",
        "region_id": "R1",
        "attempt_no": 1,
        "page": 1,
        "bbox": (0.1, 0.1, 0.5, 0.3),
        "config_fingerprint": "fp",
        "action_kind": "adjust_crop",
        "action_params": (("margin", 0.01),),
        "failed_checks": ("count",),
    }
    rec = AttemptRecord(**base, outcome="promoted", artifact_ref="artifacts/a1", duration_ms=5)
    assert (
        rec.to_json()
        == AttemptRecord(
            **base, outcome="promoted", artifact_ref="artifacts/a1", duration_ms=5
        ).to_json()
    )
    for patch in (
        {"artifact_ref": "https://u:p@host/x?token=1"},
        {"artifact_ref": "../../etc/passwd"},
        {"region_id": "has space and text"},
        {"failed_checks": ("a b",)},
        {"outcome": "weird"},
        {"attempt_no": 0},
        {"action_params": (("note", "free text value"),)},
    ):
        with pytest.raises(ValueError):
            AttemptRecord(**{**base, **patch})
    store = InMemoryAttemptStore()
    store.append("tenant-a", rec)
    assert store.list("tenant-b", "r") == [] and len(store.list("tenant-a", "r", "R1")) == 1
    with pytest.raises(ValueError, match="already"):
        store.append("tenant-a", rec)
    with pytest.raises(KeyError):
        store.update("tenant-a", AttemptRecord(**{**base, "attempt_no": 9}))
    with pytest.raises(ValueError):
        store.list("bad tenant!", "r")
    reader = AuditReader(store, {"alice": frozenset({"tenant-a"})})
    assert len(reader.read("alice", "tenant-a", "r")) == 1
    for who, tenant in (("alice", "tenant-b"), ("mallory", "tenant-a")):
        with pytest.raises(PermissionError):
            reader.read(who, tenant, "r")
    # same run ID in two tenants never interferes in the state machine
    machine(Executor(), store).run(tenant_id="tenant-b", run_id="r", regions=[ref()])
    assert len(store.list("tenant-a", "r")) == 1 and len(store.list("tenant-b", "r")) == 3


@given(
    st.lists(
        st.sampled_from(["count", "coverage", "shape", "arithmetic"]),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    st.integers(min_value=1, max_value=3),
)
def test_property_attempts_bounded_and_unique_for_any_failure_mix(checks, limit):
    ex = Executor()
    store = InMemoryAttemptStore()
    machine(ex, store, limits=RepairLimits(max_attempts_per_region=limit)).run(
        tenant_id="t", run_id="p", regions=[ref(checks=tuple(checks))]
    )
    records = store.list("t", "p")
    assert len(records) <= limit
    assert len({(r.action_kind, r.action_params) for r in records}) == len(records)
    for r in records:
        json.loads(r.to_json())


# ---------------------------------------------------------------- publication
def report_for(page, status="verified", checks=("completeness",), fp="f"):
    from ragnor.ingestion.verification import VerificationReport

    return VerificationReport(page, status, (), tuple(checks), "1", fp)


def test_strict_failure_keeps_previous_active_revision():
    store = InMemoryRevisionStore()
    ok = publish_verified(store, "doc", "rev1", [report_for(1), report_for(2)], expected_pages=2)
    assert ok.published and store.active("doc").revision_id == "rev1"
    bad = publish_verified(
        store, "doc", "rev2", [report_for(1), report_for(2, "needs_review")], expected_pages=2
    )
    assert not bad.published and bad.blocked_pages == (2,) and bad.reason == "page_needs_review"
    assert store.active("doc").revision_id == "rev1"
    failed = publish_verified(
        store,
        "doc",
        "rev3",
        [report_for(1, "failed"), report_for(2)],
        expected_pages=2,
        policy=PublishPolicy(strict=False),
    )
    assert not failed.published and store.active("doc").revision_id == "rev1"
    lax = publish_verified(
        store,
        "doc",
        "rev4",
        [report_for(1), report_for(2, "needs_review")],
        expected_pages=2,
        policy=PublishPolicy(strict=False),
    )
    assert lax.published and store.active("doc").pages == ((1, "verified"), (2, "needs_review"))


def test_publication_requires_complete_consistent_reports_and_required_checks():
    store = InMemoryRevisionStore()
    assert (
        publish_verified(store, "d", "r", [report_for(1)], expected_pages=2).reason
        == "incomplete_reports"
    )
    assert (
        publish_verified(store, "d", "r", [report_for(1), report_for(1)], expected_pages=2).reason
        == "incomplete_reports"
    )
    assert (
        publish_verified(
            store, "d", "r", [report_for(1, fp="a"), report_for(2, fp="b")], expected_pages=2
        ).reason
        == "mixed_verifier_settings"
    )
    skipped = publish_verified(store, "d", "r", [report_for(1, checks=())], expected_pages=1)
    assert skipped.reason == "required_check_skipped" and store.active("d") is None


def test_commit_failure_never_exposes_partial_publication():
    class Flaky(InMemoryRevisionStore):
        def commit(self, document_id, revision):
            raise OSError("disk full")

    store = Flaky()
    decision = publish_verified(store, "d", "r", [report_for(1)], expected_pages=1)
    assert decision.reason == "commit_failed" and store.active("d") is None


# ---------------------------------------------------------------- base install
def test_base_install_has_no_network_dependency():
    script = (
        "import builtins\n"
        "orig = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.split('.')[0] in {'litellm','httpx','requests','PIL','openai','mcp'}:\n"
        "        raise ImportError(name)\n"
        "    return orig(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "import ragnor.ingestion.verification as v\n"
        "assert v.PageVerificationPipeline\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert CheckUnavailableError


def test_ex02_corpus_false_positive_and_negative_rates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.pdf_extraction.verification_eval import evaluate

    result = evaluate()
    assert result["total"]["pages"] >= 25 and result["total"]["faulty_pages"] >= 20
    assert result["false_positive_rate"] == 0.0
    assert result["false_negative_rate"] == 0.0
    assert {"two_column", "ruled_table_invoice"} <= set(result["per_family"])
