"""Independent, cheap verification stages. Checks never modify extracted values."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Protocol

from ragnor.ingestion.extraction_types import (
    BBox,
    ExtractionTable,
    PageExtractionOutcome,
    checked_bbox,
)
from ragnor.ingestion.verification.geometry import covered_fraction, iou, region_ref
from ragnor.ingestion.verification.report import Finding, VerifierSettings

_CLASS = {"table": "table", "figure": "figure", "image": "figure", "heading": "heading"}
_DIGITS = {
    **{0x660 + i: ord("0") + i for i in range(10)},
    **{0x6F0 + i: ord("0") + i for i in range(10)},
}
_NUMBER = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def region_class(kind: str) -> str:
    return _CLASS.get(kind, "text")


@dataclass(frozen=True)
class ExpectedRegion:
    region_id: str
    kind: str
    bbox: BBox

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        if not self.region_id:
            raise ValueError("Expected regions need an ID")


@dataclass(frozen=True)
class ExpectedPage:
    """Independent expectation: the inventory plus foreground content boxes.

    ``content_boxes`` come from foreground-ink analysis with margins and
    decorations already excluded. Raw page area is never used as proof.
    """

    regions: tuple[ExpectedRegion, ...] = ()
    content_boxes: tuple[BBox, ...] = ()

    def __post_init__(self) -> None:
        for box in self.content_boxes:
            checked_bbox(box)


@dataclass(frozen=True)
class ExtractedRegion:
    region_id: str
    kind: str
    bbox: BBox


@dataclass(frozen=True)
class ArithmeticRelation:
    """An explicitly declared column relationship: sum of detail rows equals total row."""

    table_index: int
    column: int
    first_row: int
    last_row: int
    total_row: int


@dataclass(frozen=True)
class VerificationContext:
    outcome: PageExtractionOutcome
    expected: ExpectedPage | None = None
    alternate_tables: tuple[ExtractionTable, ...] | None = None
    relations: tuple[ArithmeticRelation, ...] = ()
    settings: VerifierSettings = field(default_factory=VerifierSettings)

    def extracted(self) -> tuple[ExtractedRegion, ...]:
        taken: set[str] = set()
        page = self.outcome.page
        items = [("table", t.bbox) for t in self.outcome.tables]
        items += [(r.kind, r.bbox) for r in self.outcome.regions]
        return tuple(ExtractedRegion(region_ref(page, k, b, taken), k, b) for k, b in items)

    def table_ids(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self.extracted()[: len(self.outcome.tables)])


class Check(Protocol):
    check_id: str

    def run(self, ctx: VerificationContext) -> list[Finding]:
        raise NotImplementedError


class CheckUnavailableError(RuntimeError):
    """Raised when a check lacks the independent input it needs."""


class CompletenessCheck:
    """Reuses EX-01's page status: an incomplete page can never verify."""

    check_id = "completeness"

    def run(self, ctx: VerificationContext) -> list[Finding]:
        if ctx.outcome.complete:
            return []
        return [
            Finding(
                ctx.outcome.page,
                f"page-{ctx.outcome.page}",
                self.check_id,
                "error",
                ctx.outcome.status.value,
                "complete",
                "extraction_status",
                "needs_review",
            )
        ]


class CountCheck:
    """Expected inventory versus extracted regions, including tables, figures, headers."""

    check_id = "count"

    def run(self, ctx: VerificationContext) -> list[Finding]:
        if ctx.expected is None:
            raise CheckUnavailableError(self.check_id)
        page = ctx.outcome.page
        extracted = ctx.extracted()
        findings: list[Finding] = []
        claimed: set[str] = set()
        for expected in sorted(ctx.expected.regions, key=lambda r: r.region_id):
            cls = region_class(expected.kind)
            best = max(
                (
                    e
                    for e in extracted
                    if region_class(e.kind) == cls and e.region_id not in claimed
                ),
                key=lambda e: iou(e.bbox, expected.bbox),
                default=None,
            )
            if best is not None and iou(best.bbox, expected.bbox) >= ctx.settings.match_iou:
                claimed.add(best.region_id)
                continue
            findings.append(
                Finding(
                    page,
                    expected.region_id,
                    self.check_id,
                    "error",
                    f"no {cls} extracted at expected location",
                    f"{cls} present",
                    "inventory",
                    "repair_candidate",
                )
            )
        findings.extend(
            Finding(
                page,
                e.region_id,
                self.check_id,
                "warning",
                f"unexpected {region_class(e.kind)}",
                "not in inventory",
                "inventory",
            )
            for e in extracted
            if e.region_id not in claimed
        )
        return findings


class CoverageCheck:
    """Each expected content box must be covered by extracted boxes, not by page area."""

    check_id = "coverage"

    def run(self, ctx: VerificationContext) -> list[Finding]:
        if ctx.expected is None:
            raise CheckUnavailableError(self.check_id)
        boxes = [e.bbox for e in ctx.extracted()]
        findings = []
        for index, content in enumerate(ctx.expected.content_boxes):
            fraction = covered_fraction(content, boxes)
            if fraction < ctx.settings.coverage_min:
                findings.append(
                    Finding(
                        ctx.outcome.page,
                        f"content-{index}",
                        self.check_id,
                        "error",
                        f"covered {fraction:.2f}",
                        f"at least {ctx.settings.coverage_min:.2f}",
                        "foreground",
                        "repair_candidate",
                    )
                )
        return findings


class ShapeCheck:
    """Compare table grids against an independent parser's result, when one exists."""

    check_id = "shape"

    def run(self, ctx: VerificationContext) -> list[Finding]:
        if ctx.alternate_tables is None:
            raise CheckUnavailableError(self.check_id)
        primary = list(ctx.outcome.tables)
        ids = ctx.table_ids()
        findings = []
        for index, table in enumerate(primary):
            twin = max(ctx.alternate_tables, key=lambda a: iou(a.bbox, table.bbox), default=None)
            if twin is None or iou(twin.bbox, table.bbox) < ctx.settings.match_iou:
                findings.append(self._finding(ctx, ids[index], "no independent table", "table"))
                continue
            spans = {(c.row, c.col, c.rowspan, c.colspan) for c in table.cells}
            alt_spans = {(c.row, c.col, c.rowspan, c.colspan) for c in twin.cells}
            if (table.rows, table.cols) != (twin.rows, twin.cols):
                findings.append(
                    self._finding(
                        ctx,
                        ids[index],
                        f"{table.rows}x{table.cols}",
                        f"{twin.rows}x{twin.cols}",
                    )
                )
            elif spans != alt_spans:
                findings.append(self._finding(ctx, ids[index], "merged spans differ", "same spans"))
        return findings

    def _finding(
        self, ctx: VerificationContext, region_id: str, observed: str, expected: str
    ) -> Finding:
        return Finding(
            ctx.outcome.page,
            region_id,
            self.check_id,
            "error",
            observed,
            expected,
            "alternate_parser",
            "repair_candidate",
        )


def parse_decimal(text: str) -> Decimal | None:
    """Decimal-safe amount parser. Unknown formats return ``None``, never a guess."""
    value = unicodedata.normalize("NFKC", text).translate(_DIGITS).strip()
    value = value.lstrip("$€£").strip() if value[:1] in "$€£" else value
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1].strip()
        value = value.lstrip("$€£").strip()
    if not _NUMBER.match(value):
        return None
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    return -number if negative else number


class ArithmeticCheck:
    """Totals on declared relationships only. Never autofixes; unknown means warning."""

    check_id = "arithmetic"

    def run(self, ctx: VerificationContext) -> list[Finding]:
        page = ctx.outcome.page
        ids = ctx.table_ids()
        findings: list[Finding] = []
        declared = {r.table_index for r in ctx.relations}
        for relation in ctx.relations:
            if not 0 <= relation.table_index < len(ctx.outcome.tables):
                findings.append(
                    Finding(
                        page,
                        f"table-{relation.table_index}",
                        self.check_id,
                        "error",
                        "table missing",
                        "table present",
                        "relation",
                        "repair_candidate",
                    )
                )
                continue
            findings.extend(self._relation(ctx, relation, ids[relation.table_index]))
        for index, table in enumerate(ctx.outcome.tables):
            if index not in declared and self._looks_like_total(table):
                findings.append(
                    Finding(
                        page,
                        ids[index],
                        self.check_id,
                        "warning",
                        "total row without a declared relationship",
                        "declared relation",
                        "heuristic",
                    )
                )
        return findings

    @staticmethod
    def _looks_like_total(table: ExtractionTable) -> bool:
        last = table.rows - 1
        return any(
            c.row == last and c.col == 0 and c.text.strip().lower().startswith("total")
            for c in table.cells
        )

    def _relation(
        self, ctx: VerificationContext, relation: ArithmeticRelation, region_id: str
    ) -> list[Finding]:
        table = ctx.outcome.tables[relation.table_index]
        page = ctx.outcome.page
        cells = {(c.row, c.col): c.text for c in table.cells}

        def finding(severity: str, observed: str, expected: str, disp: str) -> list[Finding]:
            return [
                Finding(
                    page, region_id, self.check_id, severity, observed, expected, "relation", disp
                )
            ]

        if (
            max(relation.last_row, relation.total_row) >= table.rows
            or relation.column >= table.cols
        ):
            return finding("error", "rows missing", "declared rows present", "repair_candidate")
        total = parse_decimal(cells.get((relation.total_row, relation.column), ""))
        parts = [
            parse_decimal(cells.get((r, relation.column), ""))
            for r in range(relation.first_row, relation.last_row + 1)
        ]
        if total is None or any(p is None for p in parts):
            return finding("warning", "unparseable amount", "decimal amount", "none")
        computed = sum((p for p in parts if p is not None), Decimal(0))
        if abs(computed - total) > ctx.settings.arithmetic_tolerance:
            return finding("error", f"sum {computed}", f"total {total}", "repair_candidate")
        return []


def default_checks() -> tuple[Check, ...]:
    return (CompletenessCheck(), CountCheck(), CoverageCheck(), ShapeCheck(), ArithmeticCheck())


__all__ = [
    "ArithmeticCheck",
    "ArithmeticRelation",
    "Check",
    "CheckUnavailableError",
    "CompletenessCheck",
    "CountCheck",
    "CoverageCheck",
    "ExpectedPage",
    "ExpectedRegion",
    "ExtractedRegion",
    "ShapeCheck",
    "VerificationContext",
    "default_checks",
    "parse_decimal",
    "region_class",
]
