"""Generate the synthetic ground-truth corpus (developer tool, not run in CI).

Builds every fixture PDF, the JSONL manifest and the probe set with a seeded
RNG. Annotations are emitted alongside the drawing commands so ground truth
matches the rendered page by construction; every page is then reviewed
against its rendered pixels (see docs/extraction-benchmarks.md).

Fonts: Helvetica core font for Latin text; DejaVu Sans (Bitstream Vera
license) and Noto Sans Arabic (SIL OFL 1.1) registered from the system font
directories and subset-embedded into the PDFs. Font files themselves are
not committed; regeneration requires the same system fonts.

Usage: ``python -m benchmarks.pdf_extraction.generate_fixtures`` (requires
reportlab and Pillow, generation-time only).
"""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas

from benchmarks.pdf_extraction.content import (
    ARABIC_HEADING,
    ARABIC_HEADING_2,
    ARABIC_NOTICE_PERIOD,
    ARABIC_PARA_1,
    ARABIC_PARA_2,
    ARABIC_PARA_3,
    ARABIC_PAYMENT_LINE,
    ARABIC_PROBE_QUERY,
    ARABIC_TOTAL_LINE,
    BROCHURE_COL_1,
    BROCHURE_COL_2,
    BROCHURE_COL_3,
    BROCHURE_COL_4,
    BROCHURE_COL_5,
    BROCHURE_COL_6,
    CLAUSE_GOVERNING,
    CLAUSE_LIABILITY,
    CLAUSE_TERMINATION,
    CLAUSE_WARRANTY,
    FINANCIAL_HEADER,
    FINANCIAL_ROWS,
    HANDWRITING_LINES,
    LEFT_COL_A,
    LEFT_COL_B,
    LEFT_COL_C,
    RIGHT_COL_A,
    RIGHT_COL_B,
    RIGHT_COL_C,
    SCAN_CONTRACT_P1,
    SCAN_CONTRACT_P2,
    SPLIT_ROWS,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOCUMENTS = FIXTURES / "documents"
CORPUS_VERSION = "1.0.0"
LICENSE = "synthetic-generated-cc0"
PAGE_W, PAGE_H = letter

ARABIC_FONT = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
DEJAVU_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class PageBuilder:
    """Draws one page and records its ground-truth annotation."""

    def __init__(self, canvas, document_id: str, page: int, page_count: int, family: str):
        self.canvas = canvas
        self.document_id = document_id
        self.page = page
        self.page_count = page_count
        self.family = family
        self.width = PAGE_W
        self.height = PAGE_H
        self.regions: list[dict] = []
        self.tables: list[dict] = []
        self.critical_fields: list[dict] = []
        self.notes = ""
        self.expect_error = False
        self._counter = 0

    def _region_id(self) -> str:
        self._counter += 1
        return f"r{self._counter}"

    def add_region(
        self,
        kind: str,
        bbox: list[float],
        order: int,
        expected_text: str = "",
        excluded: bool = False,
    ) -> str:
        """Annotate a region drawn outside the text/table helpers."""
        region_id = self._region_id()
        region = {
            "region_id": region_id,
            "kind": kind,
            "bbox": bbox,
            "reading_order": order,
            "expected_text": expected_text,
        }
        if excluded:
            region["excluded"] = True
        self.regions.append(region)
        return region_id

    def text(
        self,
        x: float,
        y_top: float,
        body: str,
        kind: str = "text",
        order: int = 0,
        font: str = "Helvetica",
        size: int = 10,
        bidi: bool = False,
    ) -> str:
        """Draw one line (baseline at y_top + size) and annotate the region."""
        drawn = _display_rtl(body) if bidi else body
        self.canvas.setFont(font, size)
        self.canvas.drawString(x, self.height - y_top - size, drawn)
        region_id = self._region_id()
        width = pdfmetrics.stringWidth(drawn, font, size)
        self.regions.append(
            {
                "region_id": region_id,
                "kind": kind,
                "bbox": [
                    round(x, 1),
                    round(y_top, 1),
                    round(x + width + 2, 1),
                    round(y_top + size + 3, 1),
                ],
                "reading_order": order,
                "expected_text": body,
            }
        )
        return region_id

    def paragraph(
        self,
        x: float,
        y_top: float,
        body: str,
        width_chars: int,
        kind: str = "text",
        order: int = 0,
        font: str = "Helvetica",
        size: int = 10,
        line_height: float | None = None,
        bidi: bool = False,
    ) -> tuple[str, float]:
        """Draw a wrapped paragraph; the region's expected text is the full body."""
        line_height = line_height or size * 1.5
        lines = textwrap.wrap(body, width_chars)
        self.canvas.setFont(font, size)
        for index, line in enumerate(lines):
            drawn = _display_rtl(line) if bidi else line
            self.canvas.drawString(x, self.height - y_top - size - index * line_height, drawn)
        region_id = self._region_id()
        box_height = (len(lines) - 1) * line_height + size + 3
        self.regions.append(
            {
                "region_id": region_id,
                "kind": kind,
                "bbox": [
                    round(x, 1),
                    round(y_top, 1),
                    round(x + 7.2 * width_chars, 1),
                    round(y_top + box_height, 1),
                ],
                "reading_order": order,
                "expected_text": " ".join(lines),
            }
        )
        return region_id, y_top + box_height + 6

    def table(
        self,
        x: float,
        y_top: float,
        col_widths: list[float],
        row_height: float,
        cells: list[dict],
        order: int,
        ruled: bool = True,
        size: int = 9,
        region_text: str = "",
    ) -> tuple[str, float]:
        """Draw a grid (optionally ruled) with spanned cells and annotate it."""
        rows = max(c["row"] + c.get("rowspan", 1) for c in cells)
        region_id = self._region_id()
        self._draw_table_grid(x, y_top, col_widths, row_height, rows, cells, ruled, size)
        total_height = row_height * rows
        total_width = sum(col_widths)
        self.regions.append(
            {
                "region_id": region_id,
                "kind": "table",
                "bbox": [
                    round(x, 1),
                    round(y_top, 1),
                    round(x + total_width, 1),
                    round(y_top + total_height, 1),
                ],
                "reading_order": order,
                "expected_text": region_text,
            }
        )
        self.tables.append(
            {"region_id": region_id, "rows": rows, "cols": len(col_widths), "cells": cells}
        )
        return region_id, y_top + total_height + 8

    def _draw_rules(
        self,
        x: float,
        y_top: float,
        col_widths: list[float],
        row_height: float,
        rows: int,
        cells: list[dict],
    ) -> None:
        """Ruled borders that respect merged cells (no line through a span)."""
        total_width = sum(col_widths)
        total_height = row_height * rows
        self.canvas.rect(x, self.height - y_top - total_height, total_width, total_height)
        for index in range(1, rows):
            self._horizontal_rule(x, y_top, col_widths, row_height, index, cells)
        boundaries = [x]
        for width in col_widths[:-1]:
            boundaries.append(boundaries[-1] + width)
        for col_index, boundary in enumerate(boundaries[1:], start=1):
            self._vertical_rule(boundary, y_top, row_height, rows, col_index, cells)

    def _horizontal_rule(
        self,
        x: float,
        y_top: float,
        col_widths: list[float],
        row_height: int,
        index: int,
        cells: list[dict],
    ) -> None:
        """Rule between rows ``index - 1`` and ``index``, skipping column ranges
        covered by a cell spanning across the boundary."""
        y = self.height - y_top - index * row_height
        edges = [x]
        for width in col_widths:
            edges.append(edges[-1] + width)
        covered_cols = set()
        for cell in cells:
            span = cell.get("rowspan", 1)
            if cell["row"] < index < cell["row"] + span:
                for col in range(cell["col"], cell["col"] + cell.get("colspan", 1)):
                    covered_cols.add(col)
        for col in range(len(col_widths)):
            if col not in covered_cols:
                self.canvas.line(edges[col], y, edges[col + 1], y)

    def _vertical_rule(
        self,
        boundary: float,
        y_top: float,
        row_height: float,
        rows: int,
        col_index: int,
        cells: list[dict],
    ) -> None:
        for row in range(rows):
            covered = any(
                c["row"] <= row < c["row"] + c.get("rowspan", 1)
                and c["col"] < col_index < c["col"] + c.get("colspan", 1)
                for c in cells
            )
            if not covered:
                y1 = self.height - y_top - row * row_height
                self.canvas.line(boundary, y1, boundary, y1 - row_height)

    def _draw_table_grid(
        self,
        x: float,
        y_top: float,
        col_widths: list[float],
        row_height: float,
        rows: int,
        cells: list[dict],
        ruled: bool,
        size: int,
    ) -> None:
        self.canvas.setFont("Helvetica", size)
        if ruled:
            self._draw_rules(x, y_top, col_widths, row_height, rows, cells)
        for cell in cells:
            cell_x = x + sum(col_widths[: cell["col"]])
            cell_y = y_top + cell["row"] * row_height + (row_height - size) / 2
            self.canvas.drawString(cell_x + 3, self.height - cell_y - size, cell["text"])

    def field(self, name: str, value: str, numeric: bool = False) -> None:
        self.critical_fields.append({"name": name, "value": value, "numeric": numeric})

    def finish_page(self) -> dict:
        self.canvas.showPage()
        return self

    def annotation(self, document: Document) -> dict:
        record = {
            "document_id": self.document_id,
            "file": f"documents/{document.filename}",
            "family": self.family,
            "page": self.page,
            "page_count": self.page_count,
            "width": self.width,
            "height": self.height,
            "license": LICENSE,
            "source_hash": document.source_hash,
            "regions": self.regions,
            "tables": self.tables,
            "critical_fields": self.critical_fields,
        }
        if self.expect_error:
            record["expect_error"] = True
        if self.notes:
            record["notes"] = self.notes
        return record


class Document:
    def __init__(self, filename: str):
        self.filename = filename
        self.source_hash = ""

    def render_with(self, builder) -> list[PageBuilder]:
        path = DOCUMENTS / self.filename
        canvas = pdf_canvas.Canvas(str(path), pagesize=letter)
        pages = builder(canvas)
        canvas.save()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.source_hash = f"sha256:{digest}"
        return pages


def _register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("NotoSansArabic", ARABIC_FONT))
    pdfmetrics.registerFont(TTFont("DejaVuSans", DEJAVU_FONT))


def _display_rtl(logical: str) -> str:
    """Shape and bidi-reorder logical Arabic for drawing (generation-time deps).

    The stored ground truth stays the logical-order Unicode string; NFKC
    normalization recovers base letters from the shaped presentation forms
    at scoring time. Digit-run ordering inside RTL lines is a genuinely hard
    extraction case and is measured, not hidden.
    """
    try:
        import arabic_reshaper  # noqa: PLC0415
        from bidi.algorithm import get_display  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Arabic fixture generation requires arabic-reshaper and python-bidi"
        ) from exc
    return get_display(arabic_reshaper.reshape(logical))


# --- document builders -----------------------------------------------------
# Each builder draws pages through PageBuilder, which records the annotation
# for everything it draws. Text content is fixed (no RNG in content) so the
# corpus is stable; the seeded RNG only perturbs scan rendering.


def build_plain_digital(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "plain-digital-report", 1, 2, "plain_digital")
    p.text(72, 72, "Quarterly Operations Report", kind="heading", order=0, size=16)
    p.paragraph(
        72,
        110,
        "Revenue grew twelve percent year over year while operating costs stayed flat. "
        "The logistics division shipped 18420 orders across nine regional warehouses.",
        66,
        order=1,
    )
    p.paragraph(
        72,
        200,
        "Customer satisfaction averaged 4.6 out of 5 across surveyed accounts. "
        "Renewal rates improved in every segment except retail partnerships.",
        66,
        order=2,
    )
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "plain-digital-report", 2, 2, "plain_digital")
    p2.text(72, 72, "Outlook and Hiring Plan", kind="heading", order=0, size=16)
    p2.paragraph(
        72,
        110,
        "The company plans to hire fourteen engineers and six analysts next quarter. "
        "Capital expenditure remains focused on warehouse automation projects.",
        66,
        order=1,
    )
    pages.append(p2.finish_page())
    return pages


def build_two_column(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "two-column-newsletter", 1, 2, "two_column")
    p.text(72, 60, "Community Newsletter Issue 14", kind="heading", order=0, size=14)
    p.paragraph(72, 96, LEFT_COL_A, 34, order=1)
    p.paragraph(72, 330, LEFT_COL_B, 34, order=2)
    p.paragraph(330, 96, RIGHT_COL_A, 34, order=3)
    p.paragraph(330, 330, RIGHT_COL_B, 34, order=4)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "two-column-newsletter", 2, 2, "two_column")
    p2.text(72, 60, "Classifieds and Notices", kind="heading", order=0, size=14)
    p2.paragraph(72, 96, LEFT_COL_C, 34, order=1)
    p2.paragraph(330, 96, RIGHT_COL_C, 34, order=2)
    pages.append(p2.finish_page())
    return pages


def build_three_column(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "three-column-brochure", 1, 2, "three_column")
    p.paragraph(50, 80, BROCHURE_COL_1, 22, order=0)
    p.paragraph(226, 80, BROCHURE_COL_2, 22, order=1)
    p.paragraph(402, 80, BROCHURE_COL_3, 22, order=2)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "three-column-brochure", 2, 2, "three_column")
    p2.text(50, 60, "Pricing Tiers", kind="heading", order=0, size=14)
    p2.paragraph(50, 96, BROCHURE_COL_4, 22, order=1)
    p2.paragraph(226, 96, BROCHURE_COL_5, 22, order=2)
    p2.paragraph(402, 96, BROCHURE_COL_6, 22, order=3)
    pages.append(p2.finish_page())
    return pages


def build_scanned_contract(canvas) -> list[PageBuilder]:
    pages = []
    for page, lines in ((1, SCAN_CONTRACT_P1), (2, SCAN_CONTRACT_P2)):
        p = PageBuilder(canvas, "scanned-contract", page, 2, "scan")
        image = render_scan_image(lines, seed=40 + page, size=13)
        _draw_full_page_image(canvas, image)
        for order, line in enumerate(lines):
            p.add_region("text", [72, 90 + order * 24, 540, 90 + order * 24 + 16], order, line)
        p.notes = "image-only scan; a text-only extractor must report empty, not success"
        pages.append(p.finish_page())
    return pages


def build_mixed_memo(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "mixed-memo", 1, 2, "mixed")
    p.text(72, 72, "Internal Memo: Office Relocation", kind="heading", order=0, size=14)
    p.paragraph(
        72,
        108,
        "The downtown office moves to the harbor district on the first of next month. "
        "All badge access transfers automatically during the migration weekend.",
        66,
        order=1,
    )
    _draw_chart(canvas, 72, 260)
    p.add_region("figure", [72, 260, 320, 420], 0, excluded=True)
    p.text(72, 428, "Figure 1: monthly desk occupancy before relocation", kind="caption", order=2)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "mixed-memo", 2, 2, "mixed")
    lines = ["Memo attachment scanned from paper copy.", "Signed by facilities management."]
    image = render_scan_image(lines, seed=52, size=14)
    _draw_full_page_image(canvas, image)
    for order, line in enumerate(lines):
        p2.add_region("text", [72, 90 + order * 26, 540, 90 + order * 26 + 18], order, line)
    p2.notes = "second page is image-only"
    pages.append(p2.finish_page())
    return pages


def build_rotated(canvas) -> list[PageBuilder]:
    p = PageBuilder(canvas, "rotated-notice", 1, 1, "rotated")
    canvas.setPageRotation(90)
    # Pre-rotate the drawing so it displays upright after the viewer applies
    # the /Rotate 90 attribute; display y grows downward as negative user y.
    canvas.saveState()
    canvas.translate(PAGE_W, 0)
    canvas.rotate(90)
    lines = [
        ("Rotated Safety Notice", 16, "heading"),
        (
            "This page carries a ninety degree rotation attribute. Extraction must "
            "still recover the",
            10,
            "text",
        ),
        (
            "evacuation assembly point GATE 4 and the emergency contact extension 4820.",
            10,
            "text",
        ),
    ]
    y_top = 72
    for order, (body, size, kind) in enumerate(lines):
        canvas.setFont("Helvetica", size)
        canvas.drawString(72, -(y_top + size), body)
        width = pdfmetrics.stringWidth(body, "Helvetica", size)
        p.add_region(kind, [72, y_top, 72 + width + 2, y_top + size + 3], order, body)
        y_top += size + 14
    p.field("assembly_point", "GATE 4")
    p.field("emergency_extension", "4820", numeric=True)
    p.notes = "page carries /Rotate 90; text drawn pre-rotated to display upright"
    canvas.restoreState()
    return [p.finish_page()]


def build_ruled_invoice(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "ruled-invoice", 1, 2, "ruled_table_invoice")
    p.text(72, 72, "INVOICE INV-2026-0417", kind="heading", order=0, size=16)
    p.text(72, 100, "Bill To: Nile Trading Company", order=1)
    p.text(72, 118, "Invoice Date: 2026-10-05", order=2)
    cells = [
        {"row": 0, "col": 0, "text": "Description", "colspan": 2},
        {"row": 0, "col": 2, "text": "Qty"},
        {"row": 0, "col": 3, "text": "Amount"},
        {"row": 1, "col": 0, "text": "Consulting services", "colspan": 2},
        {"row": 1, "col": 2, "text": "12"},
        {"row": 1, "col": 3, "text": "960.00"},
        {"row": 2, "col": 0, "text": "Support retainer", "colspan": 2},
        {"row": 2, "col": 2, "text": "1"},
        {"row": 2, "col": 3, "text": "290.00"},
        {"row": 3, "col": 0, "text": "Total due", "colspan": 2},
        {"row": 3, "col": 2, "text": ""},
        {"row": 3, "col": 3, "text": "1,250.00"},
    ]
    p.table(
        72,
        150,
        [180, 80, 60, 100],
        22,
        cells,
        order=3,
        region_text="Description Qty Amount Consulting services 12 960.00 "
        "Support retainer 1 290.00 Total due 1,250.00",
    )
    p.field("invoice_number", "INV-2026-0417")
    p.field("invoice_total", "1,250.00", numeric=True)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "ruled-invoice", 2, 2, "ruled_table_invoice")
    p2.text(72, 72, "Payment Schedule", kind="heading", order=0, size=14)
    cells2 = [
        {"row": 0, "col": 0, "text": "Installment"},
        {"row": 0, "col": 1, "text": "Due date"},
        {"row": 0, "col": 2, "text": "Amount"},
        {"row": 1, "col": 0, "text": "First", "rowspan": 2},
        {"row": 1, "col": 1, "text": "2026-11-01"},
        {"row": 1, "col": 2, "text": "625.00"},
        {"row": 2, "col": 1, "text": "2026-12-01"},
        {"row": 2, "col": 2, "text": "625.00"},
    ]
    p2.table(
        72,
        110,
        [120, 140, 120],
        22,
        cells2,
        order=1,
        region_text="Installment Due date Amount First 2026-11-01 625.00 2026-12-01 625.00",
    )
    pages.append(p2.finish_page())
    return pages


def build_borderless_financials(canvas) -> list[PageBuilder]:
    pages = []
    rows_data = FINANCIAL_ROWS
    for page_index in (0, 1):
        p = PageBuilder(canvas, "borderless-financials", page_index + 1, 2, "borderless_table")
        p.text(
            72,
            64,
            f"Consolidated Financials FY2026 (page {page_index + 1})",
            kind="heading",
            order=0,
            size=13,
        )
        chunk = rows_data[page_index * 8 : page_index * 8 + 8]
        cells = [{"row": 0, "col": col, "text": head} for col, head in enumerate(FINANCIAL_HEADER)]
        for row_index, row in enumerate(chunk, start=1):
            for col, value in enumerate(row):
                cells.append({"row": row_index, "col": col, "text": value})
        region_text = (
            " ".join(FINANCIAL_HEADER) + " " + " ".join(value for row in chunk for value in row)
        )
        p.table(
            72,
            96,
            [110, 70, 70, 70, 70, 70],
            20,
            cells,
            order=1,
            ruled=False,
            region_text=region_text,
        )
        if page_index == 0:
            p.field("digital_q4_revenue", "612,900", numeric=True)
        pages.append(p.finish_page())
    return pages


def build_split_table(canvas) -> list[PageBuilder]:
    pages = []
    header = ["SKU", "Item", "Warehouse", "On hand"]
    first = SPLIT_ROWS[:6]
    second = SPLIT_ROWS[6:]
    p = PageBuilder(canvas, "split-table", 1, 2, "split_merged_table")
    p.text(72, 64, "Inventory Snapshot", kind="heading", order=0, size=13)
    cells = [{"row": 0, "col": c, "text": h} for c, h in enumerate(header)]
    for row_index, row in enumerate(first, start=1):
        for col, value in enumerate(row):
            cells.append({"row": row_index, "col": col, "text": value})
    p.table(
        72,
        96,
        [70, 180, 120, 80],
        20,
        cells,
        order=1,
        region_text=" ".join(header) + " " + " ".join(v for r in first for v in r),
    )
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "split-table", 2, 2, "split_merged_table")
    p2.text(72, 64, "Inventory Snapshot (continued)", kind="heading", order=0, size=13)
    cells2 = [{"row": 0, "col": c, "text": h} for c, h in enumerate(header)]
    for row_index, row in enumerate(second, start=1):
        for col, value in enumerate(row):
            cells2.append({"row": row_index, "col": col, "text": value})
    cells2.append({"row": len(second) + 1, "col": 0, "text": "Totals"})
    cells2.append({"row": len(second) + 1, "col": 1, "text": "All warehouses", "colspan": 2})
    cells2.append({"row": len(second) + 1, "col": 3, "text": "731"})
    p2.table(
        72,
        96,
        [70, 180, 120, 80],
        20,
        cells2,
        order=1,
        region_text=" ".join(header)
        + " "
        + " ".join(v for r in second for v in r)
        + " Totals All warehouses 731",
    )
    pages.append(p2.finish_page())
    return pages


def build_headings_manual(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "headings-manual", 1, 2, "multilevel_headings")
    p.text(72, 60, "Chapter 1: Installation", kind="heading", order=0, size=18)
    p.text(72, 96, "1.1 System Requirements", kind="heading", order=1, size=14)
    p.paragraph(
        72,
        126,
        "The agent requires Python 3.11 or newer and two gigabytes of "
        "free memory for the extraction workers.",
        66,
        order=2,
    )
    p.text(72, 190, "1.1.1 Supported Platforms", kind="heading", order=3, size=11)
    p.paragraph(
        72,
        216,
        "Linux and macOS are fully supported. Windows deployments use "
        "the provided container images exclusively.",
        66,
        order=4,
    )
    p.text(72, 280, "1.2 Quick Start", kind="heading", order=5, size=14)
    p.paragraph(
        72,
        310,
        "Install the package, export the API key, then run the smoke "
        "command to verify the local pipeline end to end.",
        66,
        order=6,
    )
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "headings-manual", 2, 2, "multilevel_headings")
    p2.text(72, 60, "Chapter 2: Configuration", kind="heading", order=0, size=18)
    p2.text(72, 96, "2.1 Storage Backends", kind="heading", order=1, size=14)
    p2.paragraph(
        72,
        126,
        "Snapshots persist through the host database adapter. "
        "SQLite suits development; PostgreSQL is required in production.",
        66,
        order=2,
    )
    p2.text(72, 190, "2.1.1 Connection Pooling", kind="heading", order=3, size=11)
    p2.paragraph(
        72,
        216,
        "Pool size defaults to four connections per worker and must "
        "remain below the database connection limit.",
        66,
        order=4,
    )
    pages.append(p2.finish_page())
    return pages


def build_arabic_rtl(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "arabic-rtl-notice", 1, 2, "arabic_rtl")
    p.text(
        72, 72, ARABIC_HEADING, kind="heading", order=0, font="NotoSansArabic", size=16, bidi=True
    )
    p.paragraph(
        72,
        120,
        ARABIC_PARA_1,
        40,
        order=1,
        font="NotoSansArabic",
        size=12,
        line_height=22,
        bidi=True,
    )
    p.paragraph(
        72,
        260,
        ARABIC_PARA_2,
        40,
        order=2,
        font="NotoSansArabic",
        size=12,
        line_height=22,
        bidi=True,
    )
    p.field("notice_period", ARABIC_NOTICE_PERIOD)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "arabic-rtl-notice", 2, 2, "arabic_rtl")
    p2.text(
        72, 72, ARABIC_HEADING_2, kind="heading", order=0, font="NotoSansArabic", size=16, bidi=True
    )
    p2.paragraph(
        72,
        120,
        ARABIC_PARA_3,
        40,
        order=1,
        font="NotoSansArabic",
        size=12,
        line_height=22,
        bidi=True,
    )
    p2.notes = (
        "Arabic is shaped and bidi-reordered at draw time; expected_text is the "
        "logical-order Unicode string recovered through NFKC normalization."
    )
    pages.append(p2.finish_page())
    return pages


def build_multilingual(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "multilingual-invoice", 1, 2, "multilingual")
    p.text(72, 72, "Statement of Charges / Releve de Frais", kind="heading", order=0, size=14)
    p.text(72, 110, "Service period: October 2026", order=1)
    p.text(72, 130, "Montant total: 2 480,75 EUR", order=2)
    p.text(72, 150, ARABIC_TOTAL_LINE, order=3, font="NotoSansArabic", size=12, bidi=True)
    p.field("eur_total", "2 480,75")
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "multilingual-invoice", 2, 2, "multilingual")
    p2.text(
        72, 72, "Payment Instructions / Instructions de Paiement", kind="heading", order=0, size=14
    )
    p2.text(
        72, 110, "Wire within thirty days to IBAN EG38 0021 0005 0000 0000 1234 5678 9", order=1
    )
    p2.text(72, 130, "Reference obligatoire: FAC-2026-118", order=2)
    p2.text(72, 150, ARABIC_PAYMENT_LINE, order=3, font="NotoSansArabic", size=12, bidi=True)
    p2.field("payment_reference", "FAC-2026-118")
    pages.append(p2.finish_page())
    return pages


def build_contract_clauses(canvas) -> list[PageBuilder]:
    pages = []
    p = PageBuilder(canvas, "contract-clauses", 1, 2, "contract")
    p.text(72, 72, "Master Services Agreement", kind="heading", order=0, size=15)
    p.paragraph(72, 110, CLAUSE_WARRANTY, 70, order=1)
    p.paragraph(72, 220, CLAUSE_LIABILITY, 70, order=2)
    pages.append(p.finish_page())
    p2 = PageBuilder(canvas, "contract-clauses", 2, 2, "contract")
    p2.paragraph(72, 72, CLAUSE_TERMINATION, 70, order=0)
    p2.paragraph(72, 190, CLAUSE_GOVERNING, 70, order=1)
    pages.append(p2.finish_page())
    return pages


def build_chart_report(canvas) -> list[PageBuilder]:
    p = PageBuilder(canvas, "chart-report", 1, 1, "chart")
    p.text(72, 72, "Quarterly Growth Dashboard", kind="heading", order=0, size=15)
    p.paragraph(
        72,
        108,
        "Signups accelerated through October while churn stayed below "
        "two percent for the third consecutive quarter.",
        66,
        order=1,
    )
    _draw_chart(canvas, 72, 200)
    p.add_region("figure", [72, 200, 360, 380], 0, excluded=True)
    p.text(72, 388, "Figure 1: weekly active workspaces by cohort", kind="caption", order=2)
    p.paragraph(
        72,
        420,
        "The October cohort retained eighty percent of its initial "
        "workspaces after four weeks of onboarding.",
        66,
        order=3,
    )
    return [p.finish_page()]


def build_formula_sheet(canvas) -> list[PageBuilder]:
    p = PageBuilder(canvas, "formula-sheet", 1, 1, "formula")
    p.text(72, 72, "Useful Identities", kind="heading", order=0, size=15)
    p.text(72, 110, "E = m c^2", kind="formula", order=1, font="DejaVuSans", size=12)
    p.text(
        72,
        140,
        "x = (-b +- sqrt(b^2 - 4ac)) / 2a",
        kind="formula",
        order=2,
        font="DejaVuSans",
        size=12,
    )
    p.text(
        72,
        170,
        "variance = sum((x - mean)^2) / (n - 1)",
        kind="formula",
        order=3,
        font="DejaVuSans",
        size=12,
    )
    p.text(
        72,
        200,
        "gross_margin = (revenue - cogs) / revenue",
        kind="formula",
        order=4,
        font="DejaVuSans",
        size=12,
    )
    return [p.finish_page()]


def build_handwriting(canvas) -> list[PageBuilder]:
    p = PageBuilder(canvas, "handwriting-note", 1, 1, "handwriting")
    lines = HANDWRITING_LINES
    image = render_scan_image(lines, seed=77, size=17, jitter=3.0, italic=True)
    _draw_full_page_image(canvas, image)
    for order, line in enumerate(lines):
        p.add_region("text", [72, 90 + order * 30, 540, 90 + order * 30 + 20], order, line)
    p.notes = "image-only handwriting sample; no text layer exists"
    return [p.finish_page()]


def build_corrupted(canvas) -> list[PageBuilder]:
    # The placeholder page is replaced on disk by truncated bytes afterwards.
    p = PageBuilder(canvas, "corrupted-report", 1, 1, "corrupted")
    p.expect_error = True
    p.notes = "file bytes are truncated at 55% after generation; parsers must fail"
    return [p.finish_page()]


def build_oversized(canvas) -> list[PageBuilder]:
    p = PageBuilder(canvas, "oversized-blank", 1, 1, "oversized")
    p.expect_error = True
    p.notes = "1001 blank pages exceed the pinned max_pages=1000 bound"
    for _ in range(1000):
        canvas.showPage()
    return [p.finish_page()]


# --- helpers ----------------------------------------------------------------


def render_scan_image(
    lines: list[str], seed: int, size: int, jitter: float = 0.0, italic: bool = False
):
    """Render text lines to a slightly rotated grayscale image (a fake scan)."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    rng = random.Random(seed)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf" if italic else DEJAVU_FONT
    font = ImageFont.truetype(font_path, size)
    image = Image.new("L", (int(PAGE_W), int(PAGE_H)), color=245)
    draw = ImageDraw.Draw(image)
    y = 90
    for line in lines:
        x = 72 + rng.uniform(-jitter, jitter)
        draw.text((x, y + rng.uniform(-jitter, jitter)), line, fill=20, font=font)
        y += int(size * (2.0 if italic else 1.9))
    for _ in range(400):
        image.putpixel(
            (rng.randrange(image.width), rng.randrange(image.height)), rng.randrange(200, 245)
        )
    return image.rotate(rng.uniform(-0.5, 0.5), fillcolor=245)


def _draw_full_page_image(canvas, image) -> None:
    from reportlab.lib.utils import ImageReader  # noqa: PLC0415

    canvas.drawImage(ImageReader(image), 0, 0, width=PAGE_W, height=PAGE_H)


def _draw_chart(canvas, x: float, y_top: float) -> None:
    """Draw a small bar chart (pure vector, no text) as a figure region."""
    base = PAGE_H - y_top - 140
    canvas.line(x, base, x + 240, base)
    canvas.line(x, base, x, base + 140)
    for index, height in enumerate((40, 75, 60, 110, 90, 130)):
        canvas.rect(x + 12 + index * 38, base, 24, height, stroke=1, fill=0)


DOCUMENT_BUILDERS = {
    "plain-digital-report.pdf": build_plain_digital,
    "two-column-newsletter.pdf": build_two_column,
    "three-column-brochure.pdf": build_three_column,
    "scanned-contract.pdf": build_scanned_contract,
    "mixed-memo.pdf": build_mixed_memo,
    "rotated-notice.pdf": build_rotated,
    "ruled-invoice.pdf": build_ruled_invoice,
    "borderless-financials.pdf": build_borderless_financials,
    "split-table.pdf": build_split_table,
    "headings-manual.pdf": build_headings_manual,
    "arabic-rtl-notice.pdf": build_arabic_rtl,
    "multilingual-invoice.pdf": build_multilingual,
    "contract-clauses.pdf": build_contract_clauses,
    "chart-report.pdf": build_chart_report,
    "formula-sheet.pdf": build_formula_sheet,
    "handwriting-note.pdf": build_handwriting,
    "corrupted-report.pdf": build_corrupted,
    "oversized-blank.pdf": build_oversized,
}


def main() -> None:
    _register_fonts()
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, builder in DOCUMENT_BUILDERS.items():
        document = Document(filename)
        pages = document.render_with(builder)
        records.extend(page.annotation(document) for page in pages)
    _corrupt(Path(DOCUMENTS) / "corrupted-report.pdf", records)
    _write_manifest(records)
    _write_probes()
    print(f"wrote {len(records)} annotated pages across {len(DOCUMENT_BUILDERS)} documents")


def _corrupt(path: Path, records: list[dict]) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * 0.55)])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for record in records:
        if record["document_id"] == "corrupted-report":
            record["source_hash"] = f"sha256:{digest}"


def _write_manifest(records: list[dict]) -> None:
    header = {"corpus_version": CORPUS_VERSION, "coordinate_space": "pdf-points-top-left"}
    lines = [json.dumps(header, sort_keys=True)]
    lines.extend(json.dumps(record, sort_keys=True) for record in records)
    (FIXTURES / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_probes() -> None:
    probes = {
        "version": CORPUS_VERSION,
        "probes": [
            {
                "id": "invoice-total",
                "query": "invoice total amount due",
                "document_id": "ruled-invoice",
                "page": 1,
                "region_id": "r4",
            },
            {
                "id": "q4-revenue",
                "query": "digital segment Q4 612,900 table value",
                "document_id": "borderless-financials",
                "page": 1,
                "region_id": "r2",
            },
            {
                "id": "split-totals",
                "query": "inventory totals all warehouses on hand",
                "document_id": "split-table",
                "page": 2,
                "region_id": "r2",
            },
            {
                "id": "warranty-clause",
                "query": "warranty professional manner industry standards implied",
                "document_id": "contract-clauses",
                "page": 1,
                "region_id": "r2",
            },
            {
                "id": "quadratic",
                "query": "solve x sqrt(b^2 - 4ac) equation roots",
                "document_id": "formula-sheet",
                "page": 1,
                "region_id": "r3",
            },
            {
                "id": "chart-caption",
                "query": "weekly active workspaces by cohort",
                "document_id": "chart-report",
                "page": 1,
                "region_id": "r4",
            },
            {
                "id": "arabic-notice",
                "query": ARABIC_PROBE_QUERY,
                "document_id": "arabic-rtl-notice",
                "page": 1,
                "region_id": "r2",
            },
            {
                "id": "newsletter-right",
                "query": "volunteer schedule park cleanup weekday shift",
                "document_id": "two-column-newsletter",
                "page": 1,
                "region_id": "r4",
            },
        ],
    }
    (FIXTURES / "probes.json").write_text(
        json.dumps(probes, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
