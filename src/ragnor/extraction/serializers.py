"""Serializers: lossless JSON and HTML, lossy text/Markdown projections.

Untrusted cell and region text is always escaped before it appears in HTML,
and every table projection is bounded; over-budget tables raise
:class:`ExtractionLimitError` instead of being truncated silently.
"""

from __future__ import annotations

import html
import json
from typing import Any

from ragnor.extraction.errors import ExtractionError, ExtractionLimitError
from ragnor.extraction.types import (
    DerivedKind,
    ExtractionResult,
    Region,
    RegionKind,
    Table,
)

MAX_JSON_BYTES = 64_000_000
DEFAULT_MAX_CELLS = 10_000
DEFAULT_MAX_CHARS = 200_000
FURNITURE = frozenset({RegionKind.HEADER, RegionKind.FOOTER, RegionKind.FOOTNOTE})


def to_json(result: ExtractionResult) -> str:
    """Canonical, stable JSON for a result (sorted keys, no NaN)."""
    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def from_json(payload: str) -> ExtractionResult:
    if not isinstance(payload, str) or len(payload.encode()) > MAX_JSON_BYTES:
        raise ExtractionLimitError
    try:
        data: Any = json.loads(payload)
        return ExtractionResult.from_dict(data)
    except ExtractionError:
        raise
    except (ValueError, KeyError, TypeError):
        raise ExtractionError("invalid_document") from None


def _budget(table: Table, max_cells: int) -> None:
    if table.rows * table.cols > max_cells:
        raise ExtractionLimitError


def table_to_html(
    table: Table, *, max_cells: int = DEFAULT_MAX_CELLS, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Faithful HTML: spans kept, values escaped, missing/absent cells marked."""
    _budget(table, max_cells)
    anchors = {(c.row, c.col): c for c in table.cells}
    covered = table.occupancy()
    out = ["<table>"]
    size = 7
    for r in range(table.rows):
        row = ["<tr>"]
        for c in range(table.cols):
            cell = anchors.get((r, c))
            if cell is not None:
                tag = "th" if cell.header_scope is not None else "td"
                attrs = ""
                if cell.header_scope is not None:
                    attrs += f' scope="{cell.header_scope.value}"'
                if cell.rowspan > 1:
                    attrs += f' rowspan="{cell.rowspan}"'
                if cell.colspan > 1:
                    attrs += f' colspan="{cell.colspan}"'
                if cell.text is None:
                    attrs += ' data-missing="true"'
                row.append(f"<{tag}{attrs}>{html.escape(cell.text or '')}</{tag}>")
            elif (r, c) not in covered:
                row.append('<td data-absent="true"></td>')
        row.append("</tr>")
        piece = "".join(row)
        size += len(piece)
        if size > max_chars:
            raise ExtractionLimitError
        out.append(piece)
    out.append("</table>")
    return "".join(out)


def table_to_markdown(table: Table, *, max_cells: int = DEFAULT_MAX_CELLS) -> str | None:
    """Markdown only when it is lossless, otherwise ``None`` (use HTML).

    Lossless means: a rectangular grid, no spans, no missing or absent cells,
    a column-header first row, and no value that Markdown cannot carry exactly.
    """
    _budget(table, max_cells)
    if table.uncovered_positions() or table.rows < 1:
        return None
    grid = table.occupancy()
    for cell in table.cells:
        if cell.rowspan != 1 or cell.colspan != 1 or cell.text is None:
            return None
        if any(ch in cell.text for ch in "\n\r<>") or cell.text != cell.text.strip():
            return None
    if any(grid[(0, c)].header_scope is None for c in range(table.cols)):
        return None

    def esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|")

    lines = []
    for r in range(table.rows):
        lines.append(
            "| " + " | ".join(esc(grid[(r, c)].text or "") for c in range(table.cols)) + " |"
        )
        if r == 0:
            lines.append("|" + "|".join([" --- "] * table.cols) + "|")
    return "\n".join(lines)


def table_to_text(
    table: Table, *, max_cells: int = DEFAULT_MAX_CELLS, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Retrieval projection: one line per row, cells joined by `` | ``.

    Values are exact; missing/absent cells project as empty, never as a guess.
    A spanning cell appears once, at its anchor.
    """
    _budget(table, max_cells)
    anchors = {(c.row, c.col): c for c in table.cells}
    covered = table.occupancy()
    lines = []
    total = 0
    for r in range(table.rows):
        parts = []
        for c in range(table.cols):
            cell = anchors.get((r, c))
            if cell is not None:
                parts.append(cell.text or "")
            elif (r, c) not in covered:
                parts.append("")
        line = " | ".join(parts)
        total += len(line) + 1
        if total > max_chars:
            raise ExtractionLimitError
        lines.append(line)
    return "\n".join(lines)


def region_text(
    region: Region,
    *,
    include_derived: bool = True,
    include_estimates: bool = False,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Plain-text projection of one region for embedding and search.

    Model-derived content is labelled and never presented as verified source
    text. Chart estimates are excluded unless explicitly requested.
    """
    parts: list[str] = []
    if region.table is not None:
        parts.append(table_to_text(region.table, max_cells=max_cells, max_chars=max_chars))
    elif region.kind is RegionKind.FORMULA and region.latex and not region.text:
        parts.append(region.latex)
    elif region.text:
        parts.append(region.text)
    if include_derived:
        for item in region.derived:
            if item.kind is DerivedKind.CHART_ESTIMATE and not include_estimates:
                continue
            tag = "verified" if item.verified else "derived, unverified"
            parts.append(f"[{tag}: {item.kind.value}] {item.text}")
    return "\n".join(p for p in parts if p)


def result_to_text(result: ExtractionResult, *, include_furniture: bool = False) -> str:
    """Document text in explicit region order; headers/footers opt-in."""
    chunks = []
    for region in result.regions():
        if region.kind in FURNITURE and not include_furniture:
            continue
        text = region_text(region)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def result_to_markdown(result: ExtractionResult) -> str:
    """Lossy Markdown view: lossless Markdown tables, HTML for complex ones."""
    chunks = []
    for region in result.regions():
        if region.kind in FURNITURE:
            continue
        if region.kind is RegionKind.HEADING and region.text:
            chunks.append("#" * max(1, min(6, len(region.section_path) or 1)) + " " + region.text)
        elif region.table is not None:
            chunks.append(table_to_markdown(region.table) or table_to_html(region.table))
        else:
            text = region_text(region)
            if text:
                chunks.append(text)
    return "\n\n".join(chunks)
