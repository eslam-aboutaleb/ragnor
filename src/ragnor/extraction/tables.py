"""Table normalization: rule-based grids and model cell lists to one schema.

Values are kept exactly as parsed. A cell the provider reported as ``None``
becomes an explicit missing cell; nothing is padded, merged by guesswork or
"corrected".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ragnor.extraction.errors import ExtractionError
from ragnor.extraction.ids import deterministic_region_id
from ragnor.extraction.types import (
    BoundingBox,
    Cell,
    ExtractionResult,
    HeaderScope,
    Page,
    Region,
    RegionKind,
    Table,
)


def table_from_grid(rows: Sequence[Sequence[str | None]], *, header_rows: int = 1) -> Table:
    """Rule-based output (e.g. ruled/borderless detectors) to a ``Table``.

    Ragged rows are rejected rather than padded. ``None`` stays a missing cell.
    """
    if not rows or not rows[0]:
        raise ValueError("A table needs at least one row and column")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Ragged table rows are not padded")
    if type(header_rows) is not int or not 0 <= header_rows <= len(rows):
        raise ValueError("Invalid header row count")
    cells = tuple(
        Cell(
            r,
            c,
            value,
            header_scope=HeaderScope.COL if r < header_rows else None,
        )
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    )
    return Table(len(rows), width, cells)


def table_from_cell_dicts(
    cells: Iterable[Mapping[str, Any]], *, rows: int | None = None, cols: int | None = None
) -> Table:
    """Model-based output: anchored cells with spans and header flags."""
    built = []
    for raw in cells:
        scope = raw.get("header_scope")
        if scope is None and raw.get("column_header"):
            scope = HeaderScope.COL
        elif scope is None and raw.get("row_header"):
            scope = HeaderScope.ROW
        if "start_row_offset_idx" in raw:  # docling-style offsets
            row, col = raw["start_row_offset_idx"], raw["start_col_offset_idx"]
            rowspan = raw["end_row_offset_idx"] - row
            colspan = raw["end_col_offset_idx"] - col
        else:
            row, col = raw["row"], raw["col"]
            rowspan, colspan = raw.get("rowspan", 1), raw.get("colspan", 1)
        built.append(Cell(row, col, raw.get("text"), rowspan, colspan, header_scope=scope))
    if not built:
        raise ValueError("A table needs at least one cell")
    n_rows = rows or max(c.row + c.rowspan for c in built)
    n_cols = cols or max(c.col + c.colspan for c in built)
    return Table(n_rows, n_cols, tuple(built))


def split_table_at(table: Table, row: int) -> tuple[Table, Table]:
    """Split a wrongly merged table at a row boundary no cell spans across."""
    if type(row) is not int or not 0 < row < table.rows:
        raise ValueError("Split row must lie inside the table")
    if any(c.row < row < c.row + c.rowspan for c in table.cells):
        raise ValueError("A merged cell spans the split boundary")
    top = tuple(c for c in table.cells if c.row < row)
    bottom = tuple(replace(c, row=c.row - row) for c in table.cells if c.row >= row)
    return Table(row, table.cols, top), Table(table.rows - row, table.cols, bottom)


def stack_tables(parts: Sequence[Table]) -> Table:
    """Concatenate vertically split boxes of one table (same column count)."""
    if not parts or any(p.cols != parts[0].cols for p in parts):
        raise ValueError("Stacked tables need equal column counts")
    cells: list[Cell] = []
    offset = 0
    for part in parts:
        cells.extend(replace(c, row=c.row + offset) for c in part.cells)
        offset += part.rows
    return Table(offset, parts[0].cols, tuple(cells))


def union_box(boxes: Sequence[BoundingBox]) -> BoundingBox:
    first = boxes[0]
    return replace(
        first,
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def merge_declared_table_boxes(
    page: Page, groups: Mapping[str, Sequence[str]], *, revision: str
) -> Page:
    """Merge detected boxes only when the provider declared they are one table.

    ``groups`` maps a group key to region ids, top to bottom. Similarity is
    never used to decide. Column counts must agree or the page is rejected.
    """
    regions = {r.region_id: r for r in page.regions}
    consumed: set[str] = set()
    merged: list[Region] = []
    for key, ids in groups.items():
        parts = [regions[i] for i in ids if i in regions]
        if (
            len(parts) != len(ids)
            or len(parts) < 2
            or any(p.kind is not RegionKind.TABLE or p.table is None for p in parts)
        ):
            raise ExtractionError("invalid_output")
        try:
            table = stack_tables([p.table for p in parts])  # type: ignore[misc]
        except ValueError:
            raise ExtractionError("invalid_output") from None
        first = parts[0]
        merged.append(
            replace(
                first,
                region_id=deterministic_region_id(revision, page.number, f"table-group:{key}"),
                bbox=union_box([p.bbox for p in parts]),
                table=table,
            )
        )
        consumed.update(ids)
    kept = [r for r in page.regions if r.region_id not in consumed]
    ordered = sorted([*kept, *merged], key=lambda r: (r.order, r.region_id))
    return replace(page, regions=tuple(replace(r, order=i) for i, r in enumerate(ordered)))


def _header_signature(table: Table) -> tuple[str | None, ...] | None:
    cells = [c for c in table.cells if c.row == 0]
    if not cells or any(c.header_scope is None for c in cells):
        return None
    return tuple(c.text for c in sorted(cells, key=lambda c: c.col))


def link_table_continuations(result: ExtractionResult) -> ExtractionResult:
    """Link a table that continues on the next page without merging anything.

    A link is set only when the previous page ends with a table, the next page
    starts with one of equal column count, and either the provider already
    declared ``continues_from`` or the first row repeats the same header cells.
    Both regions keep their own page, bbox and cells; ``page_refs`` lists every
    physical page of the logical table.
    """
    replacements: dict[str, Region] = {}
    pages = sorted(result.pages, key=lambda p: p.number)
    for prev, nxt in zip(pages, pages[1:], strict=False):
        if nxt.number != prev.number + 1:
            continue
        body_prev = [r for r in prev.ordered_regions() if r.kind not in _FURNITURE]
        body_next = [r for r in nxt.ordered_regions() if r.kind not in _FURNITURE]
        if not body_prev or not body_next:
            continue
        tail, head = replacements.get(body_prev[-1].region_id, body_prev[-1]), body_next[0]
        if tail.table is None or head.table is None or tail.table.cols != head.table.cols:
            continue
        declared = head.continues_from == tail.region_id
        sig_tail, sig_head = _header_signature(tail.table), _header_signature(head.table)
        repeated = sig_head is not None and sig_head == sig_tail
        if not (declared or repeated):
            continue
        pages_chain = tuple(sorted({*tail.page_refs, tail.page, head.page}))
        replacements[tail.region_id] = replace(tail, page_refs=pages_chain)
        replacements[head.region_id] = replace(
            head, continues_from=tail.region_id, page_refs=pages_chain
        )
    if not replacements:
        return result
    new_pages = tuple(
        replace(p, regions=tuple(replacements.get(r.region_id, r) for r in p.regions))
        for p in result.pages
    )
    return replace(result, pages=new_pages)


_FURNITURE = frozenset({RegionKind.HEADER, RegionKind.FOOTER, RegionKind.FOOTNOTE})


def logical_table(result: ExtractionResult, region_id: str) -> tuple[Table, tuple[str, ...]]:
    """Stack a continuation chain into one logical table (read-only view).

    Repeated header rows of continuation segments are dropped only when they are
    exactly equal to the first segment's header. Returns the table and the
    region ids of its segments in page order.
    """
    start = result.region(region_id)
    if start is None or start.table is None:
        raise ExtractionError("invalid_document")
    chain = [start]
    while chain[0].continues_from:
        prev = result.region(chain[0].continues_from)
        if prev is None or prev.table is None or prev in chain:
            raise ExtractionError("invalid_document")
        chain.insert(0, prev)
    followers = {r.continues_from: r for r in result.regions() if r.continues_from}
    while chain[-1].region_id in followers:
        chain.append(followers[chain[-1].region_id])
    first_header = _header_signature(chain[0].table)  # type: ignore[arg-type]
    parts = [chain[0].table]
    for seg in chain[1:]:
        table = seg.table
        if first_header is not None and _header_signature(table) == first_header:  # type: ignore[arg-type]
            table = split_table_at(table, 1)[1] if table.rows > 1 else None  # type: ignore[arg-type]
        if table is not None:
            parts.append(table)
    return stack_tables(parts), tuple(r.region_id for r in chain)  # type: ignore[arg-type]
