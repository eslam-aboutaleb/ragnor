"""Versioned fixture manifest: schema, loading and validation.

The manifest is JSONL, one annotated page per line. Coordinates use the
``pdf-points-top-left`` convention: ``[x0, y0, x1, y1]`` in PDF points with
the origin at the top-left of the page (generators convert from the
bottom-left origin used by PDF content streams).

Validation covers the checks required by issue #60: missing pages,
duplicate region ids, bounding boxes outside the page, inconsistent
rowspan/colspan grids, invalid reading orders and source checksum
mismatches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
COORDINATE_SPACE = "pdf-points-top-left"

TEXT_BEARING_KINDS = {"text", "heading", "table", "caption", "formula", "list"}
REGION_KINDS = TEXT_BEARING_KINDS | {"figure", "decoration"}


@dataclass
class BBox:
    """Axis-aligned rectangle in ``pdf-points-top-left`` coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def validate(self, width: float, height: float, where: str) -> list[str]:
        errors = []
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            errors.append(f"{where}: degenerate bbox {self}")
        if self.x0 < 0 or self.y0 < 0 or self.x1 > width or self.y1 > height:
            errors.append(f"{where}: bbox outside page {width}x{height}: {self}")
        return errors


@dataclass
class Region:
    """One annotated page region."""

    region_id: str
    kind: str
    bbox: BBox
    reading_order: int
    expected_text: str = ""
    excluded: bool = False


@dataclass
class TableCell:
    """One cell of an annotated table grid."""

    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class Table:
    """Annotated table bound to a page region."""

    region_id: str
    rows: int
    cols: int
    cells: list[TableCell]

    def validate(self, where: str) -> list[str]:
        errors = []
        if self.rows < 1 or self.cols < 1:
            errors.append(f"{where}: table must have >= 1 row and column")
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            errors.extend(self._validate_cell(cell, occupied, where))
        return errors

    def _validate_cell(
        self, cell: TableCell, occupied: set[tuple[int, int]], where: str
    ) -> list[str]:
        errors = []
        if cell.rowspan < 1 or cell.colspan < 1:
            errors.append(f"{where}: cell ({cell.row},{cell.col}) span must be >= 1")
            return errors
        if cell.row < 0 or cell.col < 0:
            errors.append(f"{where}: cell ({cell.row},{cell.col}) has negative position")
        if cell.row + cell.rowspan > self.rows or cell.col + cell.colspan > self.cols:
            errors.append(
                f"{where}: cell ({cell.row},{cell.col}) span "
                f"{cell.rowspan}x{cell.colspan} exceeds {self.rows}x{self.cols} grid"
            )
        for r in range(cell.row, min(cell.row + cell.rowspan, self.rows)):
            for c in range(cell.col, min(cell.col + cell.colspan, self.cols)):
                if (r, c) in occupied:
                    errors.append(f"{where}: cells overlap at grid position ({r},{c})")
                occupied.add((r, c))
        return errors


@dataclass
class CriticalField:
    """A business-critical field whose value must survive extraction."""

    name: str
    value: str
    numeric: bool = False


@dataclass
class PageAnnotation:
    """Ground truth for one page of one document."""

    document_id: str
    file: str
    family: str
    page: int
    page_count: int
    width: float
    height: float
    license: str
    source_hash: str
    regions: list[Region] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    critical_fields: list[CriticalField] = field(default_factory=list)
    expect_error: bool = False
    notes: str = ""

    def text_regions(self) -> list[Region]:
        """Regions expected to contribute recoverable text."""
        return [
            r
            for r in self.regions
            if not r.excluded and r.kind in TEXT_BEARING_KINDS and r.expected_text.strip()
        ]


@dataclass
class Manifest:
    """The whole corpus: versioned, hashed, page-annotated."""

    corpus_version: str
    pages: list[PageAnnotation]

    def manifest_hash(self) -> str:
        """Stable hash over the serialized page annotations."""
        digest = hashlib.sha256()
        for page in self.pages:
            digest.update(_page_identity(page).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def documents(self) -> dict[str, list[PageAnnotation]]:
        docs: dict[str, list[PageAnnotation]] = {}
        for page in self.pages:
            docs.setdefault(page.document_id, []).append(page)
        for pages in docs.values():
            pages.sort(key=lambda p: p.page)
        return docs


def _page_identity(page: PageAnnotation) -> str:
    return json.dumps(
        {
            "document_id": page.document_id,
            "page": page.page,
            "source_hash": page.source_hash,
            "regions": [r.region_id for r in page.regions],
            "expected": [r.expected_text for r in page.regions],
            "fields": [(f.name, f.value) for f in page.critical_fields],
        },
        sort_keys=True,
    )


def load_manifest(path: Path | str) -> Manifest:
    """Load a JSONL manifest. The first line may be a ``{"corpus_version": ...}`` header."""
    path = Path(path)
    pages: list[PageAnnotation] = []
    corpus_version = "0.0.0"
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        record = json.loads(line)
        if "corpus_version" in record:
            corpus_version = record["corpus_version"]
            continue
        try:
            pages.append(_parse_page(record))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"manifest line {line_number}: {exc}") from exc
    return Manifest(corpus_version=corpus_version, pages=pages)


def _parse_page(record: dict) -> PageAnnotation:
    regions = [
        Region(
            region_id=r["region_id"],
            kind=r["kind"],
            bbox=BBox(*r["bbox"]),
            reading_order=int(r["reading_order"]),
            expected_text=r.get("expected_text", ""),
            excluded=bool(r.get("excluded", False)),
        )
        for r in record.get("regions", [])
    ]
    tables = [
        Table(
            region_id=t["region_id"],
            rows=int(t["rows"]),
            cols=int(t["cols"]),
            cells=[TableCell(**c) for c in t.get("cells", [])],
        )
        for t in record.get("tables", [])
    ]
    fields = [CriticalField(**f) for f in record.get("critical_fields", [])]
    return PageAnnotation(
        document_id=record["document_id"],
        file=record["file"],
        family=record["family"],
        page=int(record["page"]),
        page_count=int(record["page_count"]),
        width=float(record["width"]),
        height=float(record["height"]),
        license=record["license"],
        source_hash=record["source_hash"],
        regions=regions,
        tables=tables,
        critical_fields=fields,
        expect_error=bool(record.get("expect_error", False)),
        notes=record.get("notes", ""),
    )


def validate_manifest(manifest: Manifest, root: Path | str) -> list[str]:
    """Return all schema/consistency errors; empty means valid.

    ``root`` is the fixtures directory containing the referenced PDF files;
    source hashes are verified against it.
    """
    root = Path(root)
    errors: list[str] = []
    errors.extend(_validate_documents(manifest))
    for page in manifest.pages:
        where = f"{page.document_id} p{page.page}"
        errors.extend(_validate_page(page, where))
        errors.extend(_validate_source(page, root, where))
    return errors


def _validate_documents(manifest: Manifest) -> list[str]:
    errors = []
    for document_id, pages in manifest.documents().items():
        expected = list(range(1, pages[0].page_count + 1))
        actual = [p.page for p in pages]
        if len(actual) != len(set(actual)):
            errors.append(f"{document_id}: duplicate page numbers {sorted(actual)}")
        missing = sorted(set(expected) - set(actual))
        if missing:
            errors.append(f"{document_id}: missing pages {missing} of {pages[0].page_count}")
        if len({p.source_hash for p in pages}) != 1:
            errors.append(f"{document_id}: pages disagree on source_hash")
    return errors


def _validate_page(page: PageAnnotation, where: str) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for region in page.regions:
        if region.region_id in seen:
            errors.append(f"{where}: duplicate region id {region.region_id}")
        seen.add(region.region_id)
        if region.kind not in REGION_KINDS:
            errors.append(f"{where}: unknown region kind {region.kind!r}")
        errors.extend(region.bbox.validate(page.width, page.height, where))
    table_region_ids = {t.region_id for t in page.tables}
    for table in page.tables:
        if table.region_id not in seen:
            errors.append(f"{where}: table {table.region_id} has no matching region")
        errors.extend(table.validate(where))
    for region in page.text_regions():
        if region.kind == "table" and region.region_id not in table_region_ids:
            errors.append(f"{where}: table region {region.region_id} has no table annotation")
    errors.extend(_validate_reading_order(page, where))
    return errors


def _validate_reading_order(page: PageAnnotation, where: str) -> list[str]:
    ordered = [r.reading_order for r in page.regions if not r.excluded]
    if sorted(ordered) != list(range(len(ordered))):
        return [f"{where}: reading order must be a permutation of 0..{len(ordered) - 1}"]
    return []


def _validate_source(page: PageAnnotation, root: Path, where: str) -> list[str]:
    path = root / page.file
    if not path.is_file():
        return [f"{where}: missing fixture file {page.file}"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if f"sha256:{digest}" != page.source_hash:
        return [f"{where}: checksum mismatch for {page.file}"]
    return []
