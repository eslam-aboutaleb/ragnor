"""Dependency-free structured extraction evidence and host verification port."""

from __future__ import annotations

import html
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from collections.abc import Iterable

if TYPE_CHECKING:
    from ragnor.ingestion.extractors import PageExtractionOutcome

BBox = tuple[float, float, float, float]


def checked_bbox(value: BBox) -> None:
    if len(value) != 4 or not all(math.isfinite(v) for v in value):
        raise ValueError("Extraction boxes require four finite coordinates")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Extraction boxes must be normalized to the page")


@dataclass(frozen=True)
class ExtractionWord:
    text: str = field(repr=False)
    bbox: BBox
    confidence: float | None = None

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        if self.confidence is not None and not (
            math.isfinite(self.confidence) and 0 <= self.confidence <= 100
        ):
            raise ValueError("Word confidence must be between zero and 100")


@dataclass(frozen=True)
class ExtractionRegion:
    kind: str
    bbox: BBox

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)


@dataclass(frozen=True)
class ExtractionCell:
    row: int
    col: int
    text: str = field(repr=False)
    bbox: BBox
    rowspan: int = 1
    colspan: int = 1

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        for key in ("row", "col", "rowspan", "colspan"):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key.endswith("span") else 0):
                raise ValueError("Invalid table cell coordinates or spans")


@dataclass(frozen=True)
class ExtractionTable:
    bbox: BBox
    rows: int
    cols: int
    cells: tuple[ExtractionCell, ...] = field(repr=False)
    provider: str = "pdfplumber"

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        if any(type(v) is not int or v < 1 for v in (self.rows, self.cols)):
            raise ValueError("Table dimensions must be positive integers")
        if self.rows * self.cols > 50_000:
            raise ValueError("Table grid exceeds the supported size")
        covered = set()
        for cell in self.cells:
            if cell.row + cell.rowspan > self.rows or cell.col + cell.colspan > self.cols:
                raise ValueError("Cell span exceeds table dimensions")
            for row in range(cell.row, cell.row + cell.rowspan):
                for col in range(cell.col, cell.col + cell.colspan):
                    if (row, col) in covered:
                        raise ValueError("Overlapping table cells")
                    covered.add((row, col))

    def to_html(self) -> str:
        """Render anchored cells, retaining merged-cell spans and escaping values."""
        anchors = {(cell.row, cell.col): cell for cell in self.cells}
        covered = {
            (row, col)
            for cell in self.cells
            for row in range(cell.row, cell.row + cell.rowspan)
            for col in range(cell.col, cell.col + cell.colspan)
        }
        rows = []
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                cell = anchors.get((row, col))
                if cell is not None:
                    cells.append(
                        f'<td rowspan="{cell.rowspan}" colspan="{cell.colspan}">'
                        + html.escape(cell.text)
                        + "</td>"
                    )
                elif (row, col) not in covered:
                    cells.append("<td></td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return "<table>" + "".join(rows) + "</table>"


class PageVerifier(Protocol):
    """Host-owned acceptance policy; identity/options must describe its behavior.

    Returning True accepts this page for ingestion. OCR success alone does not.
    Implementations must be deterministic for their declared version/options.
    """

    id: str
    version: str
    options: dict[str, Any]

    def verify(self, page: PageExtractionOutcome) -> bool: ...


def extraction_evidence_hash(page: PageExtractionOutcome) -> str:
    """Bind source bytes, provider/models, text, geometry and diagnostics to review.

    A later acceptance decision does not alter the evidence it approved.
    """
    value = asdict(page)
    value.pop("status")
    value.pop("reason")
    value["diagnostics"].pop("verification", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


class EvidenceAllowlistVerifier:
    """Accept only exact evidence hashes explicitly supplied by the host.

    Ragnor does not populate this list or claim that it has been human reviewed.
    """

    id = "evidence-allowlist"
    version = "1"

    def __init__(self, approved_hashes: Iterable[str]) -> None:
        hashes = frozenset(approved_hashes)
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError("Evidence approvals must be SHA-256 hex digests")
        self._hashes = hashes

    @property
    def options(self) -> dict[str, Any]:
        return {"approved_evidence_sha256": sorted(self._hashes)}

    def verify(self, page: PageExtractionOutcome) -> bool:
        return extraction_evidence_hash(page) in self._hashes


@dataclass(frozen=True)
class PDFExtractionLimits:
    max_bytes: int = 50_000_000
    max_pages: int = 1000
    max_pixels: int = 25_000_000
    max_output_bytes: int = 32_000_000
    max_text_chars: int = 2_000_000
    max_words: int = 200_000
    max_cells: int = 50_000
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if key == "timeout_seconds":
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Extraction timeout must be positive and finite")
            elif type(value) is not int or value < 1:
                raise ValueError("Extraction limits must be positive integers")


class PDFExtractionError(RuntimeError):
    """Fixed, content-free SDK failure code; provider exceptions stay isolated."""

    codes = frozenset(
        {
            "input_limit",
            "timeout",
            "worker_failed",
            "output_limit",
            "dependency_missing",
            "invalid_document",
            "resource_limit",
            "runtime_changed",
            "provider_failed",
            "invalid_worker_response",
            "verification_failed",
            "ocr_requires_posix_process_cleanup",
        }
    )

    def __init__(self, code: str):
        self.code = code if isinstance(code, str) and code in self.codes else "provider_failed"
        super().__init__(f"PDF extraction failed: {self.code}")
