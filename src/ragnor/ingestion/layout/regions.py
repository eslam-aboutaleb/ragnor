"""Geometry and region types shared by layout inventory and reconciliation.

All boxes are normalized ``(x0, y0, x1, y1)`` in the canonical, upright page
frame with the origin at the top-left, the same frame as ``ExtractionRegion``.
Nothing here carries text or cell values: layout planning never edits source
content.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ragnor.ingestion.extraction_types import BBox, PageExtractionOutcome, checked_bbox

REGION_TYPES = frozenset(
    {
        "text",
        "heading",
        "table",
        "figure",
        "caption",
        "formula",
        "list",
        "header",
        "footer",
        "sidebar",
        "repeated_header",
    }
)


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    total = area(inner)
    return intersection(inner, outer) / total if total > 0 else 0.0


def union_box(boxes: Iterable[BBox]) -> BBox:
    items = list(boxes)
    if not items:
        raise ValueError("union_box requires at least one box")
    return (
        min(b[0] for b in items),
        min(b[1] for b in items),
        max(b[2] for b in items),
        max(b[3] for b in items),
    )


def expand_box(box: BBox, margin: float) -> BBox:
    """Grow a box by ``margin`` (fraction of page) and clamp to the page."""
    if not math.isfinite(margin) or not 0 <= margin <= 0.25:
        raise ValueError("Crop margin must be between 0 and 0.25")
    return (
        max(0.0, box[0] - margin),
        max(0.0, box[1] - margin),
        min(1.0, box[2] + margin),
        min(1.0, box[3] + margin),
    )


def to_canonical(box: BBox, rotation: int) -> BBox:
    """Map a box from an image rotated clockwise by ``rotation`` degrees to upright."""
    x0, y0, x1, y1 = box
    if rotation == 0:
        result = box
    elif rotation == 90:
        result = (y0, 1 - x1, y1, 1 - x0)
    elif rotation == 180:
        result = (1 - x1, 1 - y1, 1 - x0, 1 - y0)
    elif rotation == 270:
        result = (1 - y1, x0, 1 - y0, x1)
    else:
        raise ValueError("Rotation must be 0, 90, 180 or 270")
    checked_bbox(result)
    return result


def from_canonical(box: BBox, rotation: int) -> BBox:
    """Inverse of :func:`to_canonical`; used by tests and by hosts building fixtures."""
    inverse = {0: 0, 90: 270, 180: 180, 270: 90}.get(rotation)
    if inverse is None:
        raise ValueError("Rotation must be 0, 90, 180 or 270")
    return to_canonical(box, inverse)


def quantize(box: BBox) -> tuple[int, int, int, int]:
    return (
        round(box[0] * 10_000),
        round(box[1] * 10_000),
        round(box[2] * 10_000),
        round(box[3] * 10_000),
    )


def stable_region_id(page: int, kind: str, box: BBox, parents: Sequence[str] = ()) -> str:
    """Deterministic ID from geometry and ancestry, never from list position."""
    payload = f"{page}|{kind}|{quantize(box)}|{'+'.join(sorted(parents))}"
    return "rg_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class LayoutRegion:
    """One region of a layout proposal produced by a layout detector or parser."""

    region_id: str
    kind: str
    bbox: BBox
    confidence: float | None = None
    reading_order: int | None = None
    caption_target: str | None = None
    grid: tuple[int, int] | None = None  # (rows, cols) for tables, when known

    def __post_init__(self) -> None:
        checked_bbox(self.bbox)
        if not isinstance(self.region_id, str) or not self.region_id:
            raise ValueError("Region IDs must be non-empty strings")
        if self.kind not in REGION_TYPES:
            raise ValueError("Unknown region type")
        if self.confidence is not None and not (
            math.isfinite(self.confidence) and 0 <= self.confidence <= 1
        ):
            raise ValueError("Region confidence must be between zero and one")
        if self.reading_order is not None and (
            type(self.reading_order) is not int or self.reading_order < 0
        ):
            raise ValueError("Reading order must be a non-negative integer")


@dataclass(frozen=True)
class ProposedLayout:
    """The baseline layout for a page plus the evidence used to judge it."""

    page: int
    regions: tuple[LayoutRegion, ...]
    parser_errors: tuple[str, ...] = ()
    unclaimed_ink: tuple[BBox, ...] = ()
    expected_counts: dict[str, int] | None = None
    page_fingerprint: str = ""

    def __post_init__(self) -> None:
        if type(self.page) is not int or self.page < 1:
            raise ValueError("Page numbers start at 1")
        ids = [r.region_id for r in self.regions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate region IDs in proposed layout")
        for box in self.unclaimed_ink:
            checked_bbox(box)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for region in self.regions:
            result[region.kind] = result.get(region.kind, 0) + 1
        return dict(sorted(result.items()))


_KIND_MAP = {
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "caption": "caption",
    "heading": "heading",
    "formula": "formula",
    "list": "list",
    "text": "text",
    "paragraph": "text",
}


def proposed_from_outcome(
    outcome: PageExtractionOutcome, *, page_fingerprint: str = "", **evidence: object
) -> ProposedLayout:
    """Build a baseline proposal from ragnor page evidence without reading text."""
    regions: list[LayoutRegion] = []
    seen: set[str] = set()

    def add(kind: str, box: BBox, grid: tuple[int, int] | None = None) -> None:
        base = stable_region_id(outcome.page, kind, box)
        region_id, n = base, 1
        while region_id in seen:
            n += 1
            region_id = f"{base}.{n}"
        seen.add(region_id)
        regions.append(LayoutRegion(region_id, kind, box, grid=grid))

    for table in outcome.tables:
        add("table", table.bbox, (table.rows, table.cols))
    for region in outcome.regions:
        kind = _KIND_MAP.get(region.kind)
        if kind is not None:
            add(kind, region.bbox)
    return ProposedLayout(
        outcome.page,
        tuple(regions),
        page_fingerprint=page_fingerprint,
        **evidence,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class PageImage:
    """A rendered page. ``rotation`` is the clockwise turn applied versus upright."""

    data: bytes = field(repr=False)
    width: int
    height: int
    dpi: int
    rotation: int = 0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.width, self.height, self.dpi)):
            raise ValueError("Image dimensions and dpi must be positive integers")
        if self.rotation not in {0, 90, 180, 270}:
            raise ValueError("Rotation must be 0, 90, 180 or 270")

    @property
    def pixels(self) -> int:
        return self.width * self.height
