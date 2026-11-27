"""Typed, provider-neutral structured-document model.

Everything here is plain frozen dataclasses with validated invariants and no
third-party imports. Coordinate invariants:

* A :class:`BoundingBox` always carries the page width/height, rotation, unit
  and the declared coordinate space it is expressed in, so a box is never
  ambiguous without its page.
* ``pdf-points-top-left`` is the canonical space: origin at the top-left of
  the *unrotated* page, y growing downwards. Providers using a bottom-left
  origin must convert (see :meth:`BoundingBox.to_top_left`).
* Region order is explicit (``Region.order``) and never inferred from index.
* Confidence is optional and is never invented: ``None`` means the provider
  supplied none.
* Table cell values are stored exactly as parsed (``None`` means the provider
  reported the cell as missing); nothing is padded or corrected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
MAX_TABLE_CELLS = 50_000
MAX_SECTION_DEPTH = 32
_EPS = 1e-6
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CoordinateSpace(StrEnum):
    PDF_POINTS_TOP_LEFT = "pdf-points-top-left"
    PDF_POINTS_BOTTOM_LEFT = "pdf-points-bottom-left"
    NORMALIZED_TOP_LEFT = "normalized-top-left"
    PIXELS_TOP_LEFT = "pixels-top-left"


class Unit(StrEnum):
    POINT = "pt"
    PIXEL = "px"
    NORMALIZED = "normalized"


_DEFAULT_UNIT = {
    CoordinateSpace.PDF_POINTS_TOP_LEFT: Unit.POINT,
    CoordinateSpace.PDF_POINTS_BOTTOM_LEFT: Unit.POINT,
    CoordinateSpace.NORMALIZED_TOP_LEFT: Unit.NORMALIZED,
    CoordinateSpace.PIXELS_TOP_LEFT: Unit.PIXEL,
}


class RegionKind(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    SIDEBAR = "sidebar"


class OrderState(StrEnum):
    """Whether the explicit region order is trustworthy."""

    CERTAIN = "certain"
    AMBIGUOUS = "ambiguous"
    PROVIDER = "provider"  # order taken from the provider, not reconciled


class HeaderScope(StrEnum):
    COL = "col"
    ROW = "row"
    COLGROUP = "colgroup"
    ROWGROUP = "rowgroup"


class DerivedKind(StrEnum):
    DESCRIPTION = "description"
    CHART_ESTIMATE = "chart_estimate"
    OCR_TEXT = "ocr_text"


class PageStatus(StrEnum):
    EXTRACTED = "extracted"
    PARTIAL = "partial"
    FAILED = "failed"
    BLANK = "blank"


class ExtractionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class AttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ReasonCode(StrEnum):
    UNSUPPORTED_FORMAT = "unsupported_format"
    MISSING_CAPABILITY = "missing_capability"
    NOT_CONFIGURED = "not_configured"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    LIMIT_EXCEEDED = "limit_exceeded"
    PARTIAL_OUTPUT = "partial_output"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_id(value: Any, what: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"Invalid {what} identifier")


def _check_confidence(value: float | None) -> None:
    if value is not None and not (_finite(value) and 0.0 <= value <= 1.0):
        raise ValueError("Confidence must be None or between 0 and 1")


def _enum(cls: type[StrEnum], value: Any, what: str) -> Any:
    try:
        return cls(value)
    except ValueError:
        raise ValueError(f"Invalid {what}") from None


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned rectangle plus the page frame that gives it meaning."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    rotation: int = 0
    unit: Unit = Unit.POINT
    space: CoordinateSpace = CoordinateSpace.PDF_POINTS_TOP_LEFT

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", _enum(Unit, self.unit, "unit"))
        object.__setattr__(self, "space", _enum(CoordinateSpace, self.space, "coordinate space"))
        if not all(
            _finite(v)
            for v in (self.x0, self.y0, self.x1, self.y1, self.page_width, self.page_height)
        ):
            raise ValueError("Bounding boxes require finite coordinates")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("Page dimensions must be positive")
        if self.rotation not in {0, 90, 180, 270}:
            raise ValueError("Page rotation must be 0, 90, 180 or 270")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("Bounding box must have positive area")
        if (
            self.x0 < -_EPS
            or self.y0 < -_EPS
            or self.x1 > self.page_width + _EPS
            or self.y1 > self.page_height + _EPS
        ):
            raise ValueError("Bounding box lies outside the page")
        if self.unit is Unit.NORMALIZED and (self.page_width != 1 or self.page_height != 1):
            raise ValueError("Normalized boxes use a unit page")

    @classmethod
    def normalized(cls, x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
        return cls(
            x0, y0, x1, y1, 1.0, 1.0, 0, Unit.NORMALIZED, CoordinateSpace.NORMALIZED_TOP_LEFT
        )

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_top_left(self) -> BoundingBox:
        """Return the same box with a top-left origin."""
        if self.space is not CoordinateSpace.PDF_POINTS_BOTTOM_LEFT:
            return self
        return replace(
            self,
            y0=self.page_height - self.y1,
            y1=self.page_height - self.y0,
            space=CoordinateSpace.PDF_POINTS_TOP_LEFT,
        )

    def to_display(self) -> BoundingBox:
        """Box in the displayed (rotated) frame, top-left origin.

        Page boxes are stored in the unrotated page space. Reading order is a
        property of what a reader sees, so it works on this view.
        """
        box = self.to_top_left()
        w, h = box.page_width, box.page_height
        x0, y0, x1, y1 = box.as_tuple()
        if box.rotation == 90:  # clockwise quarter turn
            return replace(
                box, x0=h - y1, y0=x0, x1=h - y0, y1=x1, page_width=h, page_height=w, rotation=0
            )
        if box.rotation == 180:
            return replace(box, x0=w - x1, y0=h - y1, x1=w - x0, y1=h - y0, rotation=0)
        if box.rotation == 270:
            return replace(
                box, x0=y0, y0=w - x1, x1=y1, y1=w - x0, page_width=h, page_height=w, rotation=0
            )
        return box

    def overlap_area(self, other: BoundingBox) -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [self.x0, self.y0, self.x1, self.y1],
            "page_width": self.page_width,
            "page_height": self.page_height,
            "rotation": self.rotation,
            "unit": self.unit.value,
            "space": self.space.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundingBox:
        box = data["box"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Bounding box needs four coordinates")
        return cls(
            *box,
            page_width=data["page_width"],
            page_height=data["page_height"],
            rotation=data.get("rotation", 0),
            unit=data.get("unit", Unit.POINT),
            space=data.get("space", CoordinateSpace.PDF_POINTS_TOP_LEFT),
        )


@dataclass(frozen=True)
class TextSpan:
    """Half-open character span ``[start, end)`` into a named source text."""

    start: int
    end: int
    source: str = "region_text"

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start <= self.end
        ):
            raise ValueError("Invalid text span")
        if not isinstance(self.source, str) or not self.source or len(self.source) > 64:
            raise ValueError("Invalid text span source")

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "source": self.source}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TextSpan:
        return cls(data["start"], data["end"], data.get("source", "region_text"))


@dataclass(frozen=True)
class AssetRef:
    """Immutable reference to a page image, crop or other artifact.

    Only references are stored: never bytes, and never credentials in the URI.
    """

    asset_id: str
    kind: str
    uri: str | None = None
    sha256: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        _check_id(self.asset_id, "asset")
        if not isinstance(self.kind, str) or not _ID.fullmatch(self.kind):
            raise ValueError("Invalid asset kind")
        if self.sha256 is not None and not (
            isinstance(self.sha256, str) and _SHA256.fullmatch(self.sha256)
        ):
            raise ValueError("Asset sha256 must be a lowercase hex digest")
        if self.uri is not None:
            if not isinstance(self.uri, str) or len(self.uri) > 2048:
                raise ValueError("Invalid asset URI")
            parts = urlsplit(self.uri)
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("Asset URIs must not carry credentials, queries or fragments")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetRef:
        return cls(
            **{k: data[k] for k in ("asset_id", "kind", "uri", "sha256", "media_type") if k in data}
        )


@dataclass(frozen=True)
class Cell:
    """One table cell anchored at ``(row, col)``.

    ``text`` is the exact parsed value; ``None`` means the provider reported
    the cell as missing and nothing was inferred for it.
    """

    row: int
    col: int
    text: str | None = field(default=None, repr=False)
    rowspan: int = 1
    colspan: int = 1
    header_scope: HeaderScope | None = None
    bbox: BoundingBox | None = None
    source_span: TextSpan | None = None

    def __post_init__(self) -> None:
        for name, minimum in (("row", 0), ("col", 0), ("rowspan", 1), ("colspan", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid table cell coordinates or spans")
        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("Cell text must be a string or None")
        if self.header_scope is not None:
            object.__setattr__(
                self, "header_scope", _enum(HeaderScope, self.header_scope, "header scope")
            )

    @property
    def missing(self) -> bool:
        return self.text is None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
        }
        if self.header_scope is not None:
            out["header_scope"] = self.header_scope.value
        if self.bbox is not None:
            out["bbox"] = self.bbox.to_dict()
        if self.source_span is not None:
            out["source_span"] = self.source_span.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Cell:
        return cls(
            row=data["row"],
            col=data["col"],
            text=data.get("text"),
            rowspan=data.get("rowspan", 1),
            colspan=data.get("colspan", 1),
            header_scope=data.get("header_scope"),
            bbox=BoundingBox.from_dict(data["bbox"]) if data.get("bbox") else None,
            source_span=TextSpan.from_dict(data["source_span"])
            if data.get("source_span")
            else None,
        )


@dataclass(frozen=True)
class Table:
    """Table grid with anchored cells; uncovered positions are explicitly absent."""

    rows: int
    cols: int
    cells: tuple[Cell, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.rows, self.cols)):
            raise ValueError("Table dimensions must be positive integers")
        if self.rows * self.cols > MAX_TABLE_CELLS:
            raise ValueError("Table grid exceeds the supported size")
        object.__setattr__(self, "cells", tuple(self.cells))
        covered: set[tuple[int, int]] = set()
        for cell in self.cells:
            if not isinstance(cell, Cell):
                raise ValueError("Table cells must be Cell instances")
            if cell.row + cell.rowspan > self.rows or cell.col + cell.colspan > self.cols:
                raise ValueError("Cell span exceeds table dimensions")
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if (r, c) in covered:
                        raise ValueError("Overlapping table cells")
                    covered.add((r, c))

    def occupancy(self) -> dict[tuple[int, int], Cell]:
        """Map every covered grid position to its anchoring cell."""
        grid: dict[tuple[int, int], Cell] = {}
        for cell in self.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    grid[(r, c)] = cell
        return grid

    def uncovered_positions(self) -> list[tuple[int, int]]:
        """Grid positions no cell covers: explicitly absent, never inferred."""
        covered = self.occupancy()
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in covered]

    def cell_at(self, row: int, col: int) -> Cell | None:
        return self.occupancy().get((row, col))

    def iter_cells(self) -> Iterator[Cell]:
        return iter(sorted(self.cells, key=lambda c: (c.row, c.col)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cells": [c.to_dict() for c in self.iter_cells()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Table:
        return cls(data["rows"], data["cols"], tuple(Cell.from_dict(c) for c in data["cells"]))


@dataclass(frozen=True)
class DerivedContent:
    """Model-produced text about a region; never verified source text."""

    kind: DerivedKind
    text: str = field(repr=False)
    method: str
    confidence: float | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(DerivedKind, self.kind, "derived kind"))
        if not isinstance(self.text, str) or not isinstance(self.method, str) or not self.method:
            raise ValueError("Derived content needs text and a method")
        _check_confidence(self.confidence)
        if self.verified and self.kind is DerivedKind.CHART_ESTIMATE:
            raise ValueError("Chart estimates can never be marked verified")

    def to_dict(self) -> dict[str, Any]:
        out = {"kind": self.kind.value, "text": self.text, "method": self.method}
        out["verified"] = self.verified
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DerivedContent:
        return cls(
            data["kind"],
            data["text"],
            data["method"],
            data.get("confidence"),
            data.get("verified", False),
        )


@dataclass(frozen=True)
class Region:
    """A typed layout region with explicit order, geometry and provenance."""

    region_id: str
    kind: RegionKind
    page: int
    bbox: BoundingBox
    order: int
    text: str | None = field(default=None, repr=False)
    table: Table | None = field(default=None, repr=False)
    section_path: tuple[str, ...] = ()
    caption_target: str | None = None
    caption_id: str | None = None
    continues_from: str | None = None
    page_refs: tuple[int, ...] = ()
    latex: str | None = field(default=None, repr=False)
    assets: tuple[AssetRef, ...] = ()
    derived: tuple[DerivedContent, ...] = field(default=(), repr=False)
    method: str = ""
    confidence: float | None = None
    order_state: OrderState = OrderState.PROVIDER
    source_span: TextSpan | None = None

    def __post_init__(self) -> None:
        _check_id(self.region_id, "region")
        object.__setattr__(self, "kind", _enum(RegionKind, self.kind, "region kind"))
        object.__setattr__(self, "order_state", _enum(OrderState, self.order_state, "order state"))
        if type(self.page) is not int or self.page < 1:
            raise ValueError("Region page must be a positive integer")
        if type(self.order) is not int or self.order < 0:
            raise ValueError("Region order must be a non-negative integer")
        if not isinstance(self.bbox, BoundingBox):
            raise ValueError("Region needs a BoundingBox")
        _check_confidence(self.confidence)
        if self.kind is RegionKind.TABLE:
            if self.table is not None and not isinstance(self.table, Table):
                raise ValueError("Region table must be a Table")
        elif self.table is not None:
            raise ValueError("Only table regions carry a table")
        if self.latex is not None and self.kind is not RegionKind.FORMULA:
            raise ValueError("Only formula regions carry LaTeX")
        for ref in (self.caption_target, self.caption_id, self.continues_from):
            if ref is not None:
                _check_id(ref, "region reference")
        if self.caption_target is not None and self.kind is not RegionKind.CAPTION:
            raise ValueError("Only caption regions have a caption target")
        if self.region_id in {self.caption_target, self.caption_id, self.continues_from}:
            raise ValueError("A region cannot reference itself")
        if len(self.section_path) > MAX_SECTION_DEPTH or not all(
            isinstance(s, str) for s in self.section_path
        ):
            raise ValueError("Invalid section path")
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(self, "derived", tuple(self.derived))
        pages = tuple(self.page_refs)
        if any(type(p) is not int or p < 1 for p in pages):
            raise ValueError("Invalid page references")
        object.__setattr__(self, "page_refs", pages)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "region_id": self.region_id,
            "kind": self.kind.value,
            "page": self.page,
            "bbox": self.bbox.to_dict(),
            "order": self.order,
            "order_state": self.order_state.value,
            "section_path": list(self.section_path),
            "method": self.method,
        }
        for name in (
            "text",
            "caption_target",
            "caption_id",
            "continues_from",
            "latex",
            "confidence",
        ):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.table is not None:
            out["table"] = self.table.to_dict()
        if self.page_refs:
            out["page_refs"] = list(self.page_refs)
        if self.assets:
            out["assets"] = [a.to_dict() for a in self.assets]
        if self.derived:
            out["derived"] = [d.to_dict() for d in self.derived]
        if self.source_span is not None:
            out["source_span"] = self.source_span.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Region:
        return cls(
            region_id=data["region_id"],
            kind=data["kind"],
            page=data["page"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            order=data["order"],
            text=data.get("text"),
            table=Table.from_dict(data["table"]) if data.get("table") else None,
            section_path=tuple(data.get("section_path", ())),
            caption_target=data.get("caption_target"),
            caption_id=data.get("caption_id"),
            continues_from=data.get("continues_from"),
            page_refs=tuple(data.get("page_refs", ())),
            latex=data.get("latex"),
            assets=tuple(AssetRef.from_dict(a) for a in data.get("assets", ())),
            derived=tuple(DerivedContent.from_dict(d) for d in data.get("derived", ())),
            method=data.get("method", ""),
            confidence=data.get("confidence"),
            order_state=data.get("order_state", OrderState.PROVIDER),
            source_span=TextSpan.from_dict(data["source_span"])
            if data.get("source_span")
            else None,
        )


@dataclass(frozen=True)
class Page:
    """One physical page: its frame, status and ordered regions."""

    number: int
    width: float
    height: float
    regions: tuple[Region, ...] = ()
    rotation: int = 0
    unit: Unit = Unit.POINT
    space: CoordinateSpace = CoordinateSpace.PDF_POINTS_TOP_LEFT
    direction: str = "ltr"
    status: PageStatus = PageStatus.EXTRACTED
    image: AssetRef | None = None

    def __post_init__(self) -> None:
        if type(self.number) is not int or self.number < 1:
            raise ValueError("Page number must be a positive integer")
        object.__setattr__(self, "unit", _enum(Unit, self.unit, "unit"))
        object.__setattr__(self, "space", _enum(CoordinateSpace, self.space, "coordinate space"))
        object.__setattr__(self, "status", _enum(PageStatus, self.status, "page status"))
        if not (
            _finite(self.width) and _finite(self.height) and self.width > 0 and self.height > 0
        ):
            raise ValueError("Page dimensions must be positive and finite")
        if self.rotation not in {0, 90, 180, 270}:
            raise ValueError("Page rotation must be 0, 90, 180 or 270")
        if self.direction not in {"ltr", "rtl"}:
            raise ValueError("Page direction must be ltr or rtl")
        object.__setattr__(self, "regions", tuple(self.regions))
        ids: set[str] = set()
        orders: set[int] = set()
        for region in self.regions:
            if region.page != self.number:
                raise ValueError("Region page does not match its page")
            box = region.bbox
            if (box.page_width, box.page_height, box.rotation) != (
                self.width,
                self.height,
                self.rotation,
            ) or (box.unit, box.space) != (self.unit, self.space):
                raise ValueError("Region box frame differs from the page frame")
            if region.region_id in ids:
                raise ValueError("Duplicate region id on page")
            if region.order in orders:
                raise ValueError("Duplicate region order on page")
            ids.add(region.region_id)
            orders.add(region.order)

    def ordered_regions(self) -> tuple[Region, ...]:
        return tuple(sorted(self.regions, key=lambda r: r.order))

    def box(self, x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
        """Build a box in this page's frame."""
        return BoundingBox(
            x0, y0, x1, y1, self.width, self.height, self.rotation, self.unit, self.space
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "unit": self.unit.value,
            "space": self.space.value,
            "direction": self.direction,
            "status": self.status.value,
            "regions": [r.to_dict() for r in self.ordered_regions()],
        }
        if self.image is not None:
            out["image"] = self.image.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Page:
        return cls(
            number=data["number"],
            width=data["width"],
            height=data["height"],
            regions=tuple(Region.from_dict(r) for r in data.get("regions", ())),
            rotation=data.get("rotation", 0),
            unit=data.get("unit", Unit.POINT),
            space=data.get("space", CoordinateSpace.PDF_POINTS_TOP_LEFT),
            direction=data.get("direction", "ltr"),
            status=data.get("status", PageStatus.EXTRACTED),
            image=AssetRef.from_dict(data["image"]) if data.get("image") else None,
        )


@dataclass(frozen=True)
class DocumentProfile:
    """Cheap facts about a source used to choose a provider."""

    media_type: str
    page_count: int | None = None
    has_text_layer: bool | None = None
    has_images: bool | None = None
    languages: tuple[str, ...] = ()
    size_bytes: int | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("Profile needs a media type")
        for name in ("page_count", "size_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Profile counts must be non-negative integers")
        if self.source_sha256 is not None and not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("Profile sha256 must be a lowercase hex digest")
        object.__setattr__(self, "languages", tuple(self.languages))

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in vars(self).items() if v is not None and k != "languages"}
        out["languages"] = list(self.languages)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentProfile:
        return cls(
            media_type=data["media_type"],
            page_count=data.get("page_count"),
            has_text_layer=data.get("has_text_layer"),
            has_images=data.get("has_images"),
            languages=tuple(data.get("languages", ())),
            size_bytes=data.get("size_bytes"),
            source_sha256=data.get("source_sha256"),
        )


@dataclass(frozen=True)
class ExtractionAttempt:
    """One provider attempt. Carries codes only, never provider messages."""

    provider_id: str
    provider_version: str
    status: AttemptStatus
    reason: ReasonCode | None = None
    options_fingerprint: str = ""
    pages: int = 0

    def __post_init__(self) -> None:
        _check_id(self.provider_id, "provider")
        object.__setattr__(self, "status", _enum(AttemptStatus, self.status, "attempt status"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _enum(ReasonCode, self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "status": self.status.value,
            "options_fingerprint": self.options_fingerprint,
            "pages": self.pages,
        }
        if self.reason is not None:
            out["reason"] = self.reason.value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExtractionAttempt:
        return cls(
            data["provider_id"],
            data["provider_version"],
            data["status"],
            data.get("reason"),
            data.get("options_fingerprint", ""),
            data.get("pages", 0),
        )


@dataclass(frozen=True)
class ExtractionResult:
    """Provider-neutral extraction output for one document revision."""

    revision: str
    pages: tuple[Page, ...]
    profile: DocumentProfile | None = None
    status: ExtractionStatus = ExtractionStatus.COMPLETE
    reason: ReasonCode | None = None
    attempts: tuple[ExtractionAttempt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision or len(self.revision) > 256:
            raise ValueError("Result needs a document revision")
        object.__setattr__(self, "status", _enum(ExtractionStatus, self.status, "status"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _enum(ReasonCode, self.reason, "reason"))
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        numbers = [p.number for p in self.pages]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Duplicate page numbers")
        seen: set[str] = set()
        for page in self.pages:
            for region in page.regions:
                if region.region_id in seen:
                    raise ValueError("Duplicate region id in document")
                seen.add(region.region_id)

    def regions(self) -> Iterator[Region]:
        for page in sorted(self.pages, key=lambda p: p.number):
            yield from page.ordered_regions()

    def region(self, region_id: str) -> Region | None:
        return next((r for r in self.regions() if r.region_id == region_id), None)

    def page(self, number: int) -> Page | None:
        return next((p for p in self.pages if p.number == number), None)

    @property
    def complete(self) -> bool:
        return self.status is ExtractionStatus.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "status": self.status.value,
            "pages": [p.to_dict() for p in sorted(self.pages, key=lambda p: p.number)],
            "attempts": [a.to_dict() for a in self.attempts],
        }
        if self.profile is not None:
            out["profile"] = self.profile.to_dict()
        if self.reason is not None:
            out["reason"] = self.reason.value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExtractionResult:
        from ragnor.extraction.errors import ExtractionError  # noqa: PLC0415

        if not isinstance(data, Mapping) or data.get("schema_version") != SCHEMA_VERSION:
            raise ExtractionError("schema_version")
        return cls(
            revision=data["revision"],
            pages=tuple(Page.from_dict(p) for p in data.get("pages", ())),
            profile=DocumentProfile.from_dict(data["profile"]) if data.get("profile") else None,
            status=data.get("status", ExtractionStatus.COMPLETE),
            reason=data.get("reason"),
            attempts=tuple(ExtractionAttempt.from_dict(a) for a in data.get("attempts", ())),
        )
