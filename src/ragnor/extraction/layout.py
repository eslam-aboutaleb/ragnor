"""Map provider-specific layout objects to typed ``Region`` nodes.

Providers differ in labels, box conventions and table shapes. A
:class:`RegionNormalizer` converts a plain mapping per layout object into a
``Region`` in the canonical coordinate space, assigns deterministic ids, and
never invents confidence or text.

Generic item keys: ``label``, ``bbox`` ``[x0, y0, x1, y1]``, optional
``origin`` (``"TOPLEFT"``/``"BOTTOMLEFT"``), ``text``, ``key``, ``order``,
``confidence``, ``grid`` (rule-based rows) or ``cells`` (model cells),
``latex``, ``section_path``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ragnor.extraction.errors import ExtractionError
from ragnor.extraction.ids import deterministic_region_id
from ragnor.extraction.tables import table_from_cell_dicts, table_from_grid
from ragnor.extraction.types import (
    BoundingBox,
    CoordinateSpace,
    ExtractionResult,
    OrderState,
    Page,
    Region,
    RegionKind,
)

DEFAULT_LABELS: dict[str, RegionKind] = {
    "text": RegionKind.TEXT,
    "paragraph": RegionKind.TEXT,
    "list_item": RegionKind.TEXT,
    "code": RegionKind.TEXT,
    "title": RegionKind.HEADING,
    "section_header": RegionKind.HEADING,
    "heading": RegionKind.HEADING,
    "table": RegionKind.TABLE,
    "document_index": RegionKind.TABLE,
    "picture": RegionKind.FIGURE,
    "figure": RegionKind.FIGURE,
    "chart": RegionKind.FIGURE,
    "formula": RegionKind.FORMULA,
    "equation": RegionKind.FORMULA,
    "caption": RegionKind.CAPTION,
    "page_header": RegionKind.HEADER,
    "header": RegionKind.HEADER,
    "page_footer": RegionKind.FOOTER,
    "footer": RegionKind.FOOTER,
    "footnote": RegionKind.FOOTNOTE,
    "sidebar": RegionKind.SIDEBAR,
}


class RegionNormalizer:
    """Normalize one provider's layout objects for a document revision."""

    def __init__(
        self,
        revision: str,
        *,
        method: str,
        labels: Mapping[str, RegionKind] | None = None,
        strict_labels: bool = False,
    ) -> None:
        if not revision or not method:
            raise ValueError("A normalizer needs a revision and a method name")
        self.revision = revision
        self.method = method
        self.labels = dict(labels or DEFAULT_LABELS)
        self.strict_labels = strict_labels

    def _kind(self, label: str) -> RegionKind:
        kind = self.labels.get(str(label).lower())
        if kind is not None:
            return kind
        if self.strict_labels:
            raise ExtractionError("invalid_output")
        return RegionKind.TEXT

    def region(self, page: Page, item: Mapping[str, Any], index: int) -> Region:
        kind = self._kind(item.get("label", "text"))
        x0, y0, x1, y1 = item["bbox"]
        origin = str(item.get("origin", "TOPLEFT")).upper()
        space = (
            CoordinateSpace.PDF_POINTS_BOTTOM_LEFT if origin.endswith("BOTTOMLEFT") else page.space
        )
        box = BoundingBox(
            min(x0, x1),
            min(y0, y1),
            max(x0, x1),
            max(y0, y1),
            page.width,
            page.height,
            page.rotation,
            page.unit,
            space,
        ).to_top_left()
        key = str(item.get("key") or f"{kind.value}:{index}:{x0:.2f}:{y0:.2f}:{x1:.2f}:{y1:.2f}")
        table = None
        if kind is RegionKind.TABLE:
            if item.get("cells"):
                table = table_from_cell_dicts(
                    item["cells"], rows=item.get("rows"), cols=item.get("cols")
                )
            elif item.get("grid"):
                table = table_from_grid(item["grid"], header_rows=item.get("header_rows", 1))
        latex = item.get("latex") if kind is RegionKind.FORMULA else None
        return Region(
            deterministic_region_id(self.revision, page.number, key),
            kind,
            page.number,
            box,
            int(item.get("order", index)),
            text=item.get("text"),
            table=table,
            section_path=tuple(item.get("section_path", ())),
            latex=latex,
            method=self.method,
            confidence=item.get("confidence"),
            order_state=OrderState.PROVIDER,
        )

    def page(  # noqa: PLR0913
        self,
        number: int,
        width: float,
        height: float,
        items: Iterable[Mapping[str, Any]],
        *,
        rotation: int = 0,
        direction: str = "ltr",
    ) -> Page:
        frame = Page(number, width, height, rotation=rotation, direction=direction)
        regions = [self.region(frame, item, i) for i, item in enumerate(items)]
        if len({r.order for r in regions}) != len(regions):  # provider orders collide
            regions = [replace(r, order=i) for i, r in enumerate(regions)]
        return Page(number, width, height, tuple(regions), rotation=rotation, direction=direction)


def docling_items(document: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group a DoclingDocument-shaped export (``texts``/``tables``/``pictures``)
    into generic per-page items, preserving provenance boxes and table cells.

    Box fields follow docling-core's ``BoundingBox`` (``l, t, r, b,
    coord_origin``), the same convention the repository's Docling adapter reads.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for collection in ("texts", "tables", "pictures"):
        for n, node in enumerate(document.get(collection, ())):
            for prov in node.get("prov", ())[:1]:
                box = prov["bbox"]
                item: dict[str, Any] = {
                    "label": node.get(
                        "label", {"tables": "table", "pictures": "picture"}.get(collection, "text")
                    ),
                    "bbox": [box["l"], box["t"], box["r"], box["b"]],
                    "origin": box.get("coord_origin", "TOPLEFT"),
                    "text": node.get("text"),
                    "key": node.get("self_ref") or f"{collection}/{n}",
                }
                if collection == "tables":
                    data = node.get("data", {})
                    item["cells"] = data.get("table_cells", [])
                    item["rows"], item["cols"] = data.get("num_rows"), data.get("num_cols")
                out.setdefault(prov["page_no"], []).append(item)
    return out


def normalize_docling(document: Mapping[str, Any], *, revision: str) -> ExtractionResult:
    """Docling-shaped export to an ``ExtractionResult`` (provider order kept)."""
    normalizer = RegionNormalizer(revision, method="docling")
    grouped = docling_items(document)
    pages = []
    for number in sorted(grouped):
        size = document.get("pages", {}).get(str(number), {}).get("size", {})
        pages.append(
            normalizer.page(
                number, size.get("width", 612.0), size.get("height", 792.0), grouped[number]
            )
        )
    return ExtractionResult(revision, tuple(pages))
