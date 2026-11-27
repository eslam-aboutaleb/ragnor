# ruff: noqa: B023 -- closures are called immediately within the loop iteration
"""Bridges between the structured model and the legacy ingestion records.

* :func:`to_raw_documents` projects a result onto :class:`RawDocument` so the
  existing ``IngestionPipeline`` and chunkers keep working. Page, region id,
  bounding box and revision travel in ``metadata``/``location`` and therefore
  reach every chunk and vector-store row.
* :func:`from_page_outcomes` lifts the legacy per-page evidence
  (``PageExtractionOutcome``) into the structured model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ragnor.extraction.errors import ExtractionLimitError
from ragnor.extraction.serializers import FURNITURE, region_text
from ragnor.extraction.types import (
    BoundingBox,
    Cell,
    ExtractionResult,
    ExtractionStatus,
    Page,
    PageStatus,
    Region,
    RegionKind,
    Table,
)
from ragnor.ingestion.extraction_types import PageExtractionOutcome, PageExtractionStatus
from ragnor.ingestion.records import RawDocument

MAX_REGIONS_IN_PAGE_METADATA = 200
MAX_REGION_TEXT_CHARS = 100_000


def region_metadata(region: Region, revision: str) -> dict[str, Any]:
    """Compact, JSON-safe provenance for one region (no content, no cells)."""
    box = region.bbox
    meta: dict[str, Any] = {
        "revision": revision,
        "region_id": region.region_id,
        "kind": region.kind.value,
        "page": region.page,
        "order": region.order,
        "order_state": region.order_state.value,
        "bbox": [box.x0, box.y0, box.x1, box.y1],
        "bbox_space": box.space.value,
        "page_size": [box.page_width, box.page_height],
        "rotation": box.rotation,
        "section_path": list(region.section_path),
        "method": region.method,
    }
    if region.table is not None:
        meta["table"] = {"rows": region.table.rows, "cols": region.table.cols}
    if region.caption_target:
        meta["caption_target"] = region.caption_target
    if region.continues_from:
        meta["continues_from"] = region.continues_from
    if region.page_refs:
        meta["page_refs"] = list(region.page_refs)
    return meta


def _bounded_text(region: Region) -> tuple[str, bool]:
    try:
        text = region_text(region, max_chars=MAX_REGION_TEXT_CHARS)
    except ExtractionLimitError:
        rows = region.table.rows if region.table else 0
        cols = region.table.cols if region.table else 0
        return f"[table omitted from index text: {rows}x{cols} exceeds the size budget]", True
    if len(text) > MAX_REGION_TEXT_CHARS:
        return text[:MAX_REGION_TEXT_CHARS], True
    return text, False


def to_raw_documents(  # noqa: PLR0913
    result: ExtractionResult,
    *,
    source_name: str,
    external_id: str | None = None,
    granularity: str = "region",
    extractor: str = "structured",
    include_furniture: bool = False,
    metadata: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[RawDocument]:
    """Project a structured result onto legacy ``RawDocument`` records.

    ``granularity="region"`` emits one document per text-bearing region so a
    chunk can never straddle regions and always inherits its region id,
    page and bbox. ``"page"`` emits one document per page with a bounded
    region index.
    """
    if granularity not in {"region", "page"}:
        raise ValueError("granularity must be 'region' or 'page'")
    base = dict(metadata or {})
    docs: list[RawDocument] = []
    for page in sorted(result.pages, key=lambda p: p.number):
        regions = [
            r for r in page.ordered_regions() if include_furniture or r.kind not in FURNITURE
        ]
        if granularity == "page":
            texts = [(r, *_bounded_text(r)) for r in regions]
            content = "\n\n".join(t for _, t, _ in texts if t)
            if not content:
                continue
            docs.append(
                RawDocument(
                    source_name=source_name,
                    content=content,
                    external_id=external_id,
                    metadata={
                        **base,
                        "page": page.number,
                        "structured": {
                            "revision": result.revision,
                            "regions": [
                                region_metadata(r, result.revision)
                                for r in regions[:MAX_REGIONS_IN_PAGE_METADATA]
                            ],
                            "regions_truncated": len(regions) > MAX_REGIONS_IN_PAGE_METADATA,
                        },
                    },
                    provenance=dict(provenance or {}),
                    extractor=extractor,
                    location={"page": page.number},
                )
            )
            continue
        for region in regions:
            text, truncated = _bounded_text(region)
            if not text:
                continue
            meta = region_metadata(region, result.revision)
            if truncated:
                meta["truncated"] = True
            docs.append(
                RawDocument(
                    source_name=source_name,
                    content=text,
                    external_id=external_id,
                    metadata={
                        **base,
                        "page": page.number,
                        "region_id": region.region_id,
                        "structured": meta,
                    },
                    provenance=dict(provenance or {}),
                    extractor=extractor,
                    location={
                        "page": page.number,
                        "region_id": region.region_id,
                        "bbox": meta["bbox"],
                    },
                )
            )
    return docs


_LEGACY_KIND = {"text": RegionKind.TEXT, "table": RegionKind.TABLE, "heading": RegionKind.HEADING}
_LEGACY_STATUS = {
    PageExtractionStatus.EXTRACTED: PageStatus.EXTRACTED,
    PageExtractionStatus.VERIFIED_BLANK: PageStatus.BLANK,
    PageExtractionStatus.POLICY_EXCLUDED: PageStatus.BLANK,
    PageExtractionStatus.NEEDS_OCR: PageStatus.PARTIAL,
    PageExtractionStatus.PARTIAL: PageStatus.PARTIAL,
    PageExtractionStatus.FAILED: PageStatus.FAILED,
}


def _words_in(box: tuple[float, float, float, float], words: Iterable[Any]) -> str:
    inside = [
        w
        for w in words
        if box[0] <= (w.bbox[0] + w.bbox[2]) / 2 <= box[2]
        and box[1] <= (w.bbox[1] + w.bbox[3]) / 2 <= box[3]
    ]
    inside.sort(key=lambda w: (round((w.bbox[1] + w.bbox[3]) / 2, 2), w.bbox[0]))
    return " ".join(w.text for w in inside)


def from_page_outcomes(
    outcomes: Iterable[PageExtractionOutcome], *, revision: str, method: str | None = None
) -> ExtractionResult:
    """Lift legacy page evidence (normalized coordinates) into the new model.

    Provider confidences are kept only when the legacy word supplied one; no
    confidence is invented. Pages without regions get one page-sized text
    region holding the page text so nothing is dropped.
    """
    pages: list[Page] = []
    complete = True
    for outcome in outcomes:
        complete = complete and outcome.complete
        how = method or outcome.provider
        regions: list[Region] = []
        n = 0

        def add(kind: RegionKind, bbox: BoundingBox, **kw: Any) -> None:
            nonlocal n
            regions.append(
                Region(f"p{outcome.page}-r{n}", kind, outcome.page, bbox, n, method=how, **kw)
            )
            n += 1

        for legacy in outcome.regions:
            kind = _LEGACY_KIND.get(legacy.kind, RegionKind.TEXT)
            if kind is RegionKind.TABLE:
                continue
            add(
                kind,
                BoundingBox.normalized(*legacy.bbox),
                text=_words_in(legacy.bbox, outcome.words),
            )
        for legacy_table in outcome.tables:
            table = Table(
                legacy_table.rows,
                legacy_table.cols,
                tuple(
                    Cell(
                        c.row,
                        c.col,
                        c.text,
                        c.rowspan,
                        c.colspan,
                        bbox=BoundingBox.normalized(*c.bbox),
                    )
                    for c in legacy_table.cells
                ),
            )
            add(RegionKind.TABLE, BoundingBox.normalized(*legacy_table.bbox), table=table)
        if not regions and outcome.text.strip():
            add(RegionKind.TEXT, BoundingBox.normalized(0, 0, 1, 1), text=outcome.text)
        pages.append(
            Page(
                outcome.page,
                1.0,
                1.0,
                tuple(regions),
                unit="normalized",
                space="normalized-top-left",
                status=_LEGACY_STATUS[outcome.status],
            )
        )
    return ExtractionResult(
        revision,
        tuple(pages),
        status=ExtractionStatus.COMPLETE if complete else ExtractionStatus.PARTIAL,
    )
