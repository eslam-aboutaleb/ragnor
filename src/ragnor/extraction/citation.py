"""Resolve retrieval hits back to page, region, bbox and table cell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ragnor.extraction.errors import ExtractionError
from ragnor.extraction.types import BoundingBox, Cell, ExtractionResult, Region


@dataclass(frozen=True)
class Citation:
    """What a chunk's metadata says about where its text came from."""

    source: str | None
    revision: str
    page: int
    region_id: str


@dataclass(frozen=True)
class ResolvedCitation:
    citation: Citation
    region: Region
    bbox: BoundingBox
    cell: Cell | None = None


def citation_from_metadata(metadata: Mapping[str, Any]) -> Citation:
    """Read the provenance ``to_raw_documents`` stamps onto every chunk."""
    structured = metadata.get("structured")
    if not isinstance(structured, Mapping):
        raise ExtractionError("invalid_document")
    try:
        return Citation(
            metadata.get("source"),
            str(structured["revision"]),
            int(structured["page"]),
            str(structured["region_id"]),
        )
    except (KeyError, TypeError, ValueError):
        raise ExtractionError("invalid_document") from None


def resolve_citation(
    result: ExtractionResult, citation: Citation, *, cell_text: str | None = None
) -> ResolvedCitation:
    """Find the cited region in ``result`` (and a cell by exact value if asked).

    Fails closed on a revision mismatch: a citation never resolves against a
    different document revision. ``cell_text`` must match exactly one cell,
    otherwise no cell is returned (no guessing between duplicates).
    """
    if result.revision != citation.revision:
        raise ExtractionError("invalid_document")
    region = result.region(citation.region_id)
    if region is None or region.page != citation.page:
        raise ExtractionError("invalid_document")
    cell = None
    box = region.bbox
    if cell_text is not None and region.table is not None:
        hits = [c for c in region.table.cells if c.text == cell_text]
        if len(hits) == 1:
            cell = hits[0]
            box = cell.bbox or region.bbox
    return ResolvedCitation(citation, region, box, cell)
