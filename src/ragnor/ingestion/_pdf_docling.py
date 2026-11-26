"""Docling conversion, isolated inside the bounded SDK worker.

Heavy imports stay lazy and model downloads are disabled: hosts provision
Docling layout/OCR models ahead of time (for example through Docling's own
model download tooling) and point ``artifacts_path`` at them.
"""

from __future__ import annotations

import io
import os
from dataclasses import replace

from ragnor.ingestion.extraction_types import (
    ExtractionCell,
    ExtractionTable,
    PDFExtractionError,
)
from ragnor.ingestion.extractors import PageExtractionStatus


def convert_document(data: bytes, config: dict) -> dict[int, dict]:
    """Run Docling once and return plain per-page text and tables.

    The returned mapping is deliberately simple so the evidence mapping below
    stays testable without Docling installed.
    """
    # Never download models implicitly inside the sandboxed worker.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from docling.datamodel.base_models import DocumentStream, InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
    except ImportError:
        raise PDFExtractionError("dependency_missing") from None
    try:
        options = PdfPipelineOptions(
            do_ocr=config["do_ocr"],
            do_table_structure=config["do_table_structure"],
            artifacts_path=config["artifacts_path"],
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        result = converter.convert(DocumentStream(name="source.pdf", stream=io.BytesIO(data)))
    except PDFExtractionError:
        raise
    except Exception:
        raise PDFExtractionError("provider_failed") from None
    document = result.document
    page_sizes = {
        number: getattr(page, "size", None)
        for number, page in getattr(document, "pages", {}).items()
    }
    pages: dict[int, dict] = {}
    for item in getattr(document, "texts", []):
        text = getattr(item, "text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        for prov in getattr(item, "prov", []):
            page = pages.setdefault(prov.page_no, {"text": [], "tables": []})
            if text not in page["text"]:
                page["text"].append(text)
    for table in getattr(document, "tables", []):
        page_numbers = {prov.page_no for prov in getattr(table, "prov", [])}
        if len(page_numbers) != 1:
            continue
        grid = getattr(getattr(table, "data", None), "grid", None)
        if not grid:
            continue
        page_no = page_numbers.pop()
        cells = [
            {
                "row": cell.start_row_offset_idx,
                "col": cell.start_col_offset_idx,
                "text": cell.text,
                "rowspan": cell.end_row_offset_idx - cell.start_row_offset_idx,
                "colspan": cell.end_col_offset_idx - cell.start_col_offset_idx,
            }
            for row in grid
            for cell in row
            if getattr(cell, "text", "").strip()
        ]
        if cells:
            bbox = _normalized_box(table, page_sizes.get(page_no))
            pages.setdefault(page_no, {"text": [], "tables": []})["tables"].append(
                {
                    "rows": len(grid),
                    "cols": max((len(row) for row in grid), default=0),
                    "cells": cells,
                    "bbox": bbox,
                }
            )
    return pages


def _normalized_box(item, size) -> tuple[float, float, float, float]:
    """Coarse table-level geometry; Docling cells share their table's box."""
    try:
        bbox = item.prov[0].bbox
        left, top, right, bottom = bbox.l, bbox.t, bbox.r, bbox.b
        if str(getattr(bbox, "coord_origin", "")).endswith("BOTTOMLEFT"):
            top, bottom = size.height - bottom, size.height - top
        x0, x1 = sorted((left / size.width, right / size.width))
        y0, y1 = sorted((top / size.height, bottom / size.height))
        box = (
            min(1.0, max(0.0, x0)),
            min(1.0, max(0.0, y0)),
            min(1.0, max(0.0, x1)),
            min(1.0, max(0.0, y1)),
        )
        if box[0] < box[2] and box[1] < box[3]:
            return box
    except (AttributeError, IndexError, TypeError, ZeroDivisionError):
        pass
    return (0.0, 0.0, 1.0, 1.0)


def outcomes_from_conversion(converted: dict[int, dict], baseline, budget, runtime: dict) -> list:
    """Map plain Docling page content onto the pypdf baseline evidence.

    Docling text and tables are unverified: pages become PARTIAL and require
    host verification before publication. Pages Docling leaves empty keep
    their baseline status, so proven blanks stay blank and unknown pages stay
    ``needs_ocr`` instead of being certified empty.
    """
    pages = []
    for evidence in baseline:
        content = converted.get(evidence.page)
        if content is None or not (content["text"] or content["tables"]):
            pages.append(evidence)
            continue
        budget.add("text_chars", sum(len(text) for text in content["text"]))
        tables = []
        for raw in content["tables"]:
            budget.add("cells", len(raw["cells"]))
            budget.add("text_chars", sum(len(cell["text"]) for cell in raw["cells"]))
            tables.append(
                ExtractionTable(
                    raw["bbox"],
                    raw["rows"],
                    raw["cols"],
                    tuple(
                        ExtractionCell(
                            cell["row"],
                            cell["col"],
                            cell["text"],
                            raw["bbox"],
                            cell["rowspan"],
                            cell["colspan"],
                        )
                        for cell in raw["cells"]
                    ),
                    provider="docling",
                )
            )
        text = "\n\n".join(content["text"])
        if tables:
            text = (text + "\n\n" if text else "") + "\n".join(t.to_html() for t in tables)
        pages.append(
            replace(
                evidence,
                status=PageExtractionStatus.PARTIAL,
                reason="docling_verification_required",
                text=text,
                words=(),
                regions=(),
                tables=tuple(tables),
                provider="docling",
                version=str(runtime.get("packages", {}).get("docling") or "unknown"),
                diagnostics={
                    **evidence.diagnostics,
                    "route": "docling",
                    "verification": "pending",
                    "geometry": "table_level",
                    "docling_runtime": runtime.get("packages", {}),
                },
            )
        )
    return pages
