"""Additional v3 reference contracts; legacy v2 manifests stay readable."""

from __future__ import annotations

import hashlib

from benchmarks.quality_metrics import TableParser, validate_bbox
from benchmarks.semantic_metrics import numeric_value


def validate_references(document: dict) -> None:
    if not isinstance(document.get("license"), str) or not document["license"].strip():
        raise ValueError("A v3 document needs source license provenance")
    expected = "\n".join(page["expected_text"] for page in document["pages"])
    if (
        document.get("annotation_sha256")
        and hashlib.sha256(expected.encode()).hexdigest() != document["annotation_sha256"]
    ):
        raise ValueError("Reference annotation checksum mismatch")
    for page in document["pages"]:
        if not isinstance(page["expected_text"], str):
            raise ValueError("Invalid reference transcription")
        regions = {region["id"]: region for region in page["regions"]}
        order = page.get("reading_order", list(regions))
        if len(order) != len(set(order)) or set(order) != set(regions):
            raise ValueError("Invalid reference reading order")
        for link in page.get("caption_links", []):
            if link["caption_id"] not in regions or link["target_id"] not in regions:
                raise ValueError("Caption link references an unknown region")
            if regions[link["caption_id"]]["category"] != "Caption":
                raise ValueError("Caption link does not identify a caption")
        field_ids = set()
        for field in page.get("critical_fields", []):
            if not isinstance(field.get("value"), str):
                raise ValueError("Critical references use explicit canonical strings")
            if field["id"] in field_ids:
                raise ValueError("Duplicate critical field id")
            field_ids.add(field["id"])
            if field["kind"] not in {"number", "text", "formula", "chart_value"}:
                raise ValueError("Unknown critical field kind")
            if field["kind"] in {"number", "chart_value"}:
                numeric_value(field["value"])
                if numeric_value(field.get("tolerance_abs", "0")) < 0:
                    raise ValueError("Negative numeric tolerance")
            elif field["kind"] == "formula":
                from benchmarks.semantic_metrics import _expression  # noqa: PLC0415

                _expression(field["value"])
            if field.get("region_id") and field["region_id"] not in regions:
                raise ValueError("Critical field references unknown region")
        for region in regions.values():
            if region["category"] == "Table":
                table = TableParser(region["html"])
                if "cells" in region:
                    expected_cells = {
                        (r, c): (cell.text, cell.rowspan, cell.colspan)
                        for (r, c), cell in table.grid().items()
                    }
                    actual_cells = {}
                    for cell in region["cells"]:
                        if any(
                            type(cell.get(key)) is not int
                            for key in ["row", "col", "rowspan", "colspan"]
                        ):
                            raise ValueError("Table-cell coordinates/spans must be integers")
                        position = (cell["row"], cell["col"])
                        if position in actual_cells:
                            raise ValueError("Duplicate table-cell coordinate")
                        actual_cells[position] = (cell["text"], cell["rowspan"], cell["colspan"])
                    if actual_cells != expected_cells:
                        raise ValueError("Table-cell coordinates/spans disagree with HTML")


def validate_prediction(page) -> None:
    if not isinstance(page.text, str) or not isinstance(page.status, str):
        raise ValueError("Invalid provider text or status")
    for collection in [page.words, page.tables, page.regions]:
        if collection is not None:
            if not isinstance(collection, list):
                raise ValueError("Invalid provider region collection")
            for entry in collection:
                if not isinstance(entry, dict) or "bbox" not in entry:
                    raise ValueError("Missing prediction bounding box")
                validate_bbox(entry["bbox"])
    if page.semantic_fields is not None and not isinstance(page.semantic_fields, dict):
        raise ValueError("Invalid provider semantic field mapping")
