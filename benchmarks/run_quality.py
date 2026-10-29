"""Compare offline extractors on text, regions, order, table cells and stress cases.

Outputs contain IDs and scores only. Corpus, annotations and JSON reports stay
in the host-selected data directory; only an aggregate Markdown summary is shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.pdf_extraction import _extract
from benchmarks.quality_metrics import (
    TableParser,
    numeric_tokens,
    layout_scores,
    match_boxes,
    page_text_scores,
    table_scores,
    validate_bbox,
)
from benchmarks.quality_providers import PROVIDERS, QualityPage


def validate_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2 or not manifest["documents"]:
        raise ValueError("Invalid quality manifest")
    ids = set()
    for doc in manifest["documents"]:
        if doc["document_id"] in ids:
            raise ValueError("Duplicate document ID")
        ids.add(doc["document_id"])
        source = (path.parent / doc["path"]).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("Source path escapes corpus")
        if hashlib.sha256(source.read_bytes()).hexdigest() != doc["source_hash"]:
            raise ValueError("Source checksum mismatch")
        if [p["page"] for p in doc["pages"]] != list(range(1, len(doc["pages"]) + 1)):
            raise ValueError("Missing, duplicate or reordered reference pages")
        for page in doc["pages"]:
            region_ids = set()
            for region in page["regions"]:
                if region["id"] in region_ids:
                    raise ValueError("Duplicate region ID")
                region_ids.add(region["id"])
                validate_bbox(region["bbox"])
    return manifest


def score_page(
    reference: dict,
    actual: QualityPage | None,
    *,
    layout_supported: bool | None = None,
    tables_supported: bool | None = None,
) -> dict[str, Any]:
    if layout_supported is None:
        layout_supported = actual is not None and actual.words is not None
    if tables_supported is None:
        tables_supported = actual is not None and actual.tables is not None
    scores = page_text_scores(reference["expected_text"], actual.text if actual else "")
    scores.update(
        {
            "page": reference["page"],
            "missing_page": actual is None,
            "status": actual.status if actual else "missing",
            "layout_supported": layout_supported,
            "tables_supported": tables_supported,
        }
    )
    if layout_supported:
        scores.update(layout_scores(reference["regions"], (actual.words or []) if actual else []))
    reference_tables = [r for r in reference["regions"] if r["category"] == "Table"]
    scores["reference_tables"] = len(reference_tables)
    if tables_supported:
        actual_tables = (actual.tables or []) if actual else []
        matches = match_boxes(reference_tables, actual_tables, threshold=0.5)
        scores.update(
            {
                "matched_tables": len(matches),
                "predicted_tables": len(actual_tables),
                "table_matching_iou": 0.5,
            }
        )
        values = []
        for i, table in enumerate(reference_tables):
            found = next((j for ref, j in matches if ref == i), None)
            zero = {
                "teds_normalized": 0.0,
                "teds_structure": 0.0,
                "cell_exact_accuracy": 0.0,
                "span_accuracy": 0.0,
                "numeric_cell_accuracy": None,
            }
            try:
                parsed = TableParser(table["html"])
                if any(numeric_tokens(cell.text) for row in parsed.rows for cell in row):
                    zero["numeric_cell_accuracy"] = 0.0
            except (ValueError, KeyError):
                scores["invalid_table_references"] = scores.get("invalid_table_references", 0) + 1
                continue
            try:
                # Missing/invalid predictions get zero; failed pages stay in the denominator.
                if found is None:
                    values.append(zero)
                else:
                    values.append(table_scores(table["html"], actual_tables[found]["html"]))
            except (ValueError, KeyError):
                values.append(zero)
                scores["invalid_table_predictions"] = scores.get("invalid_table_predictions", 0) + 1
        scores["table_metrics"] = values
    return scores


def aggregate(rows: list[dict]) -> dict:
    def mean(key, source=rows):
        values = [row[key] for row in source if row.get(key) is not None]
        return statistics.mean(values) if values else None

    tables = [table for row in rows for table in row.get("table_metrics", [])]
    layout = [row for row in rows if row.get("layout_supported")]
    table_rows = [row for row in rows if row.get("tables_supported")]
    reference_count = sum(row["reference_tables"] for row in table_rows)
    predicted_count = sum(row.get("predicted_tables", 0) for row in table_rows)
    matched = sum(row.get("matched_tables", 0) for row in table_rows)
    return {
        "pages": len(rows),
        "missing_pages": sum(row["missing_page"] for row in rows),
        "mean_page_cer": mean("cer"),
        "mean_page_wer": mean("wer"),
        "mean_token_recall": mean("token_recall"),
        "mean_numeric_token_recall": mean("numeric_token_recall"),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "layout_scored_pages": len(layout),
        "table_capable_pages": len(table_rows),
        "mean_region_token_recall": mean("region_token_recall", layout),
        "empty_text_regions": sum(row.get("empty_text_regions", 0) for row in layout),
        "reference_text_regions": sum(row.get("text_regions", 0) for row in layout),
        "mean_reading_order_pair_accuracy": mean("reading_order_pair_accuracy", layout),
        "mean_reading_order_pair_coverage": mean("reading_order_pair_coverage", layout),
        "reference_tables": sum(row["reference_tables"] for row in rows),
        "tables_scored": len(tables),
        "table_recall_iou50": matched / reference_count if reference_count else None,
        "table_precision_iou50": matched / predicted_count if predicted_count else None,
        "mean_teds_normalized": mean("teds_normalized", tables),
        "mean_teds_structure": mean("teds_structure", tables),
        "mean_cell_exact_accuracy": mean("cell_exact_accuracy", tables),
        "mean_span_accuracy": mean("span_accuracy", tables),
        "mean_numeric_cell_accuracy": mean("numeric_cell_accuracy", tables),
        "invalid_table_references": sum(row.get("invalid_table_references", 0) for row in rows),
        "invalid_table_predictions": sum(row.get("invalid_table_predictions", 0) for row in rows),
    }


def run(path: Path, provider_name: str, *, timeout=60.0, cohort="all") -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    manifest = validate_manifest(path)
    provider = PROVIDERS[provider_name]()
    selected = manifest["documents"]
    if cohort == "common":
        selected = [
            doc
            for doc in selected
            if doc["family"] != "public" or doc["document_id"] in manifest["stress_sources"]
        ]
    rows, failures, durations, peaks, ocr_peaks = [], [], [], [], []
    for doc in selected:
        started = perf_counter()
        try:
            actual = _extract(provider, (path.parent / doc["path"]).read_bytes(), timeout)
            numbers = [page.page for page in actual]
            if sorted(numbers) != list(range(1, len(doc["pages"]) + 1)):
                raise ValueError("Provider returned invalid page numbering")
        except Exception as exc:
            # Only fixed codes are serialized; no provider messages or data.
            actual = []
            failures.append(
                {
                    "document_id": doc["document_id"],
                    "reason": "timeout"
                    if type(exc).__name__ == "ProviderTimeoutError"
                    else "provider_failure",
                }
            )
        durations.append(perf_counter() - started)
        by_page = {page.page: page for page in actual}
        for reference in doc["pages"]:
            result = score_page(
                reference,
                by_page.get(reference["page"]),
                layout_supported=provider.supports_layout,
                tables_supported=provider.supports_tables,
            )
            rows.append({"document_id": doc["document_id"], "family": doc["family"], **result})
        peaks.extend(page.python_peak_rss_bytes for page in actual)
        ocr_peaks.extend(
            page.ocr_peak_rss_bytes for page in actual if page.ocr_peak_rss_bytes is not None
        )
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "completed_documents": len(durations),
                    "total_documents": len(selected),
                }
            ),
            flush=True,
        )
    versions = {}
    for package in [
        "pypdf",
        "pdfplumber",
        "pdfminer.six",
        "pypdfium2",
        "pillow",
        "fonttools",
        "apted",
        "rapidfuzz",
    ]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    identity = {
        "provider": provider.id,
        "version": provider.version,
        "options": provider.options,
        "versions": versions,
        "cohort": cohort,
        "timeout": timeout,
    }
    return {
        "schema_version": 2,
        "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": hashlib.sha256(
            json.dumps(
                [identity, hashlib.sha256(path.read_bytes()).hexdigest()], sort_keys=True
            ).encode()
        ).hexdigest(),
        "configuration": identity,
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "documents": len(selected),
        "summary": aggregate(rows),
        "families": {
            family: aggregate([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
        "failures": failures,
        "pages": rows,
        "performance": {
            "document_p50_seconds": statistics.median(durations),
            "document_p95_seconds": sorted(durations)[max(0, int(len(durations) * 0.95) - 1)],
            "provider_python_peak_rss_bytes": max(peaks, default=None),
            "ocr_process_peak_rss_bytes": max(ocr_peaks, default=None),
            "cost_per_1000_pages": None,
        },
        "qualification": "experimental; no production release threshold or complete human signoff",
        "unmeasured": [
            "chart/formula semantics",
            "real scan degradation distribution",
            "Arabic/RTL",
            "handwriting",
            "downstream embedding retrieval",
            "authenticated citation resolution",
            "provider cost",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--cohort", choices=["all", "common"], default="all")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.provider, timeout=args.timeout, cohort=args.cohort)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"documents": result["documents"], "failures": len(result["failures"])}))


if __name__ == "__main__":
    main()
