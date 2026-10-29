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
from benchmarks.coverage_metrics import region_detection
from benchmarks.semantic_metrics import score_fields
from benchmarks.reference_schema import validate_prediction, validate_references
from benchmarks.cost_accounting import cost_report


def validate_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] not in {2, 3} or not manifest["documents"]:
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
        if manifest["schema_version"] == 3:
            validate_references(doc)
    probe_ids, queries = set(), set()
    pages = {doc["document_id"]: len(doc["pages"]) for doc in manifest["documents"]}
    for probe in manifest.get("qa_probes", []):
        from benchmarks.rag_probes import search_key  # noqa: PLC0415

        if any(
            not isinstance(probe.get(field), str) or not probe[field].strip()
            for field in ["id", "query", "answer"]
        ):
            raise ValueError("Invalid curated QA reference")
        if probe["id"] in probe_ids or search_key(probe["query"]) in queries:
            raise ValueError("Duplicate curated QA reference")
        probe_ids.add(probe["id"])
        queries.add(search_key(probe["query"]))
        if type(probe.get("page")) is not int or not 1 <= probe["page"] <= pages.get(
            probe.get("document_id"), 0
        ):
            raise ValueError("Curated QA references an unknown source page")
    return manifest


def score_page(  # noqa: PLR0915 -- separate observable page metrics and bounded adapters
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
    layout_supported = layout_supported and reference.get("geometry_ground_truth", True)
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
        by_id = {region["id"]: region for region in reference["regions"]}
        ordered = [by_id[identity] for identity in reference.get("reading_order", list(by_id))]
        scores.update(layout_scores(ordered, (actual.words or []) if actual else []))
        scores.update(region_detection(reference["regions"], actual.regions if actual else []))
    semantic = (
        dict(actual.semantic_fields) if actual and actual.semantic_fields is not None else None
    )
    scores.update(actual.raster_metrics if actual else {})
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
        for field in reference.get("critical_fields", []):
            table_id = field.get("table_region_id")
            if table_id is None:
                continue
            semantic = semantic if semantic is not None else {}
            ref_index = next(
                (i for i, table in enumerate(reference_tables) if table["id"] == table_id), None
            )
            found = next((j for i, j in matches if i == ref_index), None)
            if found is not None:
                try:
                    cell = (
                        TableParser(actual_tables[found]["html"])
                        .grid()
                        .get((field["row"], field["col"]))
                    )
                    if cell is not None:
                        semantic[field["id"]] = cell.text
                except (ValueError, KeyError):
                    pass  # Invalid tables already count as failed predictions above.
    scores.update(score_fields(reference.get("critical_fields", []), semantic))
    # A completeness claim must not hide an observable empty required region/page/table/field.
    unsafe = bool(
        scores.get("empty_text_regions", 0)
        or scores.get("field_failures")
        or (reference["expected_text"].strip() and (actual is None or not actual.text.strip()))
        or (tables_supported and scores.get("matched_tables", 0) < scores["reference_tables"])
    )
    scores["completeness_claim"] = actual.verified_complete if actual else None
    scores["unsafe_complete"] = bool(actual and actual.verified_complete is True and unsafe)
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
        "mean_region_precision": mean("region_precision", layout),
        "mean_region_recall": mean("region_recall", layout),
        "reference_region_count": sum(row.get("reference_region_count", 0) for row in layout),
        "predicted_region_count": sum(row.get("predicted_region_count") or 0 for row in layout),
        "mean_word_box_ink_coverage": mean("word_box_ink_coverage"),
        "reference_fields": sum(row.get("reference_fields", 0) for row in rows),
        "mean_field_task_accuracy": mean("field_task_accuracy"),
        "mean_field_answer_coverage": mean("field_answer_coverage"),
        "completeness_claims": sum(row.get("completeness_claim") is True for row in rows),
        "unsafe_complete_pages": sum(row.get("unsafe_complete", False) for row in rows),
        "observed_verification_false_negative_rate": (
            sum(row.get("unsafe_complete", False) for row in rows)
            / sum(row.get("completeness_claim") is True for row in rows)
            if any(row.get("completeness_claim") is True for row in rows)
            else None
        ),
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


def run(  # noqa: PLR0913 -- explicit benchmark composition and host options
    path: Path,
    provider_name: str,
    *,
    timeout=60.0,
    cohort="all",
    provider_instance=None,
    rag=True,
    rag_limit=40,
    cpu_usd_per_hour=None,
    pricing_source=None,
    predictions_out: Path | None = None,
) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    manifest = validate_manifest(path)
    provider = provider_instance or PROVIDERS[provider_name]()
    selected = manifest["documents"]
    if cohort == "common":
        selected = [
            doc
            for doc in selected
            if doc["family"] != "public" or doc["document_id"] in manifest["stress_sources"]
        ]
    rows, failures, durations, peaks, ocr_peaks = [], [], [], [], []
    predictions, cpu_times = {}, []
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
            predicted = by_page.get(reference["page"])
            if predicted is not None:
                try:
                    validate_prediction(predicted)
                except ValueError:
                    failures.append(
                        {
                            "document_id": doc["document_id"],
                            "page": reference["page"],
                            "reason": "invalid_prediction",
                        }
                    )
                    predicted = None
            result = score_page(
                reference,
                predicted,
                layout_supported=provider.supports_layout,
                tables_supported=provider.supports_tables,
            )
            rows.append({"document_id": doc["document_id"], "family": doc["family"], **result})
            if predicted is not None:
                predictions[doc["document_id"], reference["page"]] = predicted
        cpu_times.extend(page.cpu_seconds for page in actual if page.cpu_seconds is not None)
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
        "rank-bm25",
        "numpy",
        "sympy",
        "pyarrow",
        "markdown-it-py",
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
        "retrieval": {"enabled": rag, "limit": rag_limit, "k": 5, "mode": "SDK BM25Plus"},
        "pricing": {"cpu_usd_per_hour": cpu_usd_per_hour, "source": pricing_source},
    }
    if predictions_out is not None:
        from dataclasses import asdict  # noqa: PLC0415

        predictions_out.write_text(
            json.dumps(
                [
                    {"document_id": doc, "page": page, **asdict(predicted)}
                    for (doc, page), predicted in predictions.items()
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    from benchmarks.rag_probes import run_rag  # noqa: PLC0415

    result = {
        "schema_version": 3,
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
            "document_p95_seconds": sorted(durations)[math.ceil(len(durations) * 0.95) - 1],
            "provider_python_peak_rss_bytes": max(peaks, default=None),
            "ocr_process_peak_rss_bytes": max(ocr_peaks, default=None),
            "cost_per_1000_pages": None,
        },
        "cost": cost_report(
            sum(cpu_times) if len(cpu_times) == len(selected) else None,
            len(rows),
            cpu_usd_per_hour=cpu_usd_per_hour,
            pricing_source=pricing_source,
        ),
        "retrieval": run_rag({**manifest, "documents": selected}, predictions, limit=rag_limit)
        if rag
        else None,
        "qualification": "experimental; no production release threshold or complete human signoff",
        "unmeasured": [
            "chart/formula semantics",
            "population-wide scan/language/handwriting generalization beyond the selected corpus",
            "downstream semantic embedding/answer models (lexical SDK path is measured)",
            "host billing rate unless supplied explicitly",
        ],
    }
    result["performance"]["cost_per_1000_pages"] = result["cost"]["total_usd_per_1000_pages"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--cohort", choices=["all", "common"], default="all")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--rag-limit", type=int, default=40)
    parser.add_argument("--cpu-usd-per-hour", type=float)
    parser.add_argument("--pricing-source")
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--tesseract-language", default="eng")
    parser.add_argument("--tessdata-dir")
    args = parser.parse_args()
    from benchmarks.quality_providers import TesseractProvider  # noqa: PLC0415

    provider = (
        TesseractProvider(language=args.tesseract_language, tessdata_dir=args.tessdata_dir)
        if args.provider == "tesseract"
        else None
    )
    result = run(
        args.manifest,
        args.provider,
        timeout=args.timeout,
        cohort=args.cohort,
        provider_instance=provider,
        rag=not args.no_rag,
        rag_limit=args.rag_limit,
        cpu_usd_per_hour=args.cpu_usd_per_hour,
        pricing_source=args.pricing_source,
        predictions_out=args.predictions_out,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"documents": result["documents"], "failures": len(result["failures"])}))


if __name__ == "__main__":
    main()
