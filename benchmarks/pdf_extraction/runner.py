"""Benchmark runner: execute a provider over the corpus and emit a report.

Reports are machine-readable JSON containing only ids, counts, hashes and
metric values: PDF contents are never persisted (see the payload-guard
test). Structure and metric values are deterministic for a fixed provider,
pinned library versions and corpus; timings and memory peaks are recorded
but excluded from gate comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path

from benchmarks.pdf_extraction.manifest import SCHEMA_VERSION, Manifest, PageAnnotation
from benchmarks.pdf_extraction.metrics import score_page
from benchmarks.pdf_extraction.providers import ExtractionProvider, measure
from benchmarks.pdf_extraction.report import (
    aggregate_pages,
    compare_reports,
    family_slices,
    performance_summary,
)

METRIC_NOTES = [
    "table_structure is a sequence-based TEDS approximation, not full tree-edit TEDS",
    "region_precision requires provider blocks; null for whole-page text providers",
    "merged_cell_fidelity requires provider table structure; null otherwise",
    "silent_omission flags pages reporting success while missing expected regions",
]


def run_id(manifest: Manifest, provider: ExtractionProvider) -> str:
    """Deterministic run identity: corpus, provider id/version and pinned config."""
    payload = json.dumps(
        {
            "corpus": manifest.manifest_hash(),
            "provider": provider.id,
            "version": provider.version,
            "config": provider.config,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_benchmark(
    manifest: Manifest,
    root: Path | str,
    provider: ExtractionProvider,
) -> dict:
    """Run ``provider`` over every annotated document and build the report."""
    root = Path(root)
    tracemalloc.start()
    page_rows = []
    for document_id, pages in manifest.documents().items():
        data = (root / pages[0].file).read_bytes()
        results, error = measure(provider, data, document_id)
        page_rows.extend(_score_document(pages, results, error))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _build_report(manifest, provider, page_rows, peak)


def _score_document(pages: list[PageAnnotation], results, error: str | None) -> list[dict]:
    by_page = {p.page: p for p in results} if results else {}
    rows = []
    for annotation in pages:
        result = by_page.get(annotation.page)
        rows.append(_score_one(annotation, result, error))
    return rows


def _score_one(annotation: PageAnnotation, result, error: str | None) -> dict:
    where = f"{annotation.document_id} p{annotation.page}"
    if annotation.expect_error:
        return _error_expected_row(annotation, result, error, where)
    if error is not None or result is None or result.status == "error":
        return {
            "document_id": annotation.document_id,
            "page": annotation.page,
            "family": annotation.family,
            "status": "error",
            "error": error or (result.error if result else "missing page in provider output"),
            "metrics": None,
            "duration_ms": None,
            "unexpected": False,
            "note": f"{where}: expected content but provider failed",
        }
    score = score_page(
        annotation,
        result.text,
        actual_tables=_bind_tables(annotation, result.tables),
        provider_blocks=result.blocks,
    )
    return {
        "document_id": annotation.document_id,
        "page": annotation.page,
        "family": annotation.family,
        "status": result.status,
        "error": None,
        "metrics": score.as_dict(),
        "duration_ms": round(result.duration_ms, 3),
        "cost_usd": result.cost_usd,
        "unexpected": False,
    }


def _bind_tables(annotation: PageAnnotation, tables):
    """Give unlabeled provider tables the region id of the expected table at the same rank.

    Providers cannot know annotation ids. Tables and expected regions are both
    ordered top to bottom, then left to right; surplus tables stay unmatched.
    """
    if not tables or all(t.region_id for t in tables):
        return tables
    boxes = {r.region_id: r.bbox for r in annotation.regions}
    expected = sorted(
        annotation.tables,
        key=lambda t: (boxes[t.region_id].y0, boxes[t.region_id].x0)
        if t.region_id in boxes
        else (0, 0),
    )
    return [
        replace(table, region_id=expected[i].region_id) if i < len(expected) else table
        for i, table in enumerate(tables)
    ]


def _error_expected_row(annotation: PageAnnotation, result, error: str | None, where: str) -> dict:
    failed = error is not None or result is None or result.status == "error"
    return {
        "document_id": annotation.document_id,
        "page": annotation.page,
        "family": annotation.family,
        "status": "expected-error" if failed else "unexpected-success",
        "error": error if error is not None else (result.error if result else None),
        "metrics": None,
        "duration_ms": None,
        "unexpected": not failed,
        "note": None if failed else f"{where}: unsafe success on input expected to fail",
    }


def _build_report(
    manifest: Manifest, provider: ExtractionProvider, page_rows: list[dict], peak: int
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id(manifest, provider),
        "python": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "platform": platform.platform(),
        "hardware": {"machine": platform.machine(), "cpu_count": os.cpu_count()},
        "provider": {
            "id": provider.id,
            "version": provider.version,
            "config": provider.config,
            "library_versions": provider.library_versions(),
        },
        "corpus": {
            "version": manifest.corpus_version,
            "manifest_hash": manifest.manifest_hash(),
            "documents": len(manifest.documents()),
            "pages": len(manifest.pages),
        },
        "totals": aggregate_pages(page_rows),
        "families": family_slices(page_rows),
        "performance": performance_summary(page_rows, peak),
        "pages": page_rows,
        "failures": [p for p in page_rows if p["status"] == "error"],
        "metric_notes": METRIC_NOTES,
    }


__all__ = ["METRIC_NOTES", "compare_reports", "run_benchmark", "run_id"]
