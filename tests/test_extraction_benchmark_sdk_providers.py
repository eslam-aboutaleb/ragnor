"""EX-02: the fixture benchmark measures the installed SDK adapters, not only pypdf."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pdfplumber")

from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.providers import SDKProvider, registry
from benchmarks.pdf_extraction.report import compare_reports, derive_gates
from benchmarks.pdf_extraction.runner import _bind_tables, run_benchmark

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "pdf_extraction" / "fixtures"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(FIXTURES / "manifest.jsonl")


@pytest.fixture(scope="module")
def native_report(manifest):
    return run_benchmark(manifest, FIXTURES, SDKProvider(engine="native"))


def test_sdk_engines_are_registered():
    assert {"pypdf", "sdk-native", "sdk-ocr", "sdk-auto"} <= set(registry())
    with pytest.raises(ValueError):
        SDKProvider(engine="other")


def test_native_adapter_populates_table_and_region_metrics(native_report):
    totals = native_report["totals"]
    # pypdf leaves these null: only structure-aware adapters can be scored on them.
    assert totals["table_structure"] is not None
    assert totals["merged_cell_fidelity"] is not None
    assert totals["region_precision"] is not None
    assert totals["error_pages"] == 0
    assert totals["unexpected_success_pages"] == 0  # corrupted/oversized inputs still fail
    assert native_report["provider"]["id"] == "sdk-native"


def test_run_is_deterministic_apart_from_timings(manifest, native_report):
    again = run_benchmark(manifest, FIXTURES, SDKProvider(engine="native"))
    assert again["run_id"] == native_report["run_id"]
    assert again["totals"] == native_report["totals"]


def test_a_gate_derived_from_the_sdk_run_rejects_degraded_tables(manifest, native_report):
    from dataclasses import replace

    gates = derive_gates(native_report)
    baseline = {**native_report, "gates": gates}
    assert compare_reports(baseline, native_report)["passed"]

    class NoTables(SDKProvider):
        def extract(self, data, *, name=""):
            return [replace(page, tables=[]) for page in super().extract(data, name=name)]

    degraded = run_benchmark(manifest, FIXTURES, NoTables(engine="native"))
    verdict = compare_reports(baseline, degraded)
    assert not verdict["passed"]


def test_unlabeled_tables_bind_by_page_order(manifest):
    page = next(p for p in manifest.pages if len(p.tables) >= 1)
    from benchmarks.pdf_extraction.manifest import Table

    unlabeled = [Table("", 1, 1, []) for _ in page.tables] + [Table("", 1, 1, [])]
    bound = _bind_tables(page, unlabeled)
    assert [t.region_id for t in bound[: len(page.tables)]] == [
        t.region_id
        for t in sorted(
            page.tables,
            key=lambda t: next(
                (r.bbox.y0, r.bbox.x0) for r in page.regions if r.region_id == t.region_id
            ),
        )
    ]
    assert bound[-1].region_id == ""  # surplus table stays unmatched


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs host Tesseract")
def test_auto_adapter_runs_the_whole_corpus_with_real_ocr(manifest):
    report = run_benchmark(manifest, FIXTURES, SDKProvider(engine="auto"))
    assert report["totals"]["error_pages"] == 0
    assert report["totals"]["unexpected_success_pages"] == 0
    # Scanned pages are readable only through OCR; native-only cannot match this.
    assert report["totals"]["region_recall"] > 0.8
