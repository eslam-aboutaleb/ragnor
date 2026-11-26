"""Non-regression gate tests (EX-02): a baseline passes its own gates, a
candidate with intentionally degraded table extraction is rejected, and an
unsafe success on a corrupted input is rejected. Gate thresholds come from
the measured baseline, never invented."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_extraction.manifest import Table, TableCell, load_manifest
from benchmarks.pdf_extraction.providers import ProviderPage, PyPDFProvider
from benchmarks.pdf_extraction.report import (
    compare_reports,
    derive_gates,
    load_report,
)
from benchmarks.pdf_extraction.runner import run_benchmark

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "pdf_extraction" / "fixtures"
BASELINE = FIXTURES / "baseline" / "pypdf.json"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(FIXTURES / "manifest.jsonl")


@pytest.fixture(scope="module")
def baseline_report(manifest):
    report = run_benchmark(manifest, FIXTURES, PyPDFProvider())
    report["gates"] = derive_gates(report)
    return report


class StructureAwareProvider(PyPDFProvider):
    """Test double: pypdf text plus the exact annotated table structure."""

    id = "structure-aware"
    version = "1"

    def __init__(self, manifest):
        super().__init__()
        self._manifest = manifest

    def extract(self, data: bytes, *, name: str = ""):
        pages = super().extract(data, name=name)
        tables_by_page = {
            (p.document_id, p.page): p.tables for p in self._manifest.pages if p.tables
        }
        for page in pages:
            key = (name, page.page)
            if key in tables_by_page:
                page.tables = tables_by_page[key]
        return pages


class DegradedTableProvider:
    """Test double: pypdf text plus intentionally wrong table structure
    (every merged cell split), simulating a merged-table regression."""

    id = "degraded-tables"
    version = "1"
    config: dict = {}

    def __init__(self, manifest):
        self._manifest = manifest
        self._delegate = PyPDFProvider()

    def extract(self, data: bytes, *, name: str = ""):
        pages = self._delegate.extract(data, name=name)
        tables_by_page = {
            (p.document_id, p.page): p.tables for p in self._manifest.pages if p.tables
        }
        for page in pages:
            key = (name, page.page)
            if key in tables_by_page:
                page.tables = [_split_merged(t) for t in tables_by_page[key]]
        return pages

    def library_versions(self):
        return {"double": "1"}


def _split_merged(table: Table) -> Table:
    """Replace every spanned cell with single cells (a classic regression)."""
    cells = []
    for cell in table.cells:
        cells.append(TableCell(cell.row, cell.col, cell.text))
        for extra in range(1, cell.colspan):
            cells.append(TableCell(cell.row, cell.col + extra, ""))
        for extra in range(1, cell.rowspan):
            cells.append(TableCell(cell.row + extra, cell.col, ""))
    return Table(table.region_id, table.rows, table.cols, cells)


class TestBaselineGates:
    def test_baseline_passes_its_own_gates(self, baseline_report, manifest):
        candidate = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        outcome = compare_reports(baseline_report, candidate)
        assert outcome["passed"], outcome["failures"]

    def test_checked_in_baseline_matches_current_run(self, baseline_report, manifest):
        """The committed baseline file must equal a fresh deterministic run."""
        on_disk = load_report(BASELINE)
        candidate = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        assert candidate["totals"] == on_disk["totals"]
        outcome = compare_reports(on_disk, candidate)
        assert outcome["passed"], outcome["failures"]

    def test_degraded_table_extraction_is_rejected(self, manifest):
        """A merged-table regression is rejected against a structure-aware
        baseline (pypdf emits no table structure, so its baseline cannot gate
        structural metrics; that limitation is documented)."""
        baseline = run_benchmark(manifest, FIXTURES, StructureAwareProvider(manifest))
        baseline["gates"] = derive_gates(baseline)
        assert baseline["totals"]["table_structure"] == 1.0
        candidate = run_benchmark(manifest, FIXTURES, DegradedTableProvider(manifest))
        assert candidate["totals"]["table_structure"] < 1.0
        outcome = compare_reports(baseline, candidate)
        assert not outcome["passed"]
        failed_metrics = {f["metric"] for f in outcome["failures"]}
        assert failed_metrics & {"table_structure", "merged_cell_fidelity"}

    def test_missing_metric_in_candidate_is_rejected(self, baseline_report, manifest):
        candidate = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        candidate["totals"]["region_recall"] = None
        outcome = compare_reports(baseline_report, candidate)
        assert not outcome["passed"]
        assert any(f["reason"] == "metric missing in candidate" for f in outcome["failures"])

    def test_silent_region_omission_is_rejected(self, baseline_report, manifest):
        """Drop all table-region text: the region-recall gate must catch it."""

        class OmittingProvider(PyPDFProvider):
            id = "omitting"
            version = "1"

            def extract(self, data: bytes, *, name: str = ""):
                pages = super().extract(data, name=name)
                if name in {"ruled-invoice", "borderless-financials", "split-table"}:
                    for page in pages:
                        page.text = ""
                        page.status = "empty"
                return pages

        candidate = run_benchmark(manifest, FIXTURES, OmittingProvider())
        outcome = compare_reports(baseline_report, candidate)
        assert not outcome["passed"]
        failed_metrics = {f["metric"] for f in outcome["failures"]}
        assert "region_recall" in failed_metrics

    def test_unsafe_success_is_rejected(self, baseline_report, manifest):
        class LenientProvider(PyPDFProvider):
            id = "lenient"
            version = "1"

            def extract(self, data: bytes, *, name: str = ""):
                if name in {"corrupted-report", "oversized-blank"}:
                    return [ProviderPage(page=1, text="fabricated")]
                return super().extract(data, name=name)

        candidate = run_benchmark(manifest, FIXTURES, LenientProvider())
        outcome = compare_reports(baseline_report, candidate)
        assert not outcome["passed"]
        assert any("unexpected_success" in f["metric"] for f in outcome["failures"])

    def test_thresholds_come_from_baseline(self, baseline_report):
        gates = baseline_report["gates"]["gates"]
        assert gates
        for gate in gates:
            assert gate["source"] == "baseline"
            metric = gate["metric"]
            assert gate["threshold"] == baseline_report["totals"][metric]
