"""RAG probe and CLI tests (EX-02): deterministic probe evaluation, probe
impact of extraction failure, and CLI exit codes for validate/run/compare."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_extraction.__main__ import main
from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.probes import (
    build_index,
    embed,
    evaluate_probes,
    load_probes,
    rank,
)
from benchmarks.pdf_extraction.providers import PyPDFProvider
from benchmarks.pdf_extraction.report import derive_gates, write_report
from benchmarks.pdf_extraction.runner import run_benchmark

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "pdf_extraction" / "fixtures"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(FIXTURES / "manifest.jsonl")


@pytest.fixture(scope="module")
def probes():
    return load_probes(FIXTURES / "probes.json")


class TestEmbeddingDeterminism:
    def test_embedding_is_stable(self):
        assert embed("invoice total 1,250.00") == embed("invoice total 1,250.00")

    def test_ranking_is_deterministic(self, manifest):
        index = build_index(manifest, FIXTURES, PyPDFProvider())
        first = [c.chunk_id for c in rank(index, "invoice total amount due", 5)]
        second = [c.chunk_id for c in rank(index, "invoice total amount due", 5)]
        assert first == second


class TestProbes:
    def test_probes_deterministic_replay(self, manifest, probes):
        first = evaluate_probes(manifest, FIXTURES, PyPDFProvider(), probes)
        second = evaluate_probes(manifest, FIXTURES, PyPDFProvider(), probes)
        assert first == second

    def test_baseline_probe_quality(self, manifest, probes):
        result = evaluate_probes(manifest, FIXTURES, PyPDFProvider(), probes)
        assert result["recall_at_k"] == 1.0
        assert result["mrr"] == 1.0
        assert result["citation_resolution"] == 1.0

    def test_extraction_failure_breaks_citations(self, manifest, probes):
        """When a provider stops recovering a region, its citation must vanish."""

        class BlindProvider(PyPDFProvider):
            id = "blind"
            version = "1"

            def extract(self, data: bytes, *, name: str = ""):
                pages = super().extract(data, name=name)
                if name == "ruled-invoice":
                    for page in pages:
                        page.text = ""
                        page.status = "empty"
                return pages

        result = evaluate_probes(manifest, FIXTURES, BlindProvider(), probes)
        invoice = next(d for d in result["details"] if d["probe_id"] == "invoice-total")
        assert invoice["resolvable"] is False
        assert result["citation_resolution"] < 1.0


class TestCLI:
    def test_validate_passes(self, capsys):
        assert main(["validate"]) == 0
        assert "all checks passed" in capsys.readouterr().out

    def test_run_gates_against_baseline(self, tmp_path, manifest):
        out = tmp_path / "candidate.json"
        baseline = tmp_path / "baseline.json"
        report = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        report["gates"] = derive_gates(report)
        write_report(report, baseline)
        assert main(["run", "--out", str(out), "--baseline", str(baseline)]) == 0

    def test_compare_rejects_tampered_candidate(self, tmp_path, manifest):
        report = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        report["gates"] = derive_gates(report)
        baseline = tmp_path / "baseline.json"
        write_report(report, baseline)
        report["totals"]["region_recall"] = 0.0
        candidate = tmp_path / "candidate.json"
        write_report(report, candidate)
        assert main(["compare", "--baseline", str(baseline), "--candidate", str(candidate)]) == 1

    def test_run_prints_no_pdf_contents(self, manifest, capsys):
        main(["run"])
        out = capsys.readouterr().out
        assert "Harbor Logistics" not in out
        assert "INVOICE" not in out
