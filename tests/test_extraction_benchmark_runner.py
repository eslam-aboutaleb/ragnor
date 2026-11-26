"""Runner behavior tests (EX-02): deterministic replay, injected provider
exceptions and timeouts, unsafe-success detection, and the payload guard
that keeps PDF contents out of reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.providers import ProviderPage, PyPDFProvider, registry
from benchmarks.pdf_extraction.runner import run_benchmark, run_id

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "pdf_extraction" / "fixtures"
VOLATILE_KEYS = {"performance", "python", "python_implementation", "platform", "hardware"}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(FIXTURES / "manifest.jsonl")


def stable(report: dict) -> dict:
    kept = {k: v for k, v in report.items() if k not in VOLATILE_KEYS}
    kept["pages"] = [
        {k: v for k, v in page.items() if k not in {"duration_ms", "cost_usd"}}
        for page in report["pages"]
    ]
    return kept


class RaisingProvider:
    """Test double: every document fails with the given exception type."""

    id = "raising"
    version = "1"
    config: dict = {}

    def __init__(self, exc_type=RuntimeError):
        self.exc_type = exc_type

    def extract(self, data: bytes, *, name: str = ""):
        raise self.exc_type("injected failure")

    def library_versions(self):
        return {}


class TestDeterminism:
    def test_replay_is_stable(self, manifest):
        first = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        second = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        assert stable(first) == stable(second)
        assert first["run_id"] == second["run_id"]

    def test_run_id_tracks_provider_identity(self, manifest):
        assert run_id(manifest, PyPDFProvider()) != run_id(manifest, PyPDFProvider(max_pages=5))


class TestFailureInjection:
    def test_provider_exception_records_errors_and_continues(self, manifest):
        report = run_benchmark(manifest, FIXTURES, RaisingProvider())
        assert report["totals"]["scored_pages"] == 0
        assert report["totals"]["error_pages"] == len(manifest.pages) - 2
        for failure in report["failures"]:
            assert failure["error"] == "RuntimeError"

    def test_provider_timeout_is_recorded(self, manifest):
        report = run_benchmark(manifest, FIXTURES, RaisingProvider(TimeoutError))
        assert {f["error"] for f in report["failures"]} == {"TimeoutError"}

    def test_expected_failures_are_not_errors(self, manifest):
        report = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        expected_error_docs = {"corrupted-report", "oversized-blank"}
        flagged = {p["document_id"] for p in report["pages"] if p["status"] == "expected-error"}
        assert flagged == expected_error_docs
        assert report["totals"]["unexpected_success_pages"] == 0

    def test_unsafe_success_is_flagged(self, manifest):
        """A provider that 'succeeds' on corrupted input must be flagged."""

        class LenientProvider:
            id = "lenient"
            version = "1"
            config: dict = {}

            def extract(self, data: bytes, *, name: str = ""):
                return [ProviderPage(page=1, text="fabricated text")]

            def library_versions(self):
                return {}

        report = run_benchmark(manifest, FIXTURES, LenientProvider())
        assert report["totals"]["unexpected_success_pages"] == 2


class TestPayloadGuard:
    def test_report_never_contains_pdf_contents(self, manifest):
        report = run_benchmark(manifest, FIXTURES, PyPDFProvider())
        payload = json.dumps(report)
        leaks = []
        for page in manifest.pages:
            for region in page.regions:
                expected = region.expected_text
                if len(expected) >= 12 and expected in payload:
                    leaks.append(f"{page.document_id}:{region.region_id}")
        assert leaks == []

    def test_failures_do_not_leak_exception_messages(self, manifest):
        report = run_benchmark(manifest, FIXTURES, RaisingProvider())
        payload = json.dumps(report)
        assert "injected failure" not in payload


class TestProviderRegistry:
    def test_pypdf_provider_pins_config_and_version(self):
        provider = PyPDFProvider()
        assert provider.config == {"max_pages": 1000}
        assert provider.library_versions()["pypdf"] != "not-installed"

    def test_registry_exposes_pypdf(self):
        assert "pypdf" in registry()
