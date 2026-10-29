"""Hand-computed metrics and real synthetic-PDF omission gates for EX-02."""

from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from benchmarks.pdf_extraction import (
    edit_distance,
    load_manifest,
    run_benchmark,
    score_pages,
    text_error_rate,
)
from ragnor.ingestion.extractors import PDFExtractor, PageExtractionOutcome, PageExtractionStatus


@pytest.fixture
def smoke_manifest(tmp_path):
    from benchmarks.smoke_corpus import create_smoke_corpus

    return create_smoke_corpus(tmp_path / "smoke")


def test_hand_computed_text_errors():
    assert edit_distance("cat", "cut") == 1
    assert text_error_rate("cat", "cut") == pytest.approx(1 / 3)
    assert text_error_rate(["one", "two"], ["one"]) == 0.5
    assert text_error_rate("", "") == 0
    assert text_error_rate("", "unexpected") == 1
    assert text_error_rate("abc", "") == 1


def test_fixed_corpus_replays_with_stable_quality_and_identity(smoke_manifest):
    first, second = run_benchmark(smoke_manifest), run_benchmark(smoke_manifest)
    assert first["synthetic_gate_passed"]
    assert first["summary"]["pages"] == 12
    assert first["summary"] == second["summary"]
    assert first["pages"] == second["pages"]
    assert first["run_id"] == second["run_id"]
    assert first["annotation_review"] == "synthetic-only; pending human review"
    assert "table_teds" in first["unmeasured"]
    assert first["performance"]["cost_per_1000_pages"] is None
    assert "First page" not in json.dumps(first)


def test_omission_and_false_blank_rejected(smoke_manifest):
    class Omitting(PDFExtractor):
        def extract_outcomes(self, data):
            return super().extract_outcomes(data)[:1]

    report = run_benchmark(smoke_manifest, provider=Omitting())
    assert not report["synthetic_gate_passed"]
    assert report["summary"]["missing_pages"] > 0

    class FalseBlank(PDFExtractor):
        def extract_outcomes(self, data):
            return [
                PageExtractionOutcome(row.page, PageExtractionStatus.VERIFIED_BLANK)
                for row in super().extract_outcomes(data)
            ]

    report = run_benchmark(smoke_manifest, provider=FalseBlank())
    assert not report["synthetic_gate_passed"]
    assert report["summary"]["unsafe_complete_pages"] > 0


def test_provider_failure_is_sanitized_and_fails_gate(smoke_manifest):
    def fail(data):
        raise RuntimeError("private PDF text https://signed.invalid/?token=secret")

    provider = SimpleNamespace(id="failure", version="1", options={}, extract_outcomes=fail)
    report = run_benchmark(smoke_manifest, provider=provider)
    assert not report["synthetic_gate_passed"]
    assert len(report["failures"]) == 5
    assert "secret" not in json.dumps(report)
    assert "signed.invalid" not in json.dumps(report)


def test_provider_duplicate_and_extra_pages_fail():
    row = PageExtractionOutcome(1, PageExtractionStatus.EXTRACTED, "hi")
    with pytest.raises(ValueError, match="duplicate"):
        score_pages([], [row, row])
    result = score_pages([], [row])
    assert result[0]["unexpected"]
    assert not result[0]["status_match"]


@pytest.mark.parametrize(
    "defect", ["duplicate_id", "missing_page", "checksum", "escape", "license"]
)
def test_invalid_corpus_rejected(tmp_path, defect, smoke_manifest):
    manifest = deepcopy(load_manifest(smoke_manifest))
    for doc in manifest["documents"]:
        (tmp_path / doc["path"]).write_bytes((smoke_manifest.parent / doc["path"]).read_bytes())
    doc = manifest["documents"][0]
    if defect == "duplicate_id":
        manifest["documents"].append(deepcopy(doc))
    elif defect == "missing_page":
        doc["pages"][0]["page"] = 2
    elif defect == "checksum":
        doc["source_hash"] = "wrong"
    elif defect == "escape":
        doc["path"] = "../outside.pdf"
    else:
        doc["license"] = ""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_public_corpus_is_report_only_and_never_claims_a_synthetic_pass(tmp_path, smoke_manifest):
    manifest = deepcopy(load_manifest(smoke_manifest))
    manifest["quality_gate"] = "report_only"
    manifest["text_normalization"] = "whitespace"
    for doc in manifest["documents"]:
        (tmp_path / doc["path"]).write_bytes((smoke_manifest.parent / doc["path"]).read_bytes())
        for page in doc["pages"]:
            del page["expected_status"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    report = run_benchmark(path)
    assert report["synthetic_gate_passed"] is None
    assert report["quality_gate"] == "report_only"
    assert report["summary"]["status_unmeasured_pages"] == 12
    assert report["summary"]["status_mismatches"] == 0


def test_public_fetch_verifies_existing_bytes_without_network(tmp_path, monkeypatch):
    from benchmarks.fetch_public_pdf_corpus import _download
    import hashlib

    path = tmp_path / "fixture.pdf"
    data = b"fixture"
    path.write_bytes(data)
    monkeypatch.setattr(
        "benchmarks.fetch_public_pdf_corpus.urlopen", lambda *a, **k: pytest.fail("network")
    )
    checksum = hashlib.sha256(data).hexdigest()
    assert _download("https://unused.invalid", path, checksum, 20) == data
    with pytest.raises(ValueError, match="checksum"):
        _download("https://unused.invalid", path, "wrong", 20)
    with pytest.raises(ValueError, match="size"):
        _download("https://unused.invalid", path, checksum, 1)


def test_public_reference_conversion_preserves_table_values():
    from benchmarks.fetch_public_pdf_corpus import reference_text

    elements = [{"content": {"html": "<table><tr><td>42</td></tr></table>"}}]
    assert reference_text(elements) == "42"


class _SlowPDFProvider(PDFExtractor):
    def extract_outcomes(self, data):
        import time

        time.sleep(30)
        return []


def test_provider_process_deadline_terminates_worker():
    from benchmarks.pdf_extraction import ProviderTimeoutError, _extract
    from time import monotonic

    started = monotonic()
    with pytest.raises(ProviderTimeoutError):
        _extract(_SlowPDFProvider(), b"unused", 0.1)
    assert monotonic() - started < 5


def test_provider_subprocess_returns_typed_outcomes(smoke_manifest):
    from benchmarks.pdf_extraction import _extract

    data = (smoke_manifest.parent / "digital.pdf").read_bytes()
    outcomes = _extract(PDFExtractor(), data, 10)
    assert outcomes[0].text == "First page"
    assert outcomes[0].status == "extracted"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_provider_deadline_rejected_before_run(timeout):
    with pytest.raises(ValueError, match="timeout"):
        run_benchmark(timeout_seconds=timeout)
