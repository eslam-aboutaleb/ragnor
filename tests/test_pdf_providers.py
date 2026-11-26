"""Client-side PDF provider adapters: verification gate, identity and connector contracts."""

from __future__ import annotations

import json
import shutil
import sys

import pytest

from ragnor.ingestion import (
    PDFExtractionError,
    PDFExtractionLimits,
    TesseractPDFExtractor,
)
from ragnor.ingestion.connector_core import extract_documents
from ragnor.ingestion.extractors import PageExtractionStatus
from tests.pdf_fixtures import AcceptFixture, mixed_pdf, scanned_pdf
from tests.pdf_fixtures import fake_ocr as fixture_fake_ocr

fake_ocr = fixture_fake_ocr


def ocr_adapter(fake_ocr, **kwargs):
    command, models = fake_ocr
    return TesseractPDFExtractor(executable=str(command), tessdata_dir=models, **kwargs)


def test_strict_connector_rejects_unverified_ocr_but_keeps_review_text(fake_ocr):
    result = extract_documents(mixed_pdf(), "book.pdf", "book", {}, {}, ocr_adapter(fake_ocr))
    assert not result.authoritative and not result
    # The explicit OCR engine re-extracts every page, so both stay unverified.
    assert [error["page"] for error in result.errors] == [1, 2]
    result = extract_documents(
        scanned_pdf(), "scan.pdf", "scan", {}, {}, ocr_adapter(fake_ocr, policy="allow_partial")
    )
    assert not result.authoritative and len(result) == 1
    assert result[0].metadata["page_extraction"]["words"][0]["text"] == "SCANNED"


def test_host_verification_metadata_and_model_identity_reach_connector(fake_ocr):
    adapter = ocr_adapter(fake_ocr, verifier=AcceptFixture())
    result = extract_documents(scanned_pdf(), "scan.pdf", "scan", {}, {}, adapter)
    assert result.authoritative
    assert result[0].metadata["page_extraction"]["diagnostics"]["verification"]["accepted"] is True
    assert result[0].metadata["extraction"]["options"]["runtime"]["tesseract"]["models_sha256"][
        "eng"
    ]
    assert "not-in-metadata" not in json.dumps(result[0].metadata)


def test_host_policy_can_reject_otherwise_acceptable_ocr(fake_ocr):
    class Reject(AcceptFixture):
        def verify(self, page):
            return False

    page = ocr_adapter(fake_ocr, verifier=Reject()).extract_outcomes(scanned_pdf())[0]
    assert not page.complete and page.reason == "host_policy_rejected"


@pytest.mark.parametrize("result", ["yes", 1, None])
def test_nonboolean_host_verification_fails_closed(fake_ocr, result):
    class Invalid(AcceptFixture):
        def verify(self, page):
            return result

    with pytest.raises(PDFExtractionError, match="verification_failed"):
        ocr_adapter(fake_ocr, verifier=Invalid()).extract_outcomes(scanned_pdf())


def test_model_change_changes_snapshot_configuration(fake_ocr):
    adapter = ocr_adapter(fake_ocr, verifier=AcceptFixture())
    first = extract_documents(scanned_pdf(), "scan.pdf", "scan", {}, {}, adapter)
    (fake_ocr[1] / "eng.traineddata").write_bytes(b"different model")
    second = extract_documents(scanned_pdf(), "scan.pdf", "scan", {}, {}, adapter)
    assert first[0].content == second[0].content
    assert first[0].metadata["extraction"] != second[0].metadata["extraction"]


@pytest.mark.asyncio
async def test_directory_and_pipeline_preserve_active_revision_on_unverified_ocr(
    fake_ocr, tmp_path
):
    from ragnor.ingestion import DirectoryConnector
    from tests.test_ingestion_pipeline import (
        _FakeSessionFactory,
        _HostVersionStore,
        _MutableStore,
        _pipeline,
        _source,
    )

    (tmp_path / "scan.pdf").write_bytes(scanned_pdf())
    connector = DirectoryConnector(tmp_path, options={"extractor": ocr_adapter(fake_ocr)})
    incomplete = await connector.load()
    assert not incomplete.authoritative
    store = _MutableStore()
    versions = _HostVersionStore(store)
    pipeline = _pipeline(store, versions, _FakeSessionFactory())
    await pipeline.run(_source(), versions)
    assert await pipeline.run(incomplete, versions, purge_on_empty=True) == 0
    assert await store.count() == 1 and len(versions.created) == 1
    assert not versions.deleted and not versions.closed


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Install host Tesseract for real OCR")
def test_real_tesseract_extracts_legible_scan_and_reports_confidence():
    page = TesseractPDFExtractor().extract_outcomes(scanned_pdf())[0]
    assert page.status == PageExtractionStatus.PARTIAL
    assert "4200" in page.text and page.words
    assert page.diagnostics["mean_confidence"] > 0
    blank = TesseractPDFExtractor().extract_outcomes(scanned_pdf(white=True))[0]
    assert blank.status == PageExtractionStatus.VERIFIED_BLANK


def test_invalid_configuration_and_quota_weakening_are_rejected(fake_ocr):
    for values in [{"language": "../eng"}, {"dpi": 0}, {"psm": 1}, {"page_timeout": float("inf")}]:
        with pytest.raises(ValueError):
            TesseractPDFExtractor(**values)
    from ragnor.ingestion.ocr import shared_limits

    combined = shared_limits(
        PDFExtractionLimits(max_memory_bytes=900_000_000),
        PDFExtractionLimits(max_memory_bytes=500_000_000),
    )
    assert combined.max_memory_bytes == 500_000_000


def test_parallel_extractions_keep_per_document_runtime_identity(fake_ocr):
    from concurrent.futures import ThreadPoolExecutor

    adapter = ocr_adapter(fake_ocr, verifier=AcceptFixture())

    def load(raw):
        return extract_documents(raw, "input.pdf", "input", {}, {}, adapter)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(load, scanned_pdf())
        second = executor.submit(load, scanned_pdf())
        one, two = first.result(), second.result()
    assert one.authoritative and two.authoritative
    for result in (one, two):
        runtime = result[0].metadata["extraction"]["options"]["runtime"]
        assert runtime["tesseract"]["models_sha256"]["eng"]


@pytest.mark.asyncio
async def test_approved_exact_evidence_ingests_and_model_change_requires_new_review(fake_ocr):
    from ragnor.ingestion import EvidenceAllowlistVerifier
    from tests.test_ingestion_pipeline import (
        _FakeSessionFactory,
        _HostVersionStore,
        _MutableStore,
        _pipeline,
    )

    data = scanned_pdf()
    original = ocr_adapter(fake_ocr).extract_outcomes(data)[0]
    verifier = EvidenceAllowlistVerifier([original.evidence_sha256])
    adapter = ocr_adapter(fake_ocr, verifier=verifier)
    approved = extract_documents(data, "scan.pdf", "scan", {}, {}, adapter)
    assert approved.authoritative
    page = adapter.extract_outcomes(data)[0]
    assert page.complete and page.evidence_sha256 == original.evidence_sha256
    store = _MutableStore()
    versions = _HostVersionStore(store)
    pipeline = _pipeline(store, versions, _FakeSessionFactory())
    assert await pipeline.run(approved, versions) == 1
    assert await pipeline.run(approved, versions) == 1
    assert len(versions.created) == 1
    (fake_ocr[1] / "eng.traineddata").write_bytes(b"different model")
    changed = extract_documents(data, "scan.pdf", "scan", {}, {}, adapter)
    assert not changed.authoritative
    assert await pipeline.run(changed, versions, purge_on_empty=True) == 0
    assert await store.count() == 1 and len(versions.created) == 1


def test_connector_keeps_fixed_dependency_failure_code_without_provider_text():
    result = extract_documents(
        scanned_pdf(),
        "scan.pdf",
        "scan",
        {},
        {},
        TesseractPDFExtractor(executable="missing-fixture-executable"),
    )
    assert not result.authoritative
    assert result.errors[0]["code"] == "dependency_missing"
    assert result.errors[0]["reason"] == "pdf_extraction_failed"


def test_adapter_imports_and_configures_without_optional_dependencies():
    script = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'pypdf', 'pypdfium2', 'PIL', 'litellm', 'docling', 'docling_core'}:
        raise ImportError('optional package blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from ragnor.ingestion import DoclingPDFExtractor, LiteLLMOCRProvider, TesseractPDFExtractor
adapters = [
    DoclingPDFExtractor(),
    LiteLLMOCRProvider(model='openai/gpt-5'),
    TesseractPDFExtractor(),
]
for adapter in adapters:
    assert adapter.options['policy'] == 'strict'
assert not any(
    name in sys.modules for name in ('pypdf', 'pypdfium2', 'PIL', 'litellm', 'docling')
)
"""
    import subprocess

    result = subprocess.run(  # noqa: S603 -- fixed interpreter and literal test script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_page_selection_and_posix_requirement_are_enforced(fake_ocr):
    adapter = ocr_adapter(fake_ocr)
    with pytest.raises(ValueError, match="positive integers"):
        adapter.extract_outcomes(scanned_pdf(), page_numbers=frozenset({0}))
    pages = adapter.extract_outcomes(mixed_pdf(), page_numbers=frozenset({2}))
    assert pages[0].provider == "pdf" and pages[0].complete
    assert pages[1].provider == "tesseract" and not pages[1].complete
