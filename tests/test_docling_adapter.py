"""Docling adapter: thin configuration, evidence mapping and the verifier gate."""

from __future__ import annotations

import pytest

from ragnor.ingestion import DoclingPDFExtractor, PDFExtractionError, PageExtractionStatus
from ragnor.ingestion._pdf_docling import outcomes_from_conversion
from ragnor.ingestion._pdf_worker import Budget
from ragnor.ingestion.connector_core import extract_documents
from ragnor.ingestion.extraction_types import PDFExtractionLimits
from ragnor.ingestion.extractors import PDFExtractor
from tests.pdf_fixtures import AcceptFixture, mixed_pdf, scanned_pdf


def test_configuration_is_validated():
    for values in [
        {"artifacts_path": 3},
        {"do_ocr": "yes"},
        {"do_table_structure": 1},
        {"policy": "loose"},
    ]:
        with pytest.raises(ValueError):
            DoclingPDFExtractor(**values)
    adapter = DoclingPDFExtractor(artifacts_path="models", do_ocr=True)
    assert adapter.options["do_ocr"] is True
    assert adapter.options["policy"] == "strict"


def test_missing_docling_is_a_fixed_dependency_error():
    import importlib.util

    if importlib.util.find_spec("docling") is not None:
        pytest.skip("Docling is installed in this environment")
    pytest.importorskip("pypdfium2")
    with pytest.raises(PDFExtractionError, match="dependency_missing"):
        DoclingPDFExtractor(limits=PDFExtractionLimits(timeout_seconds=30)).extract_outcomes(
            scanned_pdf()
        )


def test_conversion_mapping_keeps_unknown_pages_unverified():
    pytest.importorskip("pypdf")
    baseline = PDFExtractor().extract_outcomes(mixed_pdf())
    converted = {
        2: {
            "text": ["INVOICE TOTAL 4200"],
            "tables": [
                {
                    "rows": 2,
                    "cols": 2,
                    "bbox": (0.1, 0.1, 0.9, 0.9),
                    "cells": [
                        {"row": 0, "col": 0, "text": "Total", "rowspan": 1, "colspan": 2},
                        {"row": 1, "col": 0, "text": "4200", "rowspan": 1, "colspan": 1},
                    ],
                }
            ],
        }
    }
    pages = outcomes_from_conversion(
        converted, baseline, Budget(PDFExtractionLimits()), {"packages": {"docling": "2.0.0"}}
    )
    # Page one keeps its complete pypdf baseline; page two becomes partial evidence.
    assert pages[0].complete and pages[0].provider == "pdf"
    page = pages[1]
    assert page.status == PageExtractionStatus.PARTIAL
    assert page.provider == "docling" and page.version == "2.0.0"
    assert "4200" in page.text and 'colspan="2"' in page.text
    assert page.tables[0].cells[0].colspan == 2
    assert page.diagnostics["verification"] == "pending"
    assert not page.complete


def test_conversion_without_content_preserves_baseline_status():
    pytest.importorskip("pypdf")
    baseline = PDFExtractor().extract_outcomes(mixed_pdf())
    pages = outcomes_from_conversion(
        {2: {"text": [], "tables": []}},
        baseline,
        Budget(PDFExtractionLimits()),
        {"packages": {}},
    )
    assert pages[1].status == PageExtractionStatus.NEEDS_OCR
    assert pages[1].provider == "pdf"


def test_docling_output_stays_partial_until_host_verification(monkeypatch):
    pytest.importorskip("pypdf")
    from ragnor.ingestion import pdf_extractors

    data = scanned_pdf()
    baseline = PDFExtractor().extract_outcomes(data)
    converted_page = outcomes_from_conversion(
        {1: {"text": ["INVOICE TOTAL 4200"], "tables": []}},
        baseline,
        Budget(PDFExtractionLimits()),
        {"packages": {"docling": "2.0.0"}},
    )
    monkeypatch.setattr(
        pdf_extractors,
        "execute",
        lambda payload, engine, config, limits: (converted_page, {"packages": {}}),
    )
    adapter = DoclingPDFExtractor()
    page = adapter.extract_outcomes(data)[0]
    assert page.status == PageExtractionStatus.PARTIAL
    assert page.provider == "docling"
    result = extract_documents(data, "scan.pdf", "scan", {}, {}, adapter)
    assert not result.authoritative and not result
    verified = DoclingPDFExtractor(verifier=AcceptFixture())
    result = extract_documents(data, "scan.pdf", "scan", {}, {}, verified)
    assert result.authoritative and "4200" in result[0].content


def test_real_docling_conversion_if_installed():
    pytest.importorskip("docling")
    page = DoclingPDFExtractor(limits=PDFExtractionLimits(timeout_seconds=300)).extract_outcomes(
        scanned_pdf()
    )[0]
    assert page.provider == "docling"
    assert not page.complete
