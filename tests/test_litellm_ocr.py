"""Hosted OCR contracts: generated PDFs and fake transports, never paid API calls."""

import asyncio
import base64
import json
import sys
from types import SimpleNamespace

import pytest

from ragnor.ingestion import (
    AutoPDFExtractor,
    LiteLLMOCRProvider,
    OCRProvider,
    PDFExtractionError,
    PDFExtractionLimits,
    PageExtractionStatus,
    TesseractOCRProvider,
)
from ragnor.ingestion.connector_core import extract_documents
from tests.test_sdk_pdf_extractors import digital_pdf, scanned_pdf, mixed_pdf
from tests.test_sdk_pdf_extractors import fake_ocr as fixture_fake_ocr

fake_ocr = fixture_fake_ocr


def response(text="INVOICE TOTAL 4200", finish="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, refusal=refusal), finish_reason=finish
            )
        ],
        model="resolved-model-revision",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )


@pytest.mark.parametrize("model", ["openai/gpt-5", "gemini/gemini-2.5-pro"])
def test_multimodal_litellm_payload_and_private_credentials(model, monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return response()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    provider = LiteLLMOCRProvider(model=model, api_key="PRIVATE_KEY")
    assert isinstance(provider, OCRProvider)
    page = provider.extract_outcomes(scanned_pdf())[0]
    assert page.text == "INVOICE TOTAL 4200" and page.status == PageExtractionStatus.PARTIAL
    payload = calls[0]
    assert payload["model"] == model and payload["api_key"] == "PRIVATE_KEY"
    assert payload["max_completion_tokens"] == 4096 and payload["num_retries"] == 0
    assert payload["no-log"] is True
    image = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image.split(",", 1)[1]).startswith(b"\x89PNG")
    assert not page.words and not page.tables
    assert page.diagnostics["usage"]["total_tokens"] == 110
    assert page.diagnostics["response_model"] == "resolved-model-revision"
    assert "PRIVATE_KEY" not in json.dumps(provider.options) + repr(provider) + json.dumps(
        page.diagnostics
    )
    assert "image_base64" not in json.dumps(page.diagnostics)


def test_auto_uses_hosted_provider_only_for_selected_pages_and_strict_ingestion():
    calls = []
    provider = LiteLLMOCRProvider(
        model="openai/gpt-5", completion=lambda **kw: calls.append(kw) or response()
    )
    auto = AutoPDFExtractor(ocr=provider)
    pages = auto.extract_outcomes(mixed_pdf())
    assert pages[0].provider == "pdfplumber" and pages[0].complete
    assert pages[1].provider == "litellm-ocr" and not pages[1].complete
    assert len(calls) == 1
    result = extract_documents(mixed_pdf(), "mixed.pdf", "mixed", {}, {}, auto)
    assert not result.authoritative and not result
    assert result.errors[0]["page"] == 2
    calls.clear()
    assert auto.extract_outcomes(digital_pdf())[0].complete
    assert not calls


@pytest.mark.parametrize(
    "finish,text,refusal",
    [
        ("length", "partial", None),
        ("content_filter", None, None),
        ("stop", None, "refused"),
        ("stop", "", None),
    ],
)
def test_incomplete_refused_or_empty_llm_output_cannot_be_verified(finish, text, refusal):
    class Accept:
        id, version, options = "accept", "1", {}

        def verify(self, page):
            pytest.fail("Incomplete model response reached verification")

    page = LiteLLMOCRProvider(
        model="model", completion=lambda **kw: response(text, finish, refusal), verifier=Accept()
    ).extract_outcomes(scanned_pdf())[0]
    assert page.status == PageExtractionStatus.FAILED and not page.complete


def test_limits_reject_before_cloud_calls_and_do_not_leak_errors():
    calls = []
    provider = LiteLLMOCRProvider(
        model="model", max_ocr_pages=1, completion=lambda **kw: calls.append(kw)
    )
    with pytest.raises(PDFExtractionError, match="resource_limit"):
        provider.extract_outcomes(mixed_pdf())
    assert not calls
    with pytest.raises(PDFExtractionError, match="input_limit"):
        provider.extract_outcomes(b"oversized", limits=PDFExtractionLimits(max_bytes=1))

    def failed(**kwargs):
        raise RuntimeError("PRIVATE_KEY source body")

    provider = LiteLLMOCRProvider(model="model", completion=failed)
    with pytest.raises(PDFExtractionError) as error:
        provider.extract_outcomes(scanned_pdf())
    assert "PRIVATE" not in str(error.value)
    provider = LiteLLMOCRProvider(model="model", completion=lambda **kw: response("x" * 200))
    with pytest.raises(PDFExtractionError, match="output_limit"):
        provider.extract_outcomes(scanned_pdf(), limits=PDFExtractionLimits(max_text_chars=100))


@pytest.mark.asyncio
async def test_async_litellm_transport_and_auto_routing(monkeypatch):
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return response()

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=completion))
    auto = AutoPDFExtractor(ocr=LiteLLMOCRProvider(model="gemini/gemini-2.5-pro"))
    pages = await auto.aextract_outcomes(mixed_pdf())
    assert len(calls) == 1 and pages[1].text == "INVOICE TOTAL 4200"
    assert not pages[1].complete and pages[0].complete


@pytest.mark.asyncio
async def test_async_timeout_cancels_model_request():
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    provider = LiteLLMOCRProvider(model="model", acompletion=slow, request_timeout=0.02)
    with pytest.raises(PDFExtractionError, match="timeout"):
        await provider.aextract_outcomes(scanned_pdf())
    assert cancelled.is_set()


def test_tesseract_remains_an_ocr_provider(fake_ocr):
    command, models = fake_ocr
    provider = TesseractOCRProvider(executable=str(command), tessdata_dir=models)
    assert isinstance(provider, OCRProvider)
    pages = provider.extract_outcomes(mixed_pdf(), page_numbers=frozenset({2}))
    assert pages[0].provider == "pdf" and pages[1].provider == "tesseract"
    assert "4200" in pages[1].text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_ocr_pages": 0},
        {"dpi": True},
        {"request_timeout": float("nan")},
        {"generation_options": {"messages": []}},
        {"api_base": "https://key:secret@example.com"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LiteLLMOCRProvider(model="model", **kwargs)


@pytest.mark.asyncio
async def test_async_concurrency_is_bounded_and_results_stay_in_page_order():
    import io
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for _ in range(4):
        writer.add_page(PdfReader(io.BytesIO(scanned_pdf())).pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)
    active = peak = calls = 0

    async def completion(**kwargs):
        nonlocal active, peak, calls
        calls += 1
        sequence = calls
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.04 if sequence == 1 else 0.01)
            return response(f"result{sequence}")
        finally:
            active -= 1

    pages = await LiteLLMOCRProvider(
        model="model", acompletion=completion, max_concurrency=2
    ).aextract_outcomes(buffer.getvalue())
    assert peak == 2 and active == 0 and calls == 4
    assert [page.page for page in pages] == [1, 2, 3, 4]
    assert [page.text for page in pages] == ["result1", "result2", "result3", "result4"]


def test_model_identity_changes_evidence_and_complete_output_can_be_explicitly_approved():
    from ragnor.ingestion import EvidenceAllowlistVerifier

    data = scanned_pdf()
    first = LiteLLMOCRProvider(model="openai/gpt-5", completion=lambda **kw: response())
    evidence = first.extract_outcomes(data)[0]
    approved = LiteLLMOCRProvider(
        model="openai/gpt-5",
        completion=lambda **kw: response(),
        verifier=EvidenceAllowlistVerifier([evidence.evidence_sha256]),
    )
    assert approved.extract_outcomes(data)[0].complete
    changed = LiteLLMOCRProvider(
        model="gemini/gemini-2.5-pro",
        completion=lambda **kw: response(),
        verifier=EvidenceAllowlistVerifier([evidence.evidence_sha256]),
    )
    assert not changed.extract_outcomes(data)[0].complete


@pytest.mark.asyncio
async def test_explicit_cancellation_propagates_and_reaps_pending_model_tasks():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        entered.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    provider = LiteLLMOCRProvider(model="model", acompletion=slow)
    task = asyncio.create_task(provider.aextract_outcomes(scanned_pdf()))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


def test_empty_or_invalid_page_selection_never_calls_hosted_transport():
    provider = LiteLLMOCRProvider(model="model", completion=lambda **kw: pytest.fail("network"))
    for selection in [frozenset(), frozenset({True}), frozenset({0})]:
        with pytest.raises(ValueError):
            provider.extract_outcomes(scanned_pdf(), page_numbers=selection)
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        provider.extract_outcomes(scanned_pdf(), page_numbers=frozenset({2}))


def test_missing_litellm_is_a_fixed_optional_dependency_error(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_litellm(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("private environment detail")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_litellm)
    with pytest.raises(PDFExtractionError, match="dependency_missing"):
        LiteLLMOCRProvider(model="model").extract_outcomes(scanned_pdf())


def test_truncated_hosted_response_is_withheld_by_connector():
    provider = LiteLLMOCRProvider(
        model="model", completion=lambda **kw: response("partial", "length")
    )
    result = extract_documents(scanned_pdf(), "scan.pdf", "scan", {}, {}, provider)
    assert not result.authoritative and not result
    assert result.errors[0]["code"] == "failed"


def test_auto_cannot_approve_native_fallback_after_truncated_llm_response():
    import io
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    native = writer.add_page(PdfReader(io.BytesIO(digital_pdf())).pages[0])
    native.merge_page(PdfReader(io.BytesIO(scanned_pdf())).pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)

    class Accept:
        id, version, options = "accept", "1", {}

        def verify(self, page):
            pytest.fail("Failed hosted OCR reached verification through native fallback")

    provider = LiteLLMOCRProvider(
        model="model", completion=lambda **kw: response("truncated", "length")
    )
    page = AutoPDFExtractor(ocr=provider, verifier=Accept()).extract_outcomes(buffer.getvalue())[0]
    assert page.status == PageExtractionStatus.FAILED and not page.complete
    assert "Native SDK text" in page.text
    assert page.diagnostics["ocr_diagnostics"]["usage"]["total_tokens"] == 110


@pytest.mark.parametrize("rotation,expected", [(0, (1200, 300)), (90, (300, 1200))])
def test_hosted_images_use_rotated_mediabox_instead_of_cropbox(rotation, expected):
    import io
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    writer = PdfWriter()
    page = writer.add_page(PdfReader(io.BytesIO(scanned_pdf())).pages[0])
    page.cropbox = RectangleObject([100, 0, 300, 144])
    if rotation:
        page.rotate(rotation)
    buffer = io.BytesIO()
    writer.write(buffer)
    observed = []

    def completion(**kwargs):
        encoded = kwargs["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            observed.append(image.size)
        return response()

    result = LiteLLMOCRProvider(model="model", completion=completion).extract_outcomes(
        buffer.getvalue()
    )[0]
    assert observed == [expected]
    assert (result.diagnostics["pixel_width"], result.diagnostics["pixel_height"]) == expected
