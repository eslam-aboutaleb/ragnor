"""SDK-only extraction, process isolation and connector/publication contracts."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import sys

import pytest

from ragnor.ingestion import (
    AutoPDFExtractor,
    PDFExtractionError,
    PDFExtractionLimits,
    PdfPlumberExtractor,
    TesseractPDFExtractor,
)
from ragnor.ingestion.connector_core import extract_documents
from ragnor.ingestion.extractors import PageExtractionStatus

pytest.importorskip("pypdf")
pytest.importorskip("pdfplumber")
pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")


def digital_pdf(*, table=False):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=300)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    if table:
        operations = b"20 100 m 260 100 l S 20 140 m 260 140 l S 20 180 m 260 180 l S 20 220 m 260 220 l S 20 100 m 20 220 l S 260 100 m 260 220 l S 140 100 m 140 180 l S\n"
        for text, x, y in [
            ("Summary", 40, 195),
            ("A", 40, 155),
            ("10", 160, 155),
            ("B", 40, 115),
            ("20", 160, 115),
        ]:
            operations += f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET\n".encode()
    else:
        operations = b"BT /F1 16 Tf 20 200 Td (Native SDK text) Tj ET"
    stream = DecodedStreamObject()
    stream.set_data(operations)
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def scanned_pdf(*, white=False):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (800, 200), "white")
    if not white:
        ImageDraw.Draw(image).text(
            (30, 50), "INVOICE TOTAL 4200", fill="black", font=ImageFont.load_default(size=48)
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=100)
    return buffer.getvalue()


def mixed_pdf():
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for raw in (digital_pdf(), scanned_pdf()):
        writer.add_page(PdfReader(io.BytesIO(raw)).pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class AcceptFixture:
    """Explicit fixture oracle, never a recommended production confidence bar."""

    id = "fixture-reviewed"
    version = "1"
    options = {"fixture": "synthetic", "password": "not-in-metadata"}

    def verify(self, page):
        return "4200" in page.text or bool(page.tables) or "Native SDK" in page.text


@pytest.fixture
def fake_ocr(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "eng.traineddata").write_bytes(b"fixture-model")
    command = tmp_path / "ocr"
    command.write_text(f'''#!{sys.executable}
import sys
if '--version' in sys.argv:
    print('fixture-tesseract 1.0')
elif '--list-langs' in sys.argv:
    print('List of available languages in "{models}" (1):\\neng')
else:
    sys.stdin.buffer.read()
    assert 'tessedit_create_tsv=1' in sys.argv
    assert 'tessedit_create_txt=0' in sys.argv
    print('level\\tblock_num\\tpar_num\\tline_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext')
    print('5\\t1\\t1\\t1\\t10\\t10\\t100\\t30\\t96\\tSCANNED')
    print('5\\t1\\t1\\t1\\t120\\t10\\t80\\t30\\t94\\t4200')
''')
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    return command, models


def ocr_adapter(fake_ocr, **kwargs):
    command, models = fake_ocr
    return TesseractPDFExtractor(executable=str(command), tessdata_dir=models, **kwargs)


def test_native_words_and_merged_cells_are_real_sdk_evidence():
    page = PdfPlumberExtractor().extract_outcomes(digital_pdf(table=True))[0]
    assert page.status == PageExtractionStatus.PARTIAL
    assert len(page.tables) == 1
    table = page.tables[0]
    assert (table.rows, table.cols) == (3, 2)
    assert {(c.row, c.col, c.text, c.rowspan, c.colspan) for c in table.cells} == {
        (0, 0, "Summary", 1, 2),
        (1, 0, "A", 1, 1),
        (1, 1, "10", 1, 1),
        (2, 0, "B", 1, 1),
        (2, 1, "20", 1, 1),
    }
    assert 'colspan="2"' in table.to_html()
    assert page.words and page.regions
    assert not page.complete


def test_native_text_is_complete_without_ocr_dependencies():
    adapter = AutoPDFExtractor(ocr=TesseractPDFExtractor(executable="missing-fixture-executable"))
    page = adapter.extract_outcomes(digital_pdf())[0]
    assert page.complete and page.text == "Native SDK text"
    assert page.provider == "pdfplumber"
    assert "tesseract" not in page.diagnostics["runtime"]


def test_ocr_runs_explicit_tsv_with_model_only_directory(fake_ocr):
    adapter = ocr_adapter(fake_ocr)
    page = adapter.extract_outcomes(scanned_pdf())[0]
    assert page.text == "SCANNED 4200"
    assert page.status == PageExtractionStatus.PARTIAL
    assert page.diagnostics["mean_confidence"] == 95
    assert page.words[0].confidence == 96
    assert page.diagnostics["runtime"]["tesseract"]["models_sha256"]["eng"]
    assert page.tables == ()


def test_router_keeps_native_pages_and_ocrs_only_uncertain_pages(fake_ocr):
    pages = AutoPDFExtractor(ocr=ocr_adapter(fake_ocr)).extract_outcomes(mixed_pdf())
    assert [p.page for p in pages] == [1, 2]
    assert [p.provider for p in pages] == ["pdfplumber", "tesseract"]
    assert pages[0].complete and not pages[1].complete
    assert "Native SDK" in pages[0].text and "4200" in pages[1].text


def test_router_keeps_native_tables_without_unnecessary_ocr(fake_ocr):
    page = AutoPDFExtractor(ocr=ocr_adapter(fake_ocr)).extract_outcomes(digital_pdf(table=True))[0]
    assert page.provider == "pdfplumber"
    assert page.tables[0].cells[0].colspan == 2
    assert page.text.count("<table>") == 1
    assert "4200" not in page.text
    assert "tesseract" not in page.diagnostics["runtime"]
    assert not page.complete


def test_white_render_can_be_blank_but_no_text_is_not_automatically_blank(fake_ocr):
    # The stub hallucinates words even on white input: those words prevent a blank claim.
    page = ocr_adapter(fake_ocr).extract_outcomes(scanned_pdf(white=True))[0]
    assert page.status == PageExtractionStatus.PARTIAL


def test_strict_connector_rejects_unverified_ocr_but_keeps_review_text(fake_ocr):
    result = extract_documents(
        mixed_pdf(), "book.pdf", "book", {}, {}, AutoPDFExtractor(ocr=ocr_adapter(fake_ocr))
    )
    assert not result.authoritative and not result
    assert result.errors[0]["page"] == 2
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


def test_host_policy_can_reject_otherwise_complete_native_text():
    class Reject(AcceptFixture):
        def verify(self, page):
            return False

    page = PdfPlumberExtractor(verifier=Reject()).extract_outcomes(digital_pdf())[0]
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


@pytest.mark.parametrize(
    "limits,data",
    [
        (PDFExtractionLimits(max_bytes=1), b"pdf"),
        (PDFExtractionLimits(max_cells=2), None),
        (PDFExtractionLimits(max_words=1), False),
    ],
)
def test_quotas_reject_instead_of_silently_truncating(limits, data):
    data = digital_pdf(table=True) if data is None else digital_pdf() if data is False else data
    with pytest.raises(PDFExtractionError, match="input_limit|resource_limit"):
        PdfPlumberExtractor(limits=limits).extract_outcomes(data)


def test_pixel_limit_is_checked_before_render(fake_ocr):
    adapter = ocr_adapter(fake_ocr, limits=PDFExtractionLimits(max_pixels=10))
    with pytest.raises(PDFExtractionError, match="resource_limit"):
        adapter.extract_outcomes(scanned_pdf())


def test_missing_dependencies_and_malformed_pdf_are_fixed_errors():
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        PdfPlumberExtractor().extract_outcomes(b"not a pdf")
    with pytest.raises(PDFExtractionError, match="dependency_missing"):
        TesseractPDFExtractor(executable="missing-fixture-executable").extract_outcomes(
            scanned_pdf()
        )


def test_provider_output_is_private_and_malformed_tsv_stays_failed(fake_ocr, capfd):
    command, _ = fake_ocr
    command.write_text(
        f"#!{sys.executable}\nimport sys\n"
        'if "--version" in sys.argv: print("tesseract 1.0")\n'
        'else: print("PRIVATE_SOURCE_BODY"); print("PRIVATE_PROVIDER_ERROR", file=sys.stderr)\n'
    )
    page = ocr_adapter(fake_ocr).extract_outcomes(scanned_pdf())[0]
    assert page.status == PageExtractionStatus.FAILED
    assert page.reason == "ocr_page_failed"
    captured = capfd.readouterr()
    assert "PRIVATE" not in captured.out + captured.err + repr(page)


@pytest.mark.skipif(os.name != "posix", reason="OCR process cleanup requires POSIX")
def test_document_deadline_kills_ocr_descendant(fake_ocr, tmp_path):
    import time
    import subprocess

    command, _ = fake_ocr
    pid_file = tmp_path / "pid"
    command.write_text(f"""#!{sys.executable}
import os,sys,time
if '--version' in sys.argv:
 print('fixture-tesseract 1.0')
else:
 sys.stdin.buffer.read()
 open({str(pid_file)!r},'w').write(str(os.getpid()))
 time.sleep(120)
""")
    adapter = ocr_adapter(
        fake_ocr,
        limits=PDFExtractionLimits(timeout_seconds=60 if sys.gettrace() else 5),
        page_timeout=120,
    )
    with pytest.raises(PDFExtractionError, match="timeout"):
        adapter.extract_outcomes(scanned_pdf())
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    for _ in range(30):
        result = subprocess.run(  # noqa: S603 -- fixed OS process query, numeric fixture PID
            [shutil.which("ps"), "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            break
        time.sleep(0.02)
    else:
        pytest.fail("OCR descendant remained running after document timeout")


@pytest.mark.asyncio
async def test_directory_and_pipeline_preserve_active_revision_on_unverified_ocr(
    fake_ocr, tmp_path
):
    from ragnor.ingestion import DirectoryConnector
    from tests.test_ingestion_pipeline import (
        _MutableStore,
        _HostVersionStore,
        _FakeSessionFactory,
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
    native = PdfPlumberExtractor(limits=PDFExtractionLimits(max_pages=2))
    assert AutoPDFExtractor(native=native).limits.max_pages == 2
    with pytest.raises(ValueError, match="verification on Auto"):
        AutoPDFExtractor(ocr=ocr_adapter(fake_ocr, verifier=AcceptFixture()))


def test_parallel_extractions_keep_per_document_runtime_identity(fake_ocr):
    from concurrent.futures import ThreadPoolExecutor

    adapter = AutoPDFExtractor(ocr=ocr_adapter(fake_ocr), verifier=AcceptFixture())

    def load(raw):
        return extract_documents(raw, "input.pdf", "input", {}, {}, adapter)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(load, digital_pdf())
        second = executor.submit(load, scanned_pdf())
        native, scanned = first.result(), second.result()
    assert native.authoritative and scanned.authoritative
    assert "tesseract" not in native[0].metadata["extraction"]["options"]["runtime"]
    assert "tesseract" in scanned[0].metadata["extraction"]["options"]["runtime"]


@pytest.mark.parametrize("rotation,expected", [(0, (1600, 400)), (90, (400, 1600))])
def test_ocr_boxes_use_rotated_full_mediabox_not_cropbox(fake_ocr, rotation, expected):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    writer = PdfWriter()
    page = writer.add_page(PdfReader(io.BytesIO(scanned_pdf())).pages[0])
    page.cropbox = RectangleObject([100, 0, 300, 144])
    if rotation:
        page.rotate(rotation)
    buffer = io.BytesIO()
    writer.write(buffer)
    result = ocr_adapter(fake_ocr).extract_outcomes(buffer.getvalue())[0]
    assert (result.diagnostics["pixel_width"], result.diagnostics["pixel_height"]) == expected
    assert all(0 <= value <= 1 for word in result.words for value in word.bbox)


@pytest.mark.asyncio
async def test_approved_exact_evidence_ingests_and_model_change_requires_new_review(fake_ocr):
    from ragnor.ingestion import EvidenceAllowlistVerifier
    from tests.test_ingestion_pipeline import (
        _MutableStore,
        _HostVersionStore,
        _FakeSessionFactory,
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


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Install host Tesseract for real OCR")
def test_real_bilingual_models_route_mixed_pdf():
    import subprocess
    import unicodedata

    model_dir = os.environ.get("RAGNOR_TEST_TESSDATA_DIR")
    args = [shutil.which("tesseract"), "--list-langs"]
    if model_dir:
        args.extend(["--tessdata-dir", model_dir])
    languages = subprocess.run(  # noqa: S603 -- fixed host OCR probe
        args, capture_output=True, text=True, check=True, timeout=5
    ).stdout.splitlines()
    if not {"ara", "eng"} <= set(languages):
        pytest.skip("Install Arabic and English models for real bilingual OCR")
    adapter = AutoPDFExtractor(
        ocr=TesseractPDFExtractor(language="ara+eng", tessdata_dir=model_dir)
    )
    pages = adapter.extract_outcomes(mixed_pdf())
    assert pages[0].provider == "pdfplumber" and pages[0].complete
    assert pages[1].provider == "tesseract" and pages[1].status == PageExtractionStatus.PARTIAL
    digits = "".join(str(unicodedata.decimal(c)) for c in pages[1].text if c.isdecimal())
    assert "4200" in digits
    assert set(pages[1].diagnostics["runtime"]["tesseract"]["models_sha256"]) == {"ara", "eng"}


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


def test_native_table_page_does_not_require_an_ocr_installation():
    adapter = AutoPDFExtractor(ocr=TesseractPDFExtractor(executable="missing-ocr"))
    page = adapter.extract_outcomes(digital_pdf(table=True))[0]
    assert page.provider == "pdfplumber" and page.tables
    assert not page.complete


def test_router_cannot_weaken_optional_memory_limit():
    adapter = AutoPDFExtractor(
        native=PdfPlumberExtractor(limits=PDFExtractionLimits(max_memory_bytes=500_000_000)),
        limits=PDFExtractionLimits(max_memory_bytes=900_000_000),
    )
    assert adapter.limits.max_memory_bytes == 500_000_000


def test_sdk_benchmark_uses_worker_deadline_and_real_tables():
    from benchmarks.quality_providers import SDKNativeProvider

    provider = SDKNativeProvider()
    page = provider.extract_bounded(digital_pdf(table=True), 10)[0]
    assert provider.adapter.limits.timeout_seconds == 10
    assert page.tables and 'colspan="2"' in page.tables[0]["html"]
    assert page.words and not page.verified_complete
    assert page.cpu_seconds is not None


def test_invalid_table_grid_preserves_text_but_cannot_claim_completeness(monkeypatch):
    from ragnor.ingestion import _pdf_native
    from ragnor.ingestion._pdf_worker import Budget
    from ragnor.ingestion.extractors import PDFExtractor

    def invalid_grid(*args):
        raise ValueError("Overlapping table cells")

    monkeypatch.setattr(_pdf_native, "_table", invalid_grid)
    data = digital_pdf(table=True)
    page = _pdf_native.extract_native(
        data,
        PDFExtractor().extract_outcomes(data),
        PdfPlumberExtractor()._config,
        Budget(PDFExtractionLimits()),
    )[0]
    assert "Summary" in page.text and "20" in page.text
    assert page.words and not page.tables
    assert page.status == PageExtractionStatus.PARTIAL
    assert page.diagnostics["rejected_tables"] == 1


def test_router_deduplicates_ocr_table_text_and_keeps_native_text_on_ocr_failure(monkeypatch):
    from dataclasses import replace
    from ragnor.ingestion import _pdf_native, _pdf_ocr
    from ragnor.ingestion._pdf_worker import process
    from dataclasses import asdict
    from ragnor.ingestion.extraction_types import ExtractionWord

    data = digital_pdf(table=True)
    native = PdfPlumberExtractor().extract_outcomes(data)[0]
    native = replace(native, has_images=True)
    monkeypatch.setattr(_pdf_native, "extract_native", lambda *args: [native])
    ocr = replace(
        native,
        provider="tesseract",
        text="OUTSIDE Summary",
        words=(
            ExtractionWord("OUTSIDE", (0.8, 0.8, 0.9, 0.9)),
            ExtractionWord("Summary", (0.1, 0.3, 0.2, 0.35)),
        ),
    )
    monkeypatch.setattr(_pdf_ocr, "extract_ocr", lambda *args: ({1: ocr}, {}))
    adapter = AutoPDFExtractor()
    request = {"engine": "auto", "config": adapter._config, "limits": asdict(adapter.limits)}
    page = process(data, request)["pages"][0]
    assert page["text"].count("Summary") == 1 and "OUTSIDE" in page["text"]
    assert page["text"].count("<table>") == 1
    failed = replace(ocr, text="", status=PageExtractionStatus.FAILED, words=())
    monkeypatch.setattr(_pdf_ocr, "extract_ocr", lambda *args: ({1: failed}, {}))
    page = process(data, request)["pages"][0]
    assert page["text"] == native.text
    assert page["status"] == PageExtractionStatus.PARTIAL
    assert page["diagnostics"]["ocr_fallback_status"] == "failed"


@pytest.mark.skipif(os.name != "posix", reason="Worker supervision requires POSIX")
def test_worker_reaps_ocr_when_host_process_disappears(fake_ocr, tmp_path):
    import subprocess
    import time
    import signal

    command, models = fake_ocr
    pid_file = tmp_path / "orphan-pid"
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(scanned_pdf())
    command.write_text(f"""#!{sys.executable}
import os,sys,time
if '--version' in sys.argv:
 print('fixture-tesseract 1.0')
else:
 sys.stdin.buffer.read()
 open({str(pid_file)!r},'w').write(str(os.getpid()))
 time.sleep(120)
""")
    script = """
import sys
from pathlib import Path
from ragnor.ingestion import TesseractPDFExtractor, PDFExtractionLimits
TesseractPDFExtractor(executable=sys.argv[2], tessdata_dir=sys.argv[3],
    page_timeout=120, limits=PDFExtractionLimits(timeout_seconds=120)
).extract_outcomes(Path(sys.argv[1]).read_bytes())
"""
    host = subprocess.Popen(  # noqa: S603 -- fixed fixture script, local argv
        [sys.executable, "-c", script, str(pdf), str(command), str(models)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    group = None
    try:
        deadline = time.monotonic() + 15
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "OCR fixture did not start"
        pid = int(pid_file.read_text())
        group = os.getpgid(pid)
        host.kill()
        host.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = subprocess.run(  # noqa: S603 -- fixed OS process query
                [shutil.which("ps"), "-p", str(pid), "-o", "stat="],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode or result.stdout.strip().startswith("Z"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("OCR child survived loss of its SDK host")
    finally:
        if host.poll() is None:
            host.kill()
            host.wait(timeout=5)
        if group is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                # Normal cleanup already removed the worker group.
                pass


@pytest.mark.skipif(os.name != "posix", reason="File-size limits require POSIX")
def test_ocr_output_flood_is_bounded_and_never_published(fake_ocr):
    command, _ = fake_ocr
    command.write_text(f"""#!{sys.executable}
import sys
if '--version' in sys.argv:
 print('fixture-tesseract 1.0')
else:
 sys.stdin.buffer.read()
 sys.stdout.write('X' * 2_000_000)
""")
    page = ocr_adapter(
        fake_ocr, limits=PDFExtractionLimits(max_output_bytes=256_000)
    ).extract_outcomes(scanned_pdf())[0]
    assert page.status == PageExtractionStatus.FAILED
    assert not page.text and not page.complete


def test_worker_queue_respects_deadline_without_starting_a_process(monkeypatch):
    import threading
    from ragnor.ingestion import _pdf_process

    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_pdf_process, "_WORKER_SLOTS", slots)
    with pytest.raises(PDFExtractionError, match="timeout"):
        PdfPlumberExtractor(limits=PDFExtractionLimits(timeout_seconds=0.02)).extract_outcomes(
            digital_pdf()
        )
    slots.release()
    assert PdfPlumberExtractor().extract_outcomes(digital_pdf())[0].complete


def test_degenerate_word_geometry_preserves_plain_text_and_stays_partial(monkeypatch):
    from ragnor.ingestion import _pdf_native
    from ragnor.ingestion._pdf_worker import Budget
    from ragnor.ingestion.extractors import PDFExtractor

    original = _pdf_native.ExtractionWord

    def zero_width(text, box):
        return original(text, (box[0], box[1], box[0], box[3]))

    monkeypatch.setattr(_pdf_native, "ExtractionWord", zero_width)
    data = digital_pdf()
    page = _pdf_native.extract_native(
        data,
        PDFExtractor().extract_outcomes(data),
        PdfPlumberExtractor()._config,
        Budget(PDFExtractionLimits()),
    )[0]
    assert page.text == "Native SDK text" and not page.words
    assert not page.complete and page.diagnostics["unlocated_words"] == 3


@pytest.mark.skipif(os.name != "posix", reason="POSIX resource limits")
def test_worker_resource_limits_preserve_tighter_host_limits(monkeypatch):
    import resource
    from ragnor.ingestion import _pdf_worker

    applied = []
    monkeypatch.setattr(_pdf_worker.sys, "platform", "linux")
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (1024, 2048))
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: applied.append((kind, limits)))
    _pdf_worker._set_resource_limits(
        PDFExtractionLimits(max_output_bytes=8192, max_memory_bytes=4096)
    )
    assert applied == [(resource.RLIMIT_FSIZE, (1024, 1024)), (resource.RLIMIT_AS, (1024, 1024))]


def test_explicit_memory_cap_fails_closed_on_unsupported_platform(monkeypatch):
    from ragnor.ingestion import _pdf_worker

    monkeypatch.setattr(_pdf_worker.sys, "platform", "win32")
    with pytest.raises(PDFExtractionError, match="resource_limit"):
        _pdf_worker._set_resource_limits(PDFExtractionLimits(max_memory_bytes=1_000_000))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux address-space ceiling")
def test_linux_memory_ceiling_is_applied_only_to_the_worker():
    with pytest.raises(PDFExtractionError):
        PdfPlumberExtractor(
            limits=PDFExtractionLimits(max_memory_bytes=10_000_000)
        ).extract_outcomes(digital_pdf())
    assert PdfPlumberExtractor().extract_outcomes(digital_pdf())[0].complete
