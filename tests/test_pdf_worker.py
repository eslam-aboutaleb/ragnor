"""Sandboxed PDF worker: quotas, isolation, supervision and OCR engine contracts."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import sys

import pytest

from ragnor.ingestion._pdf_process import execute
from ragnor.ingestion.extraction_types import PDFExtractionError, PDFExtractionLimits
from ragnor.ingestion.extractors import PageExtractionStatus

pytest.importorskip("pypdf")
pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")


def digital_pdf():
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
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 16 Tf 20 200 Td (Native SDK text) Tj ET")
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


def ocr_config(fake_ocr, **overrides):
    command, models = fake_ocr
    return {
        "language": "eng",
        "tessdata_dir": str(models),
        "executable": str(command),
        "dpi": 200,
        "psm": 3,
        "page_timeout": 30,
        **overrides,
    }


def run_ocr(data, config, limits=None):
    pages, runtime = execute(data, "ocr", config, limits or PDFExtractionLimits())
    assert [page.page for page in pages] == list(range(1, len(pages) + 1))
    return pages, runtime


def test_ocr_runs_explicit_tsv_with_model_only_directory(fake_ocr):
    pages, runtime = run_ocr(scanned_pdf(), ocr_config(fake_ocr))
    page = pages[0]
    assert page.text == "SCANNED 4200"
    assert page.status == PageExtractionStatus.PARTIAL
    assert page.diagnostics["mean_confidence"] == 95
    assert page.words[0].confidence == 96
    assert runtime["tesseract"]["models_sha256"]["eng"]
    assert page.tables == ()


def test_white_render_can_be_blank_but_no_text_is_not_automatically_blank(fake_ocr):
    # The stub hallucinates words even on white input: those words prevent a blank claim.
    pages, _ = run_ocr(scanned_pdf(white=True), ocr_config(fake_ocr))
    assert pages[0].status == PageExtractionStatus.PARTIAL


def test_page_selection_must_reference_real_pages(fake_ocr):
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        run_ocr(scanned_pdf(), ocr_config(fake_ocr, page_numbers=[2]))


def test_render_engine_returns_valid_png_pages():
    pages, runtime = execute(
        digital_pdf(),
        "render",
        {"dpi": 72, "max_ocr_pages": 5, "page_numbers": [1]},
        PDFExtractionLimits(),
    )
    assert [page.page for page in pages] == [1]
    assert len(runtime["images"]) == 1
    image = runtime["images"][0]
    assert image["page"] == 1 and image["width"] > 0 and image["height"] > 0


def test_render_rejects_unknown_engine_and_oversized_page_selection():
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        execute(digital_pdf(), "invented-engine", {}, PDFExtractionLimits())
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        execute(
            digital_pdf(),
            "render",
            {"dpi": 72, "max_ocr_pages": 5, "page_numbers": [9]},
            PDFExtractionLimits(),
        )


@pytest.mark.parametrize(
    "limits,data",
    [
        (PDFExtractionLimits(max_bytes=1), b"pdf"),
        (PDFExtractionLimits(max_words=1), "scanned"),
        (PDFExtractionLimits(max_text_chars=1), "scanned"),
    ])
def test_quotas_reject_instead_of_silently_truncating(fake_ocr, limits, data):
    data = scanned_pdf() if data == "scanned" else data
    with pytest.raises(PDFExtractionError, match="input_limit|resource_limit"):
        run_ocr(data, ocr_config(fake_ocr), limits)


def test_pixel_limit_is_checked_before_render(fake_ocr):
    with pytest.raises(PDFExtractionError, match="resource_limit"):
        run_ocr(scanned_pdf(), ocr_config(fake_ocr), PDFExtractionLimits(max_pixels=10))


def test_missing_dependencies_and_malformed_pdf_are_fixed_errors(fake_ocr):
    with pytest.raises(PDFExtractionError, match="invalid_document"):
        run_ocr(b"not a pdf", ocr_config(fake_ocr))
    with pytest.raises(PDFExtractionError, match="dependency_missing"):
        run_ocr(scanned_pdf(), ocr_config(fake_ocr, executable="missing-fixture-executable"))


def test_provider_output_is_private_and_malformed_tsv_stays_failed(fake_ocr, capfd):
    command, _ = fake_ocr
    command.write_text(
        f"#!{sys.executable}\nimport sys\n"
        'if "--version" in sys.argv: print("tesseract 1.0")\n'
        'else: print("PRIVATE_SOURCE_BODY"); print("PRIVATE_PROVIDER_ERROR", file=sys.stderr)\n'
    )
    pages, _ = run_ocr(scanned_pdf(), ocr_config(fake_ocr))
    assert pages[0].status == PageExtractionStatus.FAILED
    assert pages[0].reason == "ocr_page_failed"
    captured = capfd.readouterr()
    assert "PRIVATE" not in captured.out + captured.err + repr(pages[0])


@pytest.mark.skipif(os.name != "posix", reason="OCR process cleanup requires POSIX")
def test_document_deadline_kills_ocr_descendant(fake_ocr, tmp_path):
    import subprocess
    import time

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
    config = ocr_config(fake_ocr, page_timeout=120)
    limits = PDFExtractionLimits(timeout_seconds=60 if sys.gettrace() else 5)
    with pytest.raises(PDFExtractionError, match="timeout"):
        run_ocr(scanned_pdf(), config, limits)
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


@pytest.mark.skipif(os.name != "posix", reason="Worker supervision requires POSIX")
def test_worker_reaps_ocr_when_host_process_disappears(fake_ocr, tmp_path):
    import signal
    import subprocess
    import time

    command, _ = fake_ocr
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
import json, sys
from pathlib import Path
from ragnor.ingestion._pdf_process import execute
from ragnor.ingestion.extraction_types import PDFExtractionLimits
config = json.loads(sys.argv[2])
execute(Path(sys.argv[1]).read_bytes(), "ocr", config, PDFExtractionLimits(timeout_seconds=120))
"""
    host = subprocess.Popen(  # noqa: S603 -- fixed fixture script, local argv
        [
            sys.executable,
            "-c",
            script,
            str(pdf),
            json.dumps(ocr_config(fake_ocr, page_timeout=120)),
        ],
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
    pages, _ = run_ocr(
        scanned_pdf(), ocr_config(fake_ocr), PDFExtractionLimits(max_output_bytes=256_000)
    )
    assert pages[0].status == PageExtractionStatus.FAILED
    assert not pages[0].text and not pages[0].complete


def test_worker_queue_respects_deadline_without_starting_a_process(fake_ocr, monkeypatch):
    import threading

    from ragnor.ingestion import _pdf_process

    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(_pdf_process, "_WORKER_SLOTS", slots)
    with pytest.raises(PDFExtractionError, match="timeout"):
        run_ocr(scanned_pdf(), ocr_config(fake_ocr), PDFExtractionLimits(timeout_seconds=0.02))
    slots.release()
    pages, _ = run_ocr(scanned_pdf(), ocr_config(fake_ocr))
    assert pages[0].text == "SCANNED 4200"


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
def test_linux_memory_ceiling_is_applied_only_to_the_worker(fake_ocr):
    with pytest.raises(PDFExtractionError):
        run_ocr(
            scanned_pdf(), ocr_config(fake_ocr), PDFExtractionLimits(max_memory_bytes=10_000_000)
        )
    pages, _ = run_ocr(scanned_pdf(), ocr_config(fake_ocr))
    assert pages[0].text == "SCANNED 4200"


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
    pages, _ = run_ocr(buffer.getvalue(), ocr_config(fake_ocr))
    result = pages[0]
    assert (result.diagnostics["pixel_width"], result.diagnostics["pixel_height"]) == expected
    assert all(0 <= value <= 1 for word in result.words for value in word.bbox)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Install host Tesseract for real OCR")
def test_real_tesseract_extracts_legible_scan_and_reports_confidence():
    config = {
        "language": "eng",
        "tessdata_dir": None,
        "executable": "tesseract",
        "dpi": 200,
        "psm": 3,
        "page_timeout": 30,
    }
    pages, _ = run_ocr(scanned_pdf(), config)
    page = pages[0]
    assert page.status == PageExtractionStatus.PARTIAL
    assert "4200" in page.text and page.words
    assert page.diagnostics["mean_confidence"] > 0
    blank, _ = run_ocr(scanned_pdf(white=True), config)
    assert blank[0].status == PageExtractionStatus.VERIFIED_BLANK
