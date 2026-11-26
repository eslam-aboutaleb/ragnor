"""Generated-PDF fixtures and a fake OCR executable shared by provider tests."""

from __future__ import annotations

import io
import stat
import sys

import pytest

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
