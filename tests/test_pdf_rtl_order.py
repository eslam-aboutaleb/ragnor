"""Right-to-left lines come back from the native engine in logical order."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("pypdf")
from pypdf import PdfReader  # noqa: E402

from ragnor.ingestion import PdfPlumberExtractor  # noqa: E402
from ragnor.ingestion._pdf_native import to_logical_order  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "benchmarks/pdf_extraction/fixtures/documents"


def _w(text, top=10.0):
    return {"text": text, "top": top}


def test_arabic_line_is_reversed_to_logical_order():
    # Visual order as pdfminer returns it: last word first, glyphs reversed.
    visual = [_w("ﻢﻬﻣ"), _w("ﺭﺎﻌﺷﺇ")]
    assert [w["text"] for w in to_logical_order(visual)] == ["ﺇﺷﻌﺎﺭ", "ﻣﻬﻢ"]


def test_latin_lines_are_untouched():
    words = [_w("Total"), _w("due:"), _w("1,250.00")]
    assert to_logical_order(words) == words


def test_ltr_runs_inside_an_rtl_line_keep_their_order():
    visual = [_w("ﻢﻬﻣ"), _w("INV-1"), _w("2026"), _w("ﺭﺎﻌﺷﺇ")]
    texts = [w["text"] for w in to_logical_order(visual)]
    assert texts == ["ﺇﺷﻌﺎﺭ", "INV-1", "2026", "ﻣﻬﻢ"]


def test_lines_are_handled_independently():
    visual = [_w("ﻢﻬﻣ", 10), _w("ﺭﺎﻌﺷﺇ", 10), _w("Plain", 30), _w("text", 30)]
    assert [w["text"] for w in to_logical_order(visual)] == ["ﺇﺷﻌﺎﺭ", "ﻣﻬﻢ", "Plain", "text"]


def test_empty_input():
    assert to_logical_order([]) == []


def test_real_arabic_pdf_matches_the_logical_text_pypdf_extracts():
    data = (DOCS / "arabic-rtl-notice.pdf").read_bytes()
    native = PdfPlumberExtractor().extract_outcomes(data)
    reference = PdfReader(io.BytesIO(data))
    for outcome, page in zip(native, reference.pages, strict=True):
        assert outcome.text.split() == page.extract_text().split()
