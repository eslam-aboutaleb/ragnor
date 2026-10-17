"""Behavioral tests for optional PDF and standard-library HTML extraction."""

from __future__ import annotations

import builtins
import subprocess
import sys
from io import BytesIO

import pytest

from ragit.ingestion.extractors import HTMLExtractor, PDFExtractor, TextExtractor


def make_pdf(*pages: str | None, compressed: bool = False) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream.flate_encode() if compressed else stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("compressed", [False, True])
def test_pdf_uses_real_page_tree_and_keeps_blank_pages(compressed: bool) -> None:
    data = make_pdf("First page", None, "Third page", compressed=compressed)
    extractor = PDFExtractor()
    assert extractor.extract_pages(data) == [(1, "First page"), (2, ""), (3, "Third page")]
    assert extractor.extract(data, name="example.pdf") == "First page\n\nThird page"


def test_pdf_blank_and_empty_documents() -> None:
    extractor = PDFExtractor()
    assert extractor.extract_pages(make_pdf(None, None)) == [(1, ""), (2, "")]
    assert extractor.extract(make_pdf(None)) == ""
    assert extractor.extract_pages(make_pdf()) == []


def test_pdf_page_limit_raises_instead_of_truncating() -> None:
    extractor = PDFExtractor(options={"max_pages": 1})
    assert extractor.extract_pages(make_pdf("One")) == [(1, "One")]
    with pytest.raises(ValueError, match="2 pages.*max_pages=1"):
        extractor.extract_pages(make_pdf("One", "Two"))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_pdf_rejects_invalid_page_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PDFExtractor(options={"max_pages": limit})


def test_pdf_rejects_invalid_data() -> None:
    pytest.importorskip("pypdf")
    from pypdf.errors import PdfReadError

    with pytest.raises(PdfReadError):
        PDFExtractor().extract(b"%PDF-1.4\nendstream\n%%EOF")


def test_pdf_missing_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_pypdf(name: str, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pypdf)
    with pytest.raises(ImportError, match=r"install 'ragit-toolkit\[pdf\]' or 'pypdf'"):
        PDFExtractor().extract_pages(b"not parsed without dependency")
    assert TextExtractor().extract(b"still usable") == "still usable"


def test_importing_extractors_does_not_import_pypdf() -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import ragit.ingestion.extractors; import sys; assert 'pypdf' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_html_preserves_article_header_headings_links_and_entities() -> None:
    data = b"""<article><header><h1>Title &amp; context</h1></header>
        <p>Read <a href="/guide?a=1&amp;b=2"><strong>the guide</strong></a> now.</p>
        <h2>Details</h2><p>One&nbsp; two<br>Three</p></article>"""
    assert HTMLExtractor().extract(data, name="article.html") == (
        "# Title & context\nRead the guide (/guide?a=1&b=2) now.\n## Details\nOne two\nThree"
    )


def test_html_boilerplate_does_not_leak_headings_or_link_urls() -> None:
    data = b"""<head><title>Hidden title</title></head>
        <nav><h1>Menu</h1><a href="/secret-nav">Menu link</a><nav>Nested</nav></nav>
        <header role="banner"><h2>Site banner</h2></header>
        <aside><a href="/sidebar">Sidebar</a></aside>
        <script>bad()</script><style>bad{}</style><noscript>No JS</noscript>
        <template><h2>Template</h2></template><footer><a href="/legal">Legal</a></footer>
        <div hidden><h2>Hidden</h2></div><div aria-hidden="true">Invisible</div>
        <div role="navigation"><a href="/role-nav">Role menu</a></div>
        <article><h1>Article</h1><p><a href="/body">Body</a></p></article>"""
    assert HTMLExtractor().extract(data) == "# Article\nBody (/body)"


def test_html_whitespace_and_blocks_do_not_concatenate_words() -> None:
    data = b"<div>One</div><div>Two <em>inline</em> words.</div><p>Three\n  four</p>"
    assert HTMLExtractor().extract(data) == "One\nTwo inline words.\nThree four"


def test_html_skipped_self_closing_tags_and_stray_end_tags() -> None:
    data = b'<nav><div hidden/><h1>No</h1></aside><a href="/no">No</a></nav><p>Yes</p>'
    assert HTMLExtractor().extract(data) == "Yes"


def test_text_encoding_is_in_identity_options() -> None:
    supplied = {"custom": "value", "encoding": "incorrect"}
    utf8 = TextExtractor()
    latin1 = TextExtractor(encoding="latin-1", options=supplied)
    assert utf8.options == {"encoding": "utf-8"}
    assert latin1.options == {"custom": "value", "encoding": "latin-1"}
    assert supplied["encoding"] == "incorrect"
    assert latin1.extract(b"caf\xe9") == "caf\xe9"
    with pytest.raises(UnicodeDecodeError):
        utf8.extract(b"caf\xe9")
