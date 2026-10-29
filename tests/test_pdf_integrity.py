"""Real PDF regressions: textless content cannot authorize replacement."""

from io import BytesIO

import pytest
from hypothesis import given, strategies as st

from ragnor.ingestion.connector_core import extract_documents
from ragnor.ingestion.extractors import PDFExtractor, PageExtractionOutcome, PageExtractionStatus
from tests.test_extractors import make_pdf


def mixed_pdf(*, vector=False, text_on_image=False):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

    writer = PdfWriter()
    reader = PdfReader(BytesIO(make_pdf("First", "overlay" if text_on_image else None, "Third")))
    for page in reader.pages:
        writer.add_page(page)
    page = writer.pages[1]
    stream = DecodedStreamObject()
    if vector:
        stream.set_data(b"0 0 100 100 re f")
    else:
        image = DecodedStreamObject()
        image.set_data(b"\x00")
        image.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(1),
                NameObject("/Height"): NumberObject(1),
                NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                NameObject("/BitsPerComponent"): NumberObject(8),
            }
        )
        resources = page.get("/Resources", DictionaryObject())
        resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): image})
        page[NameObject("/Resources")] = resources
        stream.set_data(b"q 300 0 0 300 0 0 cm /Im0 Do Q")
    if text_on_image:
        stream.set_data(page.get_contents().get_data() + b"\n" + stream.get_data())
    page[NameObject("/Contents")] = stream
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.mark.parametrize(
    "vector,text_on_image", [(False, False), (True, False), (False, True), (True, True)]
)
def test_unknown_content_fails_entire_document(vector, text_on_image):
    result = extract_documents(
        mixed_pdf(vector=vector, text_on_image=text_on_image), "invoice.pdf", "invoice", {}, {}
    )
    assert not result.authoritative
    assert result == []  # no page from a rejected revision can reach sync
    assert result.status == "incomplete"
    assert not result.skipped
    assert result.metadata["extraction_quality"]["complete"] is False
    assert result.errors[0]["page"] == 2
    assert "overlay" not in str(result.errors)


def test_verified_blank_is_auditable_and_legacy_tuples_survive():
    data = make_pdf("First", None, "Third")
    assert PDFExtractor().extract_pages(data) == [(1, "First"), (2, ""), (3, "Third")]
    result = extract_documents(data, "book.pdf", "book", {}, {})
    assert result.authoritative
    assert result.skipped[0].reason == "verified_blank"
    assert result.metadata["extraction_quality"]["complete"] is True


@pytest.mark.parametrize("policy", ["allow_partial", "review"])
def test_explicit_policy_never_certifies_incomplete_content(policy):
    result = extract_documents(
        mixed_pdf(), "book.pdf", "book", {}, {}, PDFExtractor(options={"policy": policy})
    )
    assert not result.authoritative
    assert len(result) == (2 if policy == "allow_partial" else 0)
    if result:
        assert result[0].metadata["extraction_quality"]["complete"] is False


def test_empty_pdf_is_not_authoritative_deletion():
    result = extract_documents(make_pdf(), "empty.pdf", "empty", {}, {})
    assert not result.authoritative
    assert not result


@pytest.mark.asyncio
async def test_directory_cannot_bless_mixed_pdf(tmp_path):
    from ragnor.ingestion.directory_connector import DirectoryConnector

    (tmp_path / "invoice.pdf").write_bytes(mixed_pdf())
    result = await DirectoryConnector(tmp_path).load()
    assert not result.authoritative
    assert not result
    assert result.errors


@pytest.mark.parametrize("kind", ["encrypted", "malformed", "limit", "oversized"])
def test_unsafe_pdf_inputs_have_safe_errors(kind):
    from pypdf import PdfReader, PdfWriter

    data = make_pdf("private document text")
    extractor = PDFExtractor()
    if kind == "encrypted":
        writer = PdfWriter()
        writer.add_page(PdfReader(BytesIO(data)).pages[0])
        writer.encrypt("secret-password")
        out = BytesIO()
        writer.write(out)
        data = out.getvalue()
    elif kind == "malformed":
        data = b"%PDF-1.4\nsecret-password"
    elif kind == "limit":
        data = make_pdf("one", "two")
        extractor = PDFExtractor(options={"max_pages": 1})
    else:
        extractor = PDFExtractor(options={"max_bytes": 1})
    result = extract_documents(data, "private.pdf", "private", {}, {}, extractor)
    assert not result.authoritative
    assert not result
    assert result.errors
    assert "secret-password" not in str(result.errors)
    assert "private document text" not in str(result.errors)


def test_policy_changes_fingerprint_without_changing_plain_text():
    strict = extract_documents(make_pdf("one"), "x.pdf", "x", {}, {}, PDFExtractor())
    partial = extract_documents(
        make_pdf("one"), "x.pdf", "x", {}, {}, PDFExtractor(options={"policy": "allow_partial"})
    )
    assert strict[0].metadata["extraction"] != partial[0].metadata["extraction"]
    plain = extract_documents(b"plain", "x.txt", "x", {}, {})
    assert "extraction_quality" not in plain[0].metadata


@pytest.mark.asyncio
async def test_object_connector_cannot_bless_mixed_pdf():
    from ragnor.ingestion.object_connector import ObjectStorageConnector
    from tests.test_object_connectors import Client

    data = mixed_pdf()
    client = Client(
        pages={None: {"Contents": [{"Key": "x.pdf", "Size": len(data)}], "IsTruncated": False}},
        objects={"x.pdf": data},
    )
    result = await ObjectStorageConnector("bucket", client=client).load()
    assert not result.authoritative
    assert not result
    assert result.metadata["extraction_quality"]["complete"] is False
    assert result.errors[0]["page"] == 2


@pytest.mark.asyncio
async def test_http_connector_cannot_bless_mixed_pdf(monkeypatch):
    import socket
    import httpx
    from ragnor.ingestion.http_connector import HttpApiConnector

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    connector = HttpApiConnector(
        "https://example.com/x.pdf",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=mixed_pdf(), headers={"content-type": "application/pdf"}
            )
        ),
    )
    result = await connector.load()
    assert not result.authoritative
    assert not result
    assert result.metadata["extraction_quality"]["complete"] is False
    assert result.errors[0]["page"] == 2


@pytest.mark.asyncio
async def test_source_adapter_preserves_extraction_failure():
    from ragnor.ingestion.sources import DocumentSourceConnector
    from types import SimpleNamespace

    result = extract_documents(mixed_pdf(), "x.pdf", "x", {}, {})

    async def load():
        return result

    loaded = await DocumentSourceConnector(SimpleNamespace(load=load)).load()
    assert not loaded.authoritative
    assert not loaded
    assert loaded.metadata["extraction_quality"]["complete"] is False


def test_legacy_page_iterator_failure_does_not_leak_partial_revision():
    from types import SimpleNamespace

    def pages(data):
        yield 1, "text that must not escape"
        raise ValueError("sensitive provider message")

    extractor = SimpleNamespace(id="custom", version="1", options={}, extract_pages=pages)
    result = extract_documents(b"data", "x.pdf", "x", {}, {}, extractor)
    assert not result.authoritative
    assert not result
    assert "sensitive" not in str(result.errors)


@given(st.lists(st.sampled_from(list(PageExtractionStatus)), min_size=1, max_size=20))
def test_strict_status_aggregation_is_atomic_and_ordered(statuses):
    from types import SimpleNamespace

    outcomes = [
        PageExtractionOutcome(
            number, status, "text" if status == "extracted" else "", reason="synthetic"
        )
        for number, status in enumerate(statuses, 1)
    ]
    extractor = SimpleNamespace(
        id="test", version="1", options={}, extract_outcomes=lambda data: outcomes
    )
    result = extract_documents(b"synthetic", "x.pdf", "x", {}, {}, extractor)
    complete = all(outcome.complete for outcome in outcomes)
    assert result.authoritative == complete
    assert sum(result.metadata["extraction_quality"]["counts"].values()) == len(statuses)
    if not complete:
        assert not result
    else:
        assert [doc.location["page"] for doc in result] == [
            outcome.page for outcome in outcomes if outcome.status == "extracted"
        ]


@pytest.mark.asyncio
async def test_directory_size_limit_cannot_authorize_pdf_deletion(tmp_path):
    from ragnor.ingestion.directory_connector import DirectoryConnector

    (tmp_path / "large.pdf").write_bytes(make_pdf("existing file grew too large"))
    result = await DirectoryConnector(tmp_path, max_file_size=1).load()
    assert not result.authoritative
    assert result.errors[0]["reason"] == "pdf_size_limit"
    assert not result.skipped


@pytest.mark.asyncio
async def test_object_size_limit_cannot_authorize_pdf_deletion():
    from ragnor.ingestion.object_connector import ObjectStorageConnector
    from tests.test_object_connectors import Client

    client = Client(
        pages={None: {"Contents": [{"Key": "large.pdf", "Size": 100}], "IsTruncated": False}}
    )
    result = await ObjectStorageConnector("bucket", client=client, max_file_size=1).load()
    assert not result.authoritative
    assert result.errors[0]["reason"] == "pdf_size_limit"
    assert not result.skipped


def test_policy_exclusion_never_emits_text():
    from types import SimpleNamespace

    extractor = SimpleNamespace(
        id="host",
        version="1",
        options={},
        extract_outcomes=lambda data: [
            PageExtractionOutcome(1, PageExtractionStatus.POLICY_EXCLUDED, "excluded content")
        ],
    )
    result = extract_documents(b"data", "x.pdf", "x", {}, {}, extractor)
    assert result.authoritative
    assert not result
    assert result.skipped[0].reason == "policy_excluded"
