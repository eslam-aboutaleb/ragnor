"""Shared connector contracts, adapters, fingerprints, and pipeline safety."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ragnor.ingestion.connector_core import (
    ChangeEvent,
    DiscoveryPage,
    LoadResult,
    SkipRecord,
    SourceItem,
    SourceStatus,
    collect_pages,
    extract_documents,
)
from ragnor.ingestion.pipeline import IngestionPipeline
from ragnor.ingestion.sources import (
    ConnectorDocumentSource,
    DocumentSourceConnector,
    FileDocumentSource,
    ListDocumentSource,
    RawDocument,
)
from ragnor.ingestion.versioning import build_snapshot, compute_source_hash
from ragnor.types import Chunk


def test_records_and_list_compatibility():
    item = SourceItem(path="a.md")
    item.path = "b.md"
    assert item["path"] == "b.md"
    assert SkipRecord("a", "ignored").reason == "ignored"
    assert ChangeEvent(type="deleted").type == "deleted"
    with pytest.raises(AttributeError):
        _ = item.missing
    page = DiscoveryPage([item], metadata={"scan": "x"})
    assert page == page.items == [item]
    result = LoadResult([RawDocument("a", "text")], skipped=[SkipRecord("x", "excluded")])
    assert result.documents == list(result)
    assert result.authoritative
    assert not result.errors


@pytest.mark.parametrize("status", list(SourceStatus))
def test_authority_requires_success_status(status):
    assert LoadResult(status=status).authoritative == (
        status
        in {
            SourceStatus.OK,
            SourceStatus.SUCCESS,
            SourceStatus.EMPTY,
            SourceStatus.EMPTY_SUCCESS,
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"complete": False},
        {"errors": ["failed"]},
        {"incremental": True},
        {"has_more": True},
        {"cursor": "continuation"},
    ],
)
def test_authority_rejects_partial_errors_and_incremental(kwargs):
    assert not LoadResult(**kwargs).authoritative


def paginated_connector():
    return SimpleNamespace(
        load_page=AsyncMock(
            side_effect=[
                LoadResult(
                    [RawDocument("a", "one")],
                    next_cursor="next",
                    has_more=True,
                    skipped=[SkipRecord("policy", "excluded")],
                    total_items=1,
                ),
                LoadResult([RawDocument("b", "two")], cursor="next", total_items=1),
            ]
        )
    )


async def test_collect_pages_certifies_entire_traversal():
    connector = paginated_connector()
    result = await collect_pages(connector, page_size=1)
    assert result.authoritative
    assert result.pages == 2
    assert result.total_items == 2
    assert len(result) == 2
    assert len(result.skipped) == 1
    assert not result.errors
    connector.load_page.assert_any_await(cursor="next", limit=1)


async def test_collect_pages_bound_exposes_continuation():
    result = await collect_pages(paginated_connector(), max_pages=1)
    assert len(result) == 1
    assert result.next_cursor == "next"
    assert result.has_more
    assert not result.authoritative
    assert result.status == SourceStatus.INCOMPLETE


@pytest.mark.parametrize("token", [None, "again"])
async def test_collect_pages_missing_or_cyclic_tokens(token):
    connector = SimpleNamespace(
        load_page=AsyncMock(
            return_value=LoadResult(
                [RawDocument("a", "text")],
                has_more=True,
                next_cursor=token,
            )
        )
    )
    result = await collect_pages(connector)
    assert not result.authoritative
    assert result.errors
    assert result.pages <= 2


async def test_collect_pages_errors_never_authoritative():
    connector = paginated_connector()
    connector.load_page.side_effect = [
        LoadResult([RawDocument("a", "text")], has_more=True, next_cursor="next"),
        OSError("unavailable"),
    ]
    result = await collect_pages(connector)
    assert result.next_cursor == "next"
    assert len(result) == 1
    assert not result.authoritative
    assert result.errors
    assert not result.skipped


async def test_collect_pages_preserves_safe_retry_metadata():
    connector = SimpleNamespace(
        load_page=AsyncMock(
            return_value=LoadResult(
                status=SourceStatus.TRANSIENT,
                complete=False,
                errors=[
                    {
                        "reason": "rate_limit",
                        "status_code": 429,
                        "retry_after": 2.5,
                        "error_type": "HTTPStatusError",
                        "message": "secret-provider-key",
                    }
                ],
            )
        )
    )
    result = await collect_pages(connector)
    assert result.status == SourceStatus.TRANSIENT
    assert result.errors[0]["status_code"] == 429
    assert result.errors[0]["retry_after"] == 2.5
    assert "secret-provider-key" not in json.dumps(result.errors)


@pytest.mark.parametrize("kwargs", [{"max_pages": 0}, {"page_size": 0}])
async def test_collect_pages_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        await collect_pages(paginated_connector(), **kwargs)


async def test_adapters_paginate_without_losing_authority():
    documents = [RawDocument("a", "one"), RawDocument("b", "two")]
    legacy = ListDocumentSource(documents)
    assert await legacy.load() == documents
    connector = DocumentSourceConnector(legacy)
    page = await connector.discover(limit=1)
    assert not page.complete
    assert page.items[0].source_name == "a"
    result = await connector.load(limit=1)
    assert result == documents
    assert result.authoritative
    assert result.pages == 2
    bridge = ConnectorDocumentSource(SimpleNamespace(load=AsyncMock(return_value=result)))
    assert await bridge.load() is result


async def test_file_adapter_detects_missing_without_changing_legacy(tmp_path):
    source = FileDocumentSource(str(tmp_path / "missing.md"))
    assert await source.load() == []
    result = await DocumentSourceConnector(source).load()
    assert result.errors
    assert not result.authoritative


async def test_adapter_preserves_non_authoritative_result():
    partial = LoadResult([RawDocument("a", "text")], complete=False)
    source = SimpleNamespace(load=AsyncMock(return_value=partial))
    result = await DocumentSourceConnector(source).load()
    assert not result.authoritative
    assert result == partial


def test_legacy_hash_is_byte_for_byte_compatible():
    document = RawDocument("a", "text", "id")
    manifest = {
        "documents": [
            {
                "source_name": "a",
                "external_id": "id",
                "content_hash": hashlib.sha256(b"text").hexdigest(),
            }
        ]
    }
    old_hash = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert compute_source_hash([document]) == old_hash


def test_extraction_and_provenance_invalidate_fingerprints():
    first = extract_documents(b"text", "a.txt", "id", {}, {"revision": "one"})[0]
    assert first.metadata["extraction"]["id"] == "text"
    second = replace(
        first,
        metadata={
            "extraction": {
                **first.metadata["extraction"],
                "version": "new",
            }
        },
    )
    assert compute_source_hash([first]) != compute_source_hash([second])
    assert compute_source_hash([first]) != compute_source_hash(
        [
            replace(first, provenance={"revision": "two"}),
        ]
    )
    assert compute_source_hash([first]) != compute_source_hash(
        [
            replace(first, location={"page": 2}),
        ]
    )


def test_pdf_pages_accept_binary_and_record_blank_page_skip():
    extractor = SimpleNamespace(
        id="pdf",
        version="test",
        options={"max_pages": 4},
        extract_pages=lambda data: [(1, "first"), (2, ""), (3, "third")],
    )
    result = extract_documents(b"%PDF\x00binary", "book.pdf", "book", {}, {}, extractor)
    assert result.authoritative
    assert [doc.location for doc in result] == [{"page": 1}, {"page": 3}]
    assert result.skipped[0].location == {"page": 2}
    assert result.skipped[0].reason == "no_text"


def test_extract_failure_is_error_not_intentional_skip():
    extractor = SimpleNamespace(
        id="broken", version="1", options={}, extract=Mock(side_effect=ValueError("bad format"))
    )
    result = extract_documents(b"bad", "a", None, {}, {}, extractor)
    assert not result.authoritative
    assert result.errors
    assert not result.skipped


def test_extraction_failures_do_not_expose_provider_messages():
    extractor = SimpleNamespace(
        id="broken",
        version="1",
        options={},
        extract=Mock(side_effect=ValueError("authorization=secret-provider-key")),
    )
    result = extract_documents(b"bad", "a", None, {}, {}, extractor)
    assert "secret-provider-key" not in json.dumps(result.errors)
    assert result.errors == [{"reason": "source_failure", "error_type": "ValueError"}]


def test_binary_skip_and_explicit_encoding():
    from ragnor.ingestion.extractors import TextExtractor

    binary = extract_documents(b"\x00\x01", "a.bin", None, {}, {})
    assert binary.authoritative
    assert binary.skipped[0].reason == "binary"
    text = extract_documents(
        "hello".encode("utf-16"), "a.txt", None, {}, {}, TextExtractor(encoding="utf-16")
    )
    assert text.authoritative
    assert text[0].content == "hello"
    assert text[0].metadata["extraction"]["options"]["encoding"] == "utf-16"


async def test_adapter_preserves_intentional_skips_and_metadata():
    scan = LoadResult(
        [RawDocument("a", "text")], skipped=[SkipRecord("b", "excluded")], metadata={"scan": "one"}
    )
    connector = DocumentSourceConnector(SimpleNamespace(load=AsyncMock(return_value=scan)))
    result = await connector.load(limit=1)
    assert result.authoritative
    assert result.skipped == scan.skipped
    assert result.metadata == scan.metadata


def test_snapshot_configuration_never_serializes_credentials_or_objects():
    extractor = SimpleNamespace(
        id="custom",
        version="2",
        options={
            "mode": "text",
            "max_tokens": 100,
            "api_key": "secret",
            "client": object(),
            "nested": {"password": "secret", "size": 10},
        },
    )
    snapshot = build_snapshot(
        "hash",
        SimpleNamespace(embedding_model="text-embedding-3-small"),
        SimpleNamespace(config={}, version="1"),
        extractor=extractor,
    )
    assert snapshot.extractor_id == "custom"
    assert snapshot.extractor_version == "2"
    assert snapshot.extractor_options == {
        "mode": "text",
        "max_tokens": 100,
        "nested": {"size": 10},
    }


@pytest.mark.parametrize("documents", [[], [RawDocument("a", "nonempty")]])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"complete": False},
        {"status": SourceStatus.ERROR},
        {"errors": ["failure"]},
        {"incremental": True, "checkpoint": "new"},
    ],
)
async def test_pipeline_non_authoritative_never_mutates_or_purges(documents, kwargs):
    pipeline = IngestionPipeline(Mock(), AsyncMock(), Mock(), Mock(), Mock())
    versions = SimpleNamespace(
        find_active=AsyncMock(),
        close_active=AsyncMock(),
        delete_chunks=AsyncMock(),
        create_version=AsyncMock(),
    )
    source = SimpleNamespace(load=AsyncMock(return_value=LoadResult(documents, **kwargs)))
    assert await pipeline.run(source, versions, purge_on_empty=True) == 0
    versions.find_active.assert_not_awaited()
    versions.close_active.assert_not_awaited()
    versions.delete_chunks.assert_not_awaited()
    versions.create_version.assert_not_awaited()


def test_chunk_metadata_and_ids_preserve_legacy_single_document():
    chunker = SimpleNamespace(chunk=lambda text: [Chunk("chunk-0", text, {"heading": "Title"})])
    pipeline = IngestionPipeline(Mock(), AsyncMock(), chunker, Mock(), Mock())
    doc = RawDocument("a.pdf", "text", "a", {"owner": "me"}, {"revision": "v1"}, "pdf", {"page": 1})
    single = pipeline._chunk([doc])
    assert single[0].id == "chunk-0"
    assert single[0].metadata == {
        "heading": "Title",
        "owner": "me",
        "source": "a.pdf",
        "external_id": "a",
        "provenance": {"revision": "v1"},
        "extractor": "pdf",
        "location": {"page": 1},
    }
    multi = pipeline._chunk([doc, replace(doc, location={"page": 2})])
    assert len({chunk.id for chunk in multi}) == 2
