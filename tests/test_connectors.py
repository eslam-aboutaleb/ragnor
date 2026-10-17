"""Public connector API and cross-module extraction integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from ragit.ingestion.connectors import (
    ChangeEventSource,
    ConnectorDocumentSource,
    DirectoryConnector,
    FilesystemWatcher,
    HttpApiConnector,
    IncrementalSourceConnector,
    ObjectStorageConnector,
    PDFExtractor,
    PeriodicReconciler,
    SourceConnector,
    SourceStatus,
)
from ragit.ingestion.pipeline import IngestionPipeline
from ragit.ingestion.versioning import build_snapshot
from tests.test_extractors import make_pdf


def test_public_exports_and_protocol(tmp_path):
    import ragit.ingestion.connectors as api

    for module_name in ("ragit.connectors", "ragit.ingestion"):
        module = importlib.import_module(module_name)
        for name in api.__all__:
            assert getattr(module, name) is getattr(api, name)
    connector = DirectoryConnector(tmp_path)
    assert isinstance(connector, SourceConnector)
    assert isinstance(connector, IncrementalSourceConnector)
    assert isinstance(connector, ChangeEventSource)
    assert isinstance(FilesystemWatcher(connector), ChangeEventSource)
    assert isinstance(HttpApiConnector("https://example.com"), IncrementalSourceConnector)
    assert isinstance(HttpApiConnector("https://example.com"), ChangeEventSource)
    assert isinstance(
        ObjectStorageConnector("bucket", client=SimpleNamespace()), IncrementalSourceConnector
    )


async def test_supported_fixture_formats_and_locations(tmp_path):
    (tmp_path / "a.md").write_text("# A\nhello")
    (tmp_path / "b.txt").write_text("plain text")
    (tmp_path / "page.html").write_text(
        '<nav><a href="/nav">navigation</a></nav>'
        '<article><h1>Title</h1><p><a href="/detail">link</a></p></article>'
    )
    (tmp_path / "pages.pdf").write_bytes(make_pdf("First page", None, "Third page"))
    result = await DirectoryConnector(tmp_path).load()
    assert result.authoritative
    documents = {document.source_name: document.content for document in result}
    assert documents["a.md"] == "# A\nhello"
    assert documents["b.txt"] == "plain text"
    assert "Title" in documents["page.html"]
    assert "/detail" in documents["page.html"]
    assert "navigation" not in documents["page.html"]
    assert "/nav" not in documents["page.html"]
    pages = [document for document in result if document.source_name == "pages.pdf"]
    assert [document.location for document in pages] == [{"page": 1}, {"page": 3}]
    assert any(skip.location == {"page": 2} for skip in result.skipped)


async def test_policy_skips_and_empty_vs_missing(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "secret.log").write_text("ignore me")
    (tmp_path / "binary.txt").write_bytes(b"\x00\x01\x02")
    (tmp_path / "large.txt").write_text("x" * 100)
    result = await DirectoryConnector(tmp_path, max_bytes=40).load()
    reasons = {skip.path: skip.reason for skip in result.skipped}
    assert reasons["secret.log"] == "ignored"
    assert reasons["binary.txt"] == "binary"
    assert reasons["large.txt"] == "oversized"
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = await DirectoryConnector(empty_root).load()
    missing = await DirectoryConnector(tmp_path / "missing").load()
    assert empty.status == SourceStatus.EMPTY and empty.authoritative
    assert missing.status == SourceStatus.NOT_FOUND and not missing.authoritative


async def test_interruption_is_not_authoritative(tmp_path):
    for number in range(5):
        (tmp_path / f"{number}.txt").write_text(str(number))
    connector = DirectoryConnector(tmp_path, page_size=1)
    interrupted = await connector.load_all(max_pages=1)
    assert len(interrupted) == 1
    assert not interrupted.authoritative
    assert interrupted.next_cursor is not None
    page = await connector.load_page(cursor=interrupted.next_cursor, limit=1)
    assert not page.authoritative


def test_extractor_identity_in_snapshot():
    settings = SimpleNamespace(embedding_model="text-embedding-3-small")
    chunker = SimpleNamespace(version="v1", config={"chunk_size": 10, "overlap": 0})
    extractor = PDFExtractor(options={"max_pages": 50})
    snapshot = build_snapshot("h", settings, chunker, extractor=extractor)
    assert snapshot.extractor_id == "pdf"
    assert snapshot.extractor_version == extractor.version
    assert snapshot.extractor_options["max_pages"] == 50


async def test_directory_incremental_loading_and_checkpoint(tmp_path):
    import time
    from unittest.mock import AsyncMock, Mock

    file_a = tmp_path / "a.txt"
    file_a.write_text("initial a")
    connector = DirectoryConnector(tmp_path)

    # Initial incremental load returns all docs, incremental=True, and valid checkpoint
    first = await connector.load_incremental(checkpoint=None)
    assert first.incremental
    assert len(first) == 1
    assert first[0].source_name == "a.txt"
    assert first.checkpoint is not None
    initial_cp = first.checkpoint

    # Without new files or changes, subsequent incremental load is empty with reason=unmodified
    second = await connector.load_incremental(checkpoint=initial_cp)
    assert second.incremental
    assert len(second) == 0
    assert second.status == SourceStatus.EMPTY
    assert any(s.reason == "unmodified" for s in second.skipped)

    # Adding a new file updates checkpoint and yields only the new file
    time.sleep(0.05)
    file_b = tmp_path / "b.txt"
    file_b.write_text("new b")
    third = await connector.load_incremental(checkpoint=initial_cp)
    assert third.incremental
    assert len(third) == 1
    assert third[0].source_name == "b.txt"
    assert third.checkpoint is not None and float(third.checkpoint) > float(initial_cp)

    # Incremental results are non-authoritative so pipeline never mutates or purges on them
    pipeline = IngestionPipeline(Mock(), AsyncMock(), Mock(), Mock(), Mock())
    versions = SimpleNamespace(find_active=AsyncMock(), close_active=AsyncMock())
    assert await pipeline.run(third, versions, purge_on_empty=True) == 0
    versions.find_active.assert_not_awaited()


async def test_directory_change_events_and_watcher(tmp_path):
    connector = DirectoryConnector(tmp_path)
    watcher = FilesystemWatcher(connector, interval=0.01)

    # Changes method satisfies ChangeEventSource and yields events with checkpoints
    events = []
    async for event in watcher.changes(checkpoint="cp-1"):
        events.append(event)
        if len(events) >= 1:
            break
    assert len(events) == 1
    assert events[0].type == "sync_requested"
    assert events[0].checkpoint == "cp-1"


async def test_connector_document_source_forwards_extractor(tmp_path):
    extractor = PDFExtractor()
    connector = DirectoryConnector(tmp_path, extractor=extractor)
    adapter = ConnectorDocumentSource(connector)
    assert adapter.extractor is extractor
    assert adapter.root == connector.root

    (tmp_path / "doc.pdf").write_bytes(make_pdf("sample text"))
    loaded = await adapter.load()
    assert loaded.authoritative
    assert adapter.extractor is extractor


async def test_periodic_reconciler_sync_with_pipeline(tmp_path):
    from unittest.mock import AsyncMock

    (tmp_path / "file.txt").write_text("hello")
    connector = DirectoryConnector(tmp_path)
    reconciler = PeriodicReconciler(connector, interval=0.1)

    pipeline = SimpleNamespace(run=AsyncMock(return_value=1))
    versions = SimpleNamespace()
    count = await reconciler.sync(pipeline, versions)
    assert count == 1
    pipeline.run.assert_awaited_once()
