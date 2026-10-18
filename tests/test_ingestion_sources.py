"""Tests for ragnor.ingestion.sources."""

from __future__ import annotations

import pytest

from ragnor.ingestion import (
    DocumentSource,
    FileDocumentSource,
    RawDocument,
)


class _StubSource:
    """Minimal DocumentSource implementation."""

    def __init__(self, *documents: RawDocument) -> None:
        self._documents = list(documents)

    async def load(self) -> list[RawDocument]:
        return self._documents


def test_raw_document_defaults() -> None:
    document = RawDocument(source_name="policy.md", content="text")
    assert document.source_name == "policy.md"
    assert document.content == "text"
    assert document.external_id is None


def test_document_source_protocol_is_runtime_checkable() -> None:
    assert isinstance(_StubSource(), DocumentSource)
    assert isinstance(FileDocumentSource("x.md"), DocumentSource)


@pytest.mark.asyncio
async def test_file_source_loads_the_file_as_one_document(tmp_path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nContent.", encoding="utf-8")

    documents = await FileDocumentSource(str(path)).load()

    assert len(documents) == 1
    assert documents[0].source_name == "policy.md"
    assert documents[0].content == "# Policy\n\nContent."
    assert documents[0].external_id is None


@pytest.mark.asyncio
async def test_file_source_missing_file_yields_nothing(tmp_path) -> None:
    documents = await FileDocumentSource(str(tmp_path / "absent.md")).load()
    assert documents == []


@pytest.mark.asyncio
async def test_stub_source_loads_its_documents() -> None:
    documents = [
        RawDocument(source_name="a.md", content="a", external_id="ext-1"),
        RawDocument(source_name="b.md", content="b"),
    ]
    assert await _StubSource(*documents).load() == documents
