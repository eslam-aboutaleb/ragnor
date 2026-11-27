"""Legacy extraction API stays stable; structured mode is explicit opt-in."""

from __future__ import annotations

import inspect

from ragnor.extraction import Capability, ProviderRegistry, StructuredExtractor
from ragnor.ingestion import extractors
from ragnor.ingestion.connector_core import extract_documents


def test_default_connectors_do_not_route_through_the_structured_registry():
    source = inspect.getsource(extract_documents)
    assert "ragnor.extraction" not in source
    result = extract_documents(b"hello world", "a.txt", None, {}, {})
    assert [d.content for d in result.documents] == ["hello world"]
    assert result.documents[0].metadata["extraction"]["id"] == "text"


def test_legacy_extractor_signatures_are_unchanged():
    assert list(inspect.signature(extractors.Extractor.extract).parameters) == [
        "self",
        "data",
        "name",
    ]
    assert list(inspect.signature(extractors.PDFExtractor.extract_pages).parameters) == [
        "self",
        "data",
    ]
    assert extractors.PDFExtractor().id == "pdf" and extractors.PDFExtractor().version == "3"


def test_structured_extractor_is_opt_in_and_usable_by_connector_core():
    registry = ProviderRegistry()

    class TextProvider:
        id = "plain"
        version = "1"
        options: dict = {}

        async def extract(self, data, *, name="", revision):
            from ragnor.extraction import ExtractionResult, Page, Region, RegionKind

            page = Page(1, 100.0, 100.0)
            region = Region("r1", RegionKind.TEXT, 1, page.box(1, 1, 99, 99), 0, text=data.decode())
            return ExtractionResult(revision, (Page(1, 100.0, 100.0, (region,)),))

    registry.register(
        "plain",
        lambda o: TextProvider(),
        formats=["text/plain"],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
    )
    extractor = StructuredExtractor(registry)
    assert extractor.extract(b"structured hello", name="a.txt") == "structured hello"
    assert extractor.options["providers"] == [{"id": "plain", "version": "1", "options": {}}]
