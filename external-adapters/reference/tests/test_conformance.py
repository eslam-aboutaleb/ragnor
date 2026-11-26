"""The reference adapters pass Ragnor's published conformance suite."""

from __future__ import annotations

from ragnor.conformance import (
    check_chunker,
    check_document_source,
    check_embedder,
    check_event_sink,
    check_tokenizer,
)
from ragnor_reference_adapters import (
    EMBEDDING_DIMENSION,
    ReferenceChunker,
    ReferenceDocumentSource,
    ReferenceEmbedder,
    ReferenceEventSink,
    ReferenceTokenizer,
)


def test_reference_chunker_conforms() -> None:
    check_chunker(ReferenceChunker(), text="first section\n\nsecond section")


def test_reference_tokenizer_conforms() -> None:
    check_tokenizer(ReferenceTokenizer())


async def test_reference_embedder_conforms() -> None:
    await check_embedder(ReferenceEmbedder(), EMBEDDING_DIMENSION)


def test_reference_event_sink_conforms() -> None:
    sink = ReferenceEventSink()
    check_event_sink(sink)
    assert len(sink.events) == 1


async def test_reference_document_source_conforms() -> None:
    await check_document_source(ReferenceDocumentSource())
