"""Published conformance suite for Ragnor adapter ports.

These zero-dependency checks are the contract third-party adapters are
expected to satisfy. They ship inside the ``ragnor`` distribution so an
independently authored adapter package can run them against an installed
wheel -- from its own test suite, without editing Ragnor and without
importing Ragnor's test tree. See docs/architecture/external-adapters.md
and the reference package under ``external-adapters/reference/``.

Every check raises ``AssertionError`` naming the violated invariant, so the
suite composes with any test runner. The checks are intentionally
behavioral, not ``isinstance``-only: a protocol match alone never proves
batch cardinality, token-count sanity or event delivery.
"""

from __future__ import annotations

import math

from ragnor.chunking.base import Chunker
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.evidence.tokenizer import Tokenizer
from ragnor.ingestion.records import RawDocument
from ragnor.ingestion.sources import DocumentSource
from ragnor.telemetry.events import EventKind, EventSink, TelemetryEvent

__all__ = [
    "check_chunker",
    "check_document_source",
    "check_embedder",
    "check_event_sink",
    "check_tokenizer",
]


def check_chunker(chunker: Chunker, text: str = "alpha beta gamma") -> None:
    """Verify the Chunker contract: structure, version/config, nonempty output."""
    assert isinstance(chunker, Chunker), "chunker must satisfy the Chunker protocol"
    assert isinstance(chunker.version, str) and chunker.version, (
        "chunker.version must be a non-empty string"
    )
    assert isinstance(chunker.config, dict), "chunker.config must be a dict"
    chunks = chunker.chunk(text)
    assert chunks, "chunker must return at least one chunk for non-empty input"
    assert all(isinstance(chunk.text, str) and chunk.text for chunk in chunks), (
        "every chunk must carry non-empty string text"
    )


def check_tokenizer(tokenizer: Tokenizer) -> None:
    """Verify the Tokenizer contract: structure and sane integer counts."""
    assert isinstance(tokenizer, Tokenizer), "tokenizer must satisfy the Tokenizer protocol"
    assert isinstance(tokenizer.name, str) and tokenizer.name, (
        "tokenizer.name must be a non-empty string"
    )
    assert isinstance(tokenizer.is_exact, bool), "tokenizer.is_exact must be a bool"
    assert tokenizer.count_tokens("") == 0, "empty input must count as zero tokens"
    for content in ("alpha beta", "مرحبا بالعالم", "hello 🌍"):
        count = tokenizer.count_tokens(content)
        assert isinstance(count, int) and count >= 0, (
            f"token count for {content!r} must be a non-negative integer, got {count!r}"
        )


async def check_embedder(embedder: EmbeddingFunction, dimension: int) -> None:
    """Verify the EmbeddingFunction contract: batch cardinality and dimensions."""
    assert isinstance(embedder, EmbeddingFunction), (
        "embedder must satisfy the EmbeddingFunction protocol"
    )
    texts = ["alpha", "beta"]
    vectors = await embedder(texts)
    assert len(vectors) == len(texts), (
        f"batch call must return one vector per input: {len(vectors)} != {len(texts)}"
    )
    assert all(
        len(vec) == dimension and all(math.isfinite(component) for component in vec)
        for vec in vectors
    ), f"every vector must have {dimension} finite components"
    query = await embedder.embed_query("alpha")
    assert len(query) == dimension, f"embed_query must return {dimension} components"
    docs = await embedder.embed_documents(texts)
    assert len(docs) == len(texts) and all(len(v) == dimension for v in docs), (
        "embed_documents must preserve batch cardinality and dimension"
    )


def check_event_sink(sink: EventSink) -> None:
    """Verify the EventSink contract: emit() accepts a TelemetryEvent."""
    assert isinstance(sink, EventSink), "sink must satisfy the EventSink protocol"
    event = TelemetryEvent(event_kind=EventKind.SEARCH_COMPLETED, stage="retrieval")
    sink.emit(event)


async def check_document_source(source: DocumentSource) -> None:
    """Verify the DocumentSource contract: typed records with stable identity.

    ``load()`` must return :class:`RawDocument` records that preserve the
    source identity (``source_name``, ``external_id`` when provided) and
    carry string content. Per the port contract a source raises on failure,
    so a successful load returning records is checked for record integrity.
    """
    assert isinstance(source, DocumentSource), "source must satisfy the DocumentSource protocol"
    documents = await source.load()
    assert isinstance(documents, list), "load() must return a list of RawDocument records"
    for document in documents:
        assert isinstance(document, RawDocument), (
            f"load() must return RawDocument records, got {type(document).__name__}"
        )
        assert isinstance(document.source_name, str) and document.source_name, (
            "every record must carry a non-empty source_name"
        )
        assert isinstance(document.content, str), "every record must carry string content"
