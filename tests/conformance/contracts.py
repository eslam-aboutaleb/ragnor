"""Reusable adapter contract assertions for SDK extension authors.

Helpers are intentionally pytest-agnostic so third-party adapter packages can
reuse the invariants in their own test suites without depending on internals.
"""

from __future__ import annotations

import math

from ragnor.chunking.base import Chunker
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.evidence.tokenizer import Tokenizer
from ragnor.telemetry.events import EventSink, EventKind, TelemetryEvent


def check_chunker(chunker: Chunker, text: str = "alpha beta gamma") -> None:
    assert isinstance(chunker, Chunker)
    assert isinstance(chunker.version, str) and chunker.version
    assert isinstance(chunker.config, dict)
    chunks = chunker.chunk(text)
    assert chunks
    assert all(isinstance(chunk.text, str) and chunk.text for chunk in chunks)


def check_tokenizer(tokenizer: Tokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(tokenizer.name, str) and tokenizer.name
    assert isinstance(tokenizer.is_exact, bool)
    assert tokenizer.count_tokens("") == 0
    for content in ("alpha beta", "مرحبا بالعالم", "hello 🌍"):
        count = tokenizer.count_tokens(content)
        assert isinstance(count, int) and count >= 0


async def check_embedder(embedder: EmbeddingFunction, dimension: int) -> None:
    assert isinstance(embedder, EmbeddingFunction)
    texts = ["alpha", "beta"]
    vectors = await embedder(texts)
    assert len(vectors) == len(texts)
    assert all(
        len(vec) == dimension and all(math.isfinite(component) for component in vec)
        for vec in vectors
    )
    query = await embedder.embed_query("alpha")
    assert len(query) == dimension
    docs = await embedder.embed_documents(texts)
    assert len(docs) == len(texts) and all(len(v) == dimension for v in docs)


def check_event_sink(sink: EventSink) -> None:
    assert isinstance(sink, EventSink)
    event = TelemetryEvent(event_kind=EventKind.SEARCH_COMPLETED, stage="retrieval")
    sink.emit(event)
