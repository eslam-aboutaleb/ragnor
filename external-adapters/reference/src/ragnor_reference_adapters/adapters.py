"""Reference third-party adapters for Ragnor ports.

This package is authored the way an external adapter author would write
one: it depends only on Ragnor's public ports and types, registers nothing
globally, and is injected by the host as plain instances. It exists to
prove that every major adapter category is independently implementable and
passes the published conformance suite (``ragnor.conformance``) without
editing Ragnor.

The deterministic implementations are deliberately simple -- they are
conformance evidence, not production components.
"""

from __future__ import annotations

import hashlib

from ragnor.ingestion.records import RawDocument
from ragnor.telemetry.events import TelemetryEvent
from ragnor.types import Chunk

EMBEDDING_DIMENSION = 8


class ReferenceChunker:
    """Splits text on a configurable separator."""

    version = "reference-1.0.0"
    config = {"separator": "\n\n"}

    def chunk(self, text: str) -> list[Chunk]:
        separator = self.config["separator"]
        return [
            Chunk(id=f"reference-{index}", text=piece)
            for index, piece in enumerate(text.split(separator))
            if piece.strip()
        ] or [Chunk(id="reference-0", text=text)]


class ReferenceTokenizer:
    """Whitespace tokenizer; explicitly approximate (never claims exactness)."""

    name = "reference-whitespace"
    is_exact = False

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ReferenceEmbedder:
    """Deterministic hashing embedder with a fixed output dimension."""

    dimension = EMBEDDING_DIMENSION

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            (digest[index % len(digest)] / 127.5) - 1.0 for index in range(self.dimension)
        ]


class ReferenceEventSink:
    """Collects emitted telemetry events for host-side inspection."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class ReferenceDocumentSource:
    """Loads an in-memory document set with stable source identity."""

    def __init__(self, documents: dict[str, str] | None = None) -> None:
        self._documents = documents or {"reference-policy": "refund policy text"}

    async def load(self) -> list[RawDocument]:
        return [
            RawDocument(source_name=name, external_id=name, content=content)
            for name, content in self._documents.items()
        ]
