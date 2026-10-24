"""Independent host adapters type-checked against the *installed* Ragnor wheel.

This is a consumer contract, not part of the SDK source tree. A third-party
package can supply each port without inheriting a framework-specific base.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragnor.chunking.base import Chunker
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.evidence.tokenizer import Tokenizer
from ragnor.jobs.outbox import JobEvent, JobEventSink, JobPayload, JobProcessor
from ragnor.jobs.publication import GenerationVectorReader
from ragnor.stores.base import DocumentStore, LexicalIndex, VectorIndex
from ragnor.core.types import Chunk


class HostTokenizer:
    name = "third-party-exact-counter-v1"
    is_exact = True

    def count_tokens(self, text: str) -> int:
        return len(text)


class HostChunker:
    version = "third-party-chunker-v1"

    @property
    def config(self) -> dict[str, Any]:
        return {"chunk_size": 8, "overlap": 0}

    def chunk(self, text: str) -> list[Chunk]:
        return [Chunk(id="host-chunk", text=text)] if text else []


class HostEmbedder:
    async def __call__(self, input: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in input]

    async def embed_query(self, input: str | list[str]) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, input: list[str]) -> list[list[float]]:
        return await self(input)


class HostSearchStore:
    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return []

    async def keyword_search(
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return []

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def count(self, session: Any = None, strict: bool = False, **filters: Any) -> int:
        return 0

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        return 0

    async def get(
        self,
        ids: list[Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []


class HostProcessor:
    async def process(self, job: JobPayload) -> None:
        return None


class HostEventSink:
    async def emit(self, event: JobEvent) -> None:
        return None


class HostGenerationReader:
    async def read_generation(self, generation_key: str) -> tuple[str, ...]:
        return ()


tokenizer: Tokenizer = HostTokenizer()
chunker: Chunker = HostChunker()
embedder: EmbeddingFunction = HostEmbedder()
store: DocumentStore = HostSearchStore()
vectors: VectorIndex = HostSearchStore()
lexical: LexicalIndex = HostSearchStore()
processor: JobProcessor = HostProcessor()
events: JobEventSink = HostEventSink()
reader: GenerationVectorReader[tuple[str, ...]] = HostGenerationReader()

assert tokenizer.is_exact and chunker.version and embedder and store
assert vectors and lexical and processor and events and reader
