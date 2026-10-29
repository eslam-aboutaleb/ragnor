"""Minimal plug-and-play example: a FastAPI app on ragnor.

This example uses ragnor with **zero host-application code**: an
in-memory vector store (no database), a sliding-window
chunker, a deterministic fake embedder (used unless
``OPENAI_API_KEY`` is set in the process environment),
and the ingestion pipeline over a bundled document.

Install and run::

    pip install -e libs/ragnor fastapi uvicorn
    python libs/ragnor/examples/minimal_app.py

Then query the search endpoint::

    curl "http://127.0.0.1:8000/search?query=refund"

No ``DATABASE_URL`` is required: the in-memory store and
the fake embedder need no external services.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
"""API key captured before the ragnor imports: importing ragnor
imports litellm, which loads ``.env`` into ``os.environ`` at
import time, so the key must be read first to reflect the real
process environment."""

from ragnor.chunking import SlidingWindowChunker  # noqa: E402
from ragnor.embeddings import (  # noqa: E402
    FALLBACK_DIMENSION,
    EmbeddingFunction,
    LitellmEmbeddingFunction,
)
from ragnor.ingestion import (  # noqa: E402
    FileDocumentSource,
    IngestionPipeline,
    IngestionSnapshot,
    VersionRecord,
)
from ragnor.stores import InMemoryVectorStore  # noqa: E402


class ExampleSettings:
    """Minimal ``RagSettings`` for the example.

    The example runs without a database, so the database
    settings are empty; the in-memory store never reads
    them.
    """

    embedding_model: str = "example-fake-embedder"
    rag_distance_threshold: float = 1.3
    vector_store_provider: str = "memory"
    openai_api_key: str = _OPENAI_API_KEY
    openai_api_base: str = os.environ.get("OPENAI_API_BASE", "")
    embedding_drain_interval_seconds: float = 5.0
    embedding_drain_batch_size: int = 10
    job_stale_after_seconds: float = 300.0
    worker_id: str = "example-worker"
    ingest_on_startup: bool = False


def _fake_embed(text: str) -> list[float]:
    """Deterministic hash embedding (no API key required)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector: list[float] = []
    while len(vector) < FALLBACK_DIMENSION:
        digest = hashlib.sha256(digest).digest()
        vector.extend(byte / 255.0 for byte in digest)
    return vector[:FALLBACK_DIMENSION]


class FakeEmbedder:
    """Deterministic embedder used when no API key is configured."""

    async def __call__(self, input: list[str]) -> list[list[float]]:
        return [_fake_embed(text) for text in input]

    async def embed_query(self, input: str | list[str]) -> list[float]:
        return _fake_embed(input if isinstance(input, str) else input[0])

    async def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [_fake_embed(text) for text in input]


class _NullSession:
    """No-op session: the in-memory example commits nothing."""

    async def commit(self) -> None:
        """No-op."""


@asynccontextmanager
async def _null_session_factory():
    """Yield a no-op session (the example runs without a database)."""
    yield _NullSession()


class MemoryVersionStore:
    """In-memory ``VersionStore`` (the example keeps no version table)."""

    def __init__(self) -> None:
        self.active: VersionRecord | None = None

    async def find_active(self, session: Any) -> VersionRecord | None:
        return self.active

    async def close_active(self, session: Any) -> None:
        self.active = None

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        self.active = VersionRecord(
            version_id=f"example-{uuid.uuid4().hex[:8]}",
            snapshot=snapshot,
            source_id="example-document",
        )
        return self.active

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        """No-op: the in-memory store keeps no per-version chunk rows."""


settings = ExampleSettings()
store = InMemoryVectorStore()
embedder: EmbeddingFunction = (
    LitellmEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    if settings.openai_api_key
    else FakeEmbedder()
)
chunker = SlidingWindowChunker()
version_store = MemoryVersionStore()
pipeline = IngestionPipeline(
    store=store,
    embedder=embedder,
    chunker=chunker,
    session_factory=_null_session_factory,
    settings=settings,
    version_id_key="version_id",
    source_id_key="source_id",
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Ingest the bundled document on startup."""
    document = Path(__file__).with_name("sample_document.md")
    count = await pipeline.run(FileDocumentSource(str(document)), version_store)
    print(f"ingested {count} chunks from {document.name}")
    yield


app = FastAPI(title="ragnor minimal example", lifespan=_lifespan)


@app.get("/search")
async def search(query: str, n_results: int = 5) -> dict[str, Any]:
    """Hybrid-search the ingested document (vector + keyword, RRF-fused)."""
    embedding = await embedder.embed_query(query)
    results = await store.hybrid_search(
        query=query,
        embedding=embedding,
        n_results=n_results,
        threshold=settings.rag_distance_threshold,
        text_field="text",
        metadata_fields=["section", "source"],
    )
    return {"query": query, "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
