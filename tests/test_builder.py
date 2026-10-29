"""Tests for the RagnorBuilder high-level convenience API.

Covers the builder's offline defaults (in-memory store,
sliding-window chunker, in-memory version store, null session
provider), the facade's ingest/search/count workflow, idempotent
re-ingestion, version replacement through the store's delete
capability, component overrides, and the embedder requirement.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pytest

from ragnor.builder import (
    BuilderSettings,
    InMemoryVersionStore,
    Ragnor,
    RagnorBuilder,
)
from ragnor.chunking.base import Chunker
from ragnor.config import RagSettings
from ragnor.ingestion.pipeline import IngestionPipeline
from ragnor.ingestion.sources import (
    DocumentSource,
    ListDocumentSource,
    RawDocument,
)
from ragnor.retrieval.retriever import HybridRetriever
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.types import Chunk, Document, SearchResult


class _HashingEmbedder:
    """Deterministic embedder: tokens hash into vector dimensions.

    Shared tokens land in shared dimensions, so texts that share
    vocabulary get close L2 distance -- bag-of-words similarity
    without any network calls.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _RecordingEmbedder:
    """Embedder that records every batch call and delegates to another."""

    def __init__(self, embedder: _HashingEmbedder) -> None:
        self._embedder = embedder
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await self._embedder(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append([text])
        return await self._embedder.embed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.embed_documents(texts)


class _MarkerChunker:
    """Chunker that splits on a marker, producing predictable chunks."""

    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(id=f"marker-{index}", text=part) for index, part in enumerate(text.split("||"))
        ]

    @property
    def version(self) -> str:
        return "marker-v1"

    @property
    def config(self) -> dict[str, Any]:
        return {"chunk_size": 1, "overlap": 0}


def test_builder_settings_satisfies_rag_settings_protocol() -> None:
    settings = BuilderSettings()
    assert isinstance(settings, RagSettings)
    assert settings.embedding_model == "local"
    assert settings.rag_distance_threshold == 1.3
    assert settings.vector_store_provider == "memory"


@pytest.mark.asyncio
async def test_list_document_source_maps_documents() -> None:
    source = ListDocumentSource(
        [
            Document(id="doc-1", text="alpha content"),
            RawDocument(source_name="raw", content="beta content", external_id="raw-1"),
        ]
    )
    assert isinstance(source, DocumentSource)

    loaded = await source.load()
    assert loaded == [
        RawDocument(source_name="doc-1", content="alpha content", external_id="doc-1"),
        RawDocument(source_name="raw", content="beta content", external_id="raw-1"),
    ]


@pytest.mark.asyncio
async def test_local_build_ingest_search_round_trip() -> None:
    embedder = _HashingEmbedder(dim=8)
    rag = await RagnorBuilder.local(embedder, embedding_dim=8).build()

    count = await rag.ingest(
        [
            Document(id="policy", text="The refund policy covers water damage."),
            Document(id="hours", text="Office hours are nine to five."),
        ]
    )
    assert count == 2
    assert await rag.count() == 2

    results = await rag.search("refund policy", n_results=5)
    assert results
    assert all(isinstance(result, SearchResult) for result in results)
    assert results[0].document == "The refund policy covers water damage."
    assert results[0].metadata["source"] == "policy"

    # Equality filters reach both search stages.
    filtered = await rag.search("refund", n_results=5, source="policy")
    assert filtered
    assert all(result.metadata["source"] == "policy" for result in filtered)


@pytest.mark.asyncio
async def test_second_identical_ingest_is_noop() -> None:
    embedder = _RecordingEmbedder(_HashingEmbedder(dim=8))
    store = InMemoryVectorStore(embedding_dim=8, embedding_fn=embedder)
    version_store = InMemoryVersionStore(store)
    rag = await RagnorBuilder.local(
        embedder,
        embedding_dim=8,
        store=store,
        version_store=version_store,
    ).build()

    documents = [Document(id="policy", text="The refund policy covers water damage.")]
    first = await rag.ingest(documents)
    first_version = (await version_store.find_active(None)).version_id
    calls_after_first = list(embedder.calls)

    second = await rag.ingest(documents)
    second_version = (await version_store.find_active(None)).version_id

    # The stored chunk count is returned, no new version is created,
    # and the embedder is never called again (snapshot match).
    assert second == first
    assert second_version == first_version
    assert embedder.calls == calls_after_first


@pytest.mark.asyncio
async def test_changed_content_reingests_and_deletes_retired_chunks() -> None:
    embedder = _HashingEmbedder(dim=8)
    store = InMemoryVectorStore(embedding_dim=8, embedding_fn=embedder)
    version_store = InMemoryVersionStore(store)
    rag = await RagnorBuilder.local(
        embedder,
        embedding_dim=8,
        store=store,
        version_store=version_store,
    ).build()

    await rag.ingest([Document(id="policy", text="The original refund policy text.")])
    old_version = (await version_store.find_active(None)).version_id
    old_chunks = await store.get(
        filters={"version_id": old_version}, metadata_fields=["version_id"]
    )
    assert old_chunks

    await rag.ingest(
        [
            Document(
                id="policy",
                text="The revised refund policy text with different content.",
            )
        ]
    )
    new_version = (await version_store.find_active(None)).version_id

    assert new_version != old_version
    # The retired version's chunks are deleted from the store...
    assert await store.get(filters={"version_id": old_version}) == []
    # ...every remaining chunk carries the new version id...
    remaining = await store.get(metadata_fields=["version_id"])
    assert remaining
    assert all(row["metadata"]["version_id"] == new_version for row in remaining)
    # ...and the version store tracked the retired version's chunk ids.
    assert version_store.chunk_ids(old_version) == [row["id"] for row in old_chunks]
    assert version_store.snapshot(new_version) is not None


@pytest.mark.asyncio
async def test_component_overrides_are_honored() -> None:
    embedder = _RecordingEmbedder(_HashingEmbedder(dim=8))
    store = InMemoryVectorStore(table_name="custom_table", embedding_dim=8, embedding_fn=embedder)
    chunker = _MarkerChunker()
    rag = await RagnorBuilder.local(
        embedder,
        embedding_dim=8,
        chunker=chunker,
        store=store,
    ).build()

    # The custom store is the facade's store, and the retriever
    # searches it (a default store would stay empty).
    assert rag.store is store

    await rag.ingest([Document(id="doc", text="alpha||beta||gamma")])

    # The custom chunker's chunks were stored.
    stored = await store.get()
    assert [row["document"] for row in stored] == ["alpha", "beta", "gamma"]

    # The custom embedder embedded the chunks during ingest...
    assert embedder.calls
    embedder.calls.clear()
    # ...and the query during search.
    results = await rag.search("alpha", n_results=5)
    assert [result.document for result in results] == ["alpha"]
    assert embedder.calls


@pytest.mark.asyncio
async def test_with_methods_override_components() -> None:
    embedder = _HashingEmbedder(dim=8)
    store = InMemoryVectorStore(embedding_dim=8, embedding_fn=embedder)
    builder = (
        RagnorBuilder.local(None)
        .with_embedder(embedder)
        .with_store(store)
        .with_chunker(_MarkerChunker())
    )
    rag = await builder.build()

    assert rag.store is store
    await rag.ingest([Document(id="doc", text="one||two")])
    assert await rag.count() == 2


@pytest.mark.asyncio
async def test_build_without_embedder_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no embedder"):
        await RagnorBuilder().build()

    with pytest.raises(ValueError, match="no embedder"):
        await RagnorBuilder.local(None).build()


@pytest.mark.asyncio
async def test_purge_on_empty_removes_all_chunks() -> None:
    embedder = _HashingEmbedder(dim=8)
    rag = await RagnorBuilder.local(embedder, embedding_dim=8).build()

    await rag.ingest(
        [
            Document(id="a", text="alpha document"),
            Document(id="b", text="beta document"),
        ]
    )
    assert await rag.count() == 2

    await rag.ingest([], purge_on_empty=True)
    assert await rag.count() == 0


@pytest.mark.asyncio
async def test_facade_exposes_low_level_seams() -> None:
    rag = await RagnorBuilder.local(_HashingEmbedder(dim=8), embedding_dim=8).build()
    assert isinstance(rag, Ragnor)
    assert isinstance(rag.store, InMemoryVectorStore)
    assert isinstance(rag.pipeline, IngestionPipeline)
    assert isinstance(rag.retriever, HybridRetriever)


def test_chunker_protocol_is_satisfied_by_marker_chunker() -> None:
    assert isinstance(_MarkerChunker(), Chunker)
