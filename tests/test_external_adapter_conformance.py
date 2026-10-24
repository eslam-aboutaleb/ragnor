"""External adapter fixtures that prove protocols are independently implementable."""

from __future__ import annotations

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.chunking.base import Chunker
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.evidence.tokenizer import Tokenizer
from ragnor.ingestion.sources import RawDocument
from ragnor.jobs.outbox import (
    JobPayload,
    JobProcessor,
    InMemoryEmbeddingJobStore,
    process_pending_jobs,
)
from ragnor.registry import Registry
from ragnor.telemetry.events import EventSink, TelemetryEvent
from ragnor.types import Chunk, Document
from tests.conformance.contracts import (
    check_chunker,
    check_embedder,
    check_event_sink,
    check_tokenizer,
)


class HostChunker:
    version = "host-v1"
    config = {"separator": "|"}

    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(id=f"custom-{index}", text=part)
            for index, part in enumerate(text.split("|"))
            if part
        ]


class HostTokenizer:
    name = "host-test-tokenizer"
    is_exact = False

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class HostEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "refund" in text else 0.0, 1.0] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class HostEvents:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class HostDocumentSource:
    async def load(self) -> list[RawDocument]:
        return [
            RawDocument(
                source_name="tenant-document",
                external_id="source-001",
                content="refund|customer policy",
            )
        ]


class HostJobProcessor:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    async def process(self, job: JobPayload) -> None:
        self.job_ids.append(job.job_id)


@pytest.mark.asyncio
async def test_host_adapters_satisfy_core_ports_and_process_documents() -> None:
    embedder = HostEmbedder()
    chunker = HostChunker()
    tokenizer = HostTokenizer()
    events = HostEvents()

    assert isinstance(embedder, EmbeddingFunction)
    assert isinstance(chunker, Chunker)
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(events, EventSink)
    check_chunker(chunker)
    check_tokenizer(tokenizer)
    await check_embedder(embedder, dimension=2)
    check_event_sink(events)
    assert events.events

    # The host extension does not modify the core package or global registries.
    rag = await RagnorBuilder.local(embedder, embedding_dim=2).with_chunker(chunker).build()
    chunks = await rag.ingest([Document(id="custom", text="refund|customer policy")])
    assert chunks == 2
    assert await rag.count() == 2

    # Existing ingestion supports a separate host source loader port.
    ingested = await rag.pipeline.run(HostDocumentSource(), rag._version_store)
    assert ingested == 2


def test_host_plugin_registry_is_instance_scoped() -> None:
    first: Registry[Chunker] = Registry()
    second: Registry[Chunker] = Registry()
    first.register("My-Host")(HostChunker)
    assert first.get("my-host") is HostChunker
    assert second.list() == []


@pytest.mark.asyncio
async def test_host_job_processor_plugs_into_outbox_protocol() -> None:
    store = InMemoryEmbeddingJobStore()
    processor = HostJobProcessor()
    assert isinstance(processor, JobProcessor)
    await store.enqueue({"idempotency_key": "host-job-1", "text": "refund terms"})
    processed = await process_pending_jobs(store, processor, worker_id="worker-1")
    assert processed == 1
    assert processor.job_ids == ["host-job-1"]
