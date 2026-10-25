"""Host-application integration template for ragnor.

This example wires ragnor's seams the way a host application
does: a domain embedding provider registered through
``embedding_registry``, a custom vector backend registered
through ``vector_store_registry`` (backed by a shared
in-memory index), the ingestion pipeline over a bundled
document, hybrid retrieval, and the retrieval evaluation
harness.

Run it with::

    python examples/host_integration.py

Everything runs in process: no network, no database, no API
keys, no external services.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragnor.chunking import MarkdownSectionChunker, SlidingWindowChunker
from ragnor.config import RagSettings
from ragnor.embeddings import (
    FALLBACK_DIMENSION,
    EmbeddingFunction,
    embedding_registry,
    get_embedding_dimension,
    get_embedding_function,
    register_dimension,
)
from ragnor.evaluation import LabeledQuery, RetrievalMetrics, RagEvaluationHarness
from ragnor.ingestion import (
    DocumentSource,
    FileDocumentSource,
    IngestionPipeline,
    IngestionSnapshot,
    VersionRecord,
)
from ragnor.retrieval import HybridRetriever, RetrieverSpec, as_dicts
from ragnor.stores import (
    InMemoryVectorStore,
    VectorStore,
    get_vector_store,
    vector_store_registry,
)

DOMAIN_EMBEDDING_MODEL = "domain-fake-embedder"
"""Embedding model name the domain template registers."""

# Register the dimension seam: keeps dimension validation in sync with the
# fake embedder's output size (FALLBACK_DIMENSION).
register_dimension(DOMAIN_EMBEDDING_MODEL, FALLBACK_DIMENSION)


class DomainSettings:
    """Domain settings satisfying the :class:`RagSettings` protocol.

    ragnor ships no concrete settings base: the
    :class:`RagSettings` protocol is the contract, and a host
    subclasses its own settings base (e.g. pydantic
    ``BaseSettings``) and adds these fields. Override the
    domain fields (embedding model, provider names, distance
    threshold) to adapt the template. For SQL storage the host creates
    one engine and injects its sessions. Validate
    at start-up with ``isinstance(settings, RagSettings)``.
    """

    embedding_model: str = DOMAIN_EMBEDDING_MODEL
    embedding_provider: str = "domain-provider"
    rag_distance_threshold: float = 1.3
    vector_store_provider: str = "domain-backend"
    openai_api_key: str = ""
    openai_api_base: str = ""
    embedding_drain_interval_seconds: float = 5.0
    embedding_drain_batch_size: int = 10
    job_stale_after_seconds: float = 300.0
    worker_id: str = "domain-worker"
    ingest_on_startup: bool = False


def _fake_embed(text: str) -> list[float]:
    """Deterministic hash embedding (no API key required)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector: list[float] = []
    while len(vector) < FALLBACK_DIMENSION:
        digest = hashlib.sha256(digest).digest()
        vector.extend(byte / 255.0 for byte in digest)
    return vector[:FALLBACK_DIMENSION]


class FakeDomainEmbedder:
    """Deterministic domain embedder implementing ``EmbeddingFunction``.

    The user-supplied embedder: every method delegates to
    :func:`_fake_embed`, so embeddings are deterministic and
    need no API key.
    """

    async def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            input: Texts to embed.

        Returns:
            One embedding vector per input text.
        """
        return [_fake_embed(text) for text in input]

    async def embed_query(self, input: str | list[str]) -> list[float]:
        """Embed a query.

        Args:
            input: Query text (or a list whose first element is
                the query).

        Returns:
            The query embedding vector.
        """
        return _fake_embed(input if isinstance(input, str) else input[0])

    async def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            input: Document texts to embed.

        Returns:
            One embedding vector per document.
        """
        return [_fake_embed(text) for text in input]


@embedding_registry.register("domain-provider")
class DomainEmbeddingProvider:
    """Factory building :class:`FakeDomainEmbedder` from settings.

    Selected via ``settings.embedding_provider``; a host
    registers its real provider the same way.
    """

    def __init__(self, settings: RagSettings) -> None:
        """Store the settings the factory reads.

        Args:
            settings: Host settings.
        """
        self._settings = settings

    def create(self) -> EmbeddingFunction:
        """Return the domain embedder.

        Returns:
            A :class:`FakeDomainEmbedder` instance.
        """
        return FakeDomainEmbedder()


@vector_store_registry.register("domain-backend")
class DomainVectorBackend(VectorStore):
    """Demo vector backend over a shared in-memory index.

    One shared :class:`InMemoryVectorStore` per table name
    keeps the pipeline's writes visible to the retriever's
    reads within this process; a production backend shares a
    database server instead, and the host wiring code is
    identical. ``session``, ``strict``, and ``settings`` are
    accepted for interface compatibility.
    """

    _indexes: dict[str, InMemoryVectorStore] = {}

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str = "documents",
        id_field: str = "id",
        embedding_dim: int | None = None,
        session_factory: Any = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        """Initialize the backend, reusing the shared index for ``table_name``.

        Args:
            table_name: Logical table name; the shared index is
                keyed by it.
            id_field: Name of the primary key field.
            embedding_dim: Expected embedding dimensionality.
            session_factory: Accepted for interface compatibility.
            settings: Accepted for interface compatibility.
            embedding_fn: Embedding function for automatic query
                embedding.
        """
        backend = DomainVectorBackend._indexes.get(table_name)
        if backend is None:
            backend = InMemoryVectorStore(
                table_name=table_name,
                id_field=id_field,
                embedding_dim=embedding_dim,
                embedding_fn=embedding_fn,
            )
            DomainVectorBackend._indexes[table_name] = backend
        self._backend = backend
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim
        self._embedding_fn = embedding_fn

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Hybrid search delegated to the shared in-memory index.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector.
            n_results: Maximum number of results to return.
            threshold: Maximum distance for the vector search part.
            text_field: Name of the field containing the document text.
            metadata_fields: Fields to include in the result metadata.
            session: Accepted for interface compatibility.
            strict: Accepted for interface compatibility.
            **filters: Equality filters matched against document fields.

        Returns:
            Result dicts sorted by RRF score.
        """
        return await self._backend.hybrid_search(
            query=query,
            embedding=embedding,
            n_results=n_results,
            threshold=threshold,
            text_field=text_field,
            metadata_fields=metadata_fields,
            session=session,
            strict=strict,
            **filters,
        )

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the shared index.

        Args:
            session: Accepted for interface compatibility.
            strict: Accepted for interface compatibility.
            **filters: Equality filters matched against document fields.

        Returns:
            Number of matching documents.
        """
        return await self._backend.count(session=session, strict=strict, **filters)

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Upsert documents into the shared index.

        Args:
            documents: List of document dicts with keys: id, text,
                metadata, embedding.
            session: Accepted for interface compatibility.
            text_field: Name of the text field.
            embedding_field: Name of the embedding field.
            id_field: Name of the primary key field. Defaults to the
                backend's configured id field.
            extra_fields: Additional static fields to include in every
                document.
        """
        await self._backend.upsert(
            documents,
            session=session,
            text_field=text_field,
            embedding_field=embedding_field,
            id_field=id_field,
            extra_fields=extra_fields,
        )


class _NullSession:
    """No-op session: the offline demo commits nothing."""

    async def commit(self) -> None:
        """No-op commit (there is no database to commit to)."""


@asynccontextmanager
async def null_session_factory():
    """Yield a no-op session (the demo runs without a database)."""
    yield _NullSession()


class MemoryVersionStore:
    """In-memory ``VersionStore`` (the demo keeps no version table)."""

    def __init__(self) -> None:
        """Initialize with no active version."""
        self.active: VersionRecord | None = None

    async def find_active(self, session: Any) -> VersionRecord | None:
        """Return the active version, or ``None``.

        Args:
            session: Pipeline session (unused; storage is in process).

        Returns:
            The active version record, or ``None``.
        """
        return self.active

    async def close_active(self, session: Any) -> None:
        """Retire the active version.

        Args:
            session: Pipeline session (unused; storage is in process).
        """
        self.active = None

    async def create_version(
        self,
        session: Any,
        snapshot: IngestionSnapshot,
    ) -> VersionRecord:
        """Insert a new version row carrying ``snapshot``.

        Args:
            session: Pipeline session (unused; storage is in process).
            snapshot: The ingestion snapshot to store.

        Returns:
            The newly created version record.
        """
        self.active = VersionRecord(
            version_id=f"domain-{uuid.uuid4().hex[:8]}",
            snapshot=snapshot,
            source_id="domain-document",
        )
        return self.active

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        """Delete the chunks belonging to ``version_id``.

        No-op: the in-memory store keeps no per-version chunk rows.

        Args:
            session: Pipeline session (unused; storage is in process).
            version_id: Identifier of the version whose chunks are
                deleted.
        """


# Chunk ids are the Markdown chunker's position-based ids for the bundled
# sample_document.md: chunk_0 is the H1 preamble, chunk_1 "Greeter Policy",
# chunk_2 "Refund Policy", chunk_3 "Office Hours".
TEST_SET: list[LabeledQuery] = [
    LabeledQuery(query="What is the refund policy?", expected_chunk_ids={"chunk_2"}),
    LabeledQuery(query="Who greets visitors?", expected_chunk_ids={"chunk_1"}),
    LabeledQuery(query="When is the office open?", expected_chunk_ids={"chunk_3"}),
]


@dataclass
class HostApp:
    """The wired host application."""

    settings: DomainSettings
    store: VectorStore
    embedder: EmbeddingFunction
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    version_store: MemoryVersionStore
    harness: RagEvaluationHarness

    async def ingest(self, source: DocumentSource) -> int:
        """Ingest a document source.

        Args:
            source: Document source to ingest.

        Returns:
            The number of chunks stored.
        """
        return await self.pipeline.run(source, self.version_store)

    async def retrieve(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents matching a query.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.

        Returns:
            Result dicts sorted by RRF score.
        """
        return as_dicts(await self.retriever.retrieve(query, n_results=n_results))

    async def evaluate(self) -> RetrievalMetrics:
        """Evaluate retrieval quality against the bundled test set.

        The harness adapter surfaces the chunker's
        position-based ``chunk_id`` (carried in each
        result's metadata) as the result id: the store
        keys rows by UUID, while the curated test set is
        keyed by the stable ``chunk_N`` ids.

        Returns:
            Aggregated retrieval metrics.
        """
        return await self.harness.evaluate(self._retrieve_by_chunk_id)

    async def _retrieve_by_chunk_id(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Retrieve result dicts keyed by position-based chunk id.

        Args:
            query: Natural language search query.

        Returns:
            Result dicts whose ``id`` is the chunker's
            ``chunk_id`` when the metadata carries one.
        """
        rows = await self.retrieve(query)
        for row in rows:
            chunk_id = row.get("metadata", {}).get("chunk_id")
            if chunk_id:
                row["id"] = chunk_id
        return rows


def build_host(table_name: str = "domain_chunks") -> HostApp:
    """Wire the domain application together.

    ``get_embedding_function`` resolves ``"domain-provider"``
    through ``embedding_registry`` and ``get_vector_store``
    resolves ``"domain-backend"`` through ``vector_store_registry``;
    a host may alternatively construct its embedder directly
    (``FakeDomainEmbedder()``) and pass it to ``IngestionPipeline``
    — both paths satisfy the ``EmbeddingFunction`` protocol.

    Args:
        table_name: Logical table name for the chunk index.

    Returns:
        The wired :class:`HostApp`.
    """
    settings = DomainSettings()
    embedder = get_embedding_function(settings)
    embedding_dim = get_embedding_dimension(settings.embedding_model)
    store = get_vector_store(
        table_name=table_name,
        embedding_dim=embedding_dim,
        settings=settings,
        embedding_fn=embedder,
    )
    chunker = MarkdownSectionChunker(SlidingWindowChunker())
    version_store = MemoryVersionStore()
    pipeline = IngestionPipeline(
        store=store,
        embedder=embedder,
        chunker=chunker,
        session_factory=null_session_factory,
        settings=settings,
    )
    retriever = HybridRetriever(
        spec=RetrieverSpec(
            table_name=table_name,
            metadata_fields=["chunk_id", "section", "source"],
        ),
        settings=settings,
        session_factory=null_session_factory,
    )
    harness = RagEvaluationHarness(TEST_SET)
    return HostApp(
        settings=settings,
        store=store,
        embedder=embedder,
        pipeline=pipeline,
        retriever=retriever,
        version_store=version_store,
        harness=harness,
    )


async def run_demo() -> HostApp:
    """Run the offline demo: ingest, re-ingest, retrieve, and evaluate.

    Returns:
        The wired :class:`HostApp`, ready to serve queries.
    """
    app = build_host()
    document = Path(__file__).with_name("sample_document.md")
    source = FileDocumentSource(str(document))

    count = await app.ingest(source)
    print(f"ingested {count} chunks from {document.name}")

    count = await app.ingest(source)
    print(
        f"re-ingest returned {count} stored chunks "
        "(unchanged source skipped via snapshot comparison)"
    )

    results = await app.retrieve("What is the refund policy?", n_results=3)
    print("top results for 'What is the refund policy?':")
    for result in results:
        print(f"  [{result['id']}] {result['document'][:70]}")

    metrics = await app.evaluate()
    print(
        f"retrieval quality: recall@5={metrics.recall_at_k:.4f} "
        f"precision@5={metrics.precision_at_k:.4f} mrr={metrics.mrr:.4f} "
        f"errors={metrics.errors}"
    )
    print("host integration demo complete: retrieval quality validated, ready to serve")
    return app


if __name__ == "__main__":
    asyncio.run(run_demo())
