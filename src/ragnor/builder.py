"""High-level convenience API: the RagnorBuilder and the Ragnor facade.

The builder sits *on top* of the low-level seams -- it wires defaults
into :class:`ragnor.ingestion.pipeline.IngestionPipeline` and
:class:`ragnor.retrieval.retriever.HybridRetriever` without forking
their logic, so a host goes from zero to a working index in three
lines::

    rag = await RagnorBuilder.local(embedder=my_embedder).build()
    await rag.ingest([Document(id="doc", text="...")])
    results = await rag.search("query")

Every component is overridable (``with_store`` / ``with_embedder`` /
``with_chunker`` / ``with_version_store`` / ``with_settings``), and
the low-level seams stay reachable through the facade's ``.store``,
``.pipeline``, and ``.retriever`` attributes.

ragnor ships no local embedder in core (the ``local-embeddings`` extra
is reserved for a future local-model backend), so the builder requires
an injected embedder -- the offline example
(``examples/local_quickstart.py``) defines its own deterministic fake.
The default wiring is fully offline: the in-memory vector store, the
in-memory version store, and a null session provider need no database,
no driver, and no API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ragnor.chunking import SlidingWindowChunker
from ragnor.chunking.base import Chunker
from ragnor.config import RagSettings
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.dimensions import register_dimension
from ragnor.ingestion.pipeline import (
    IngestionPipeline,
    VersionRecord,
    VersionStore,
)
from ragnor.ingestion.sources import ListDocumentSource
from ragnor.ingestion.versioning import IngestionSnapshot
from ragnor.retrieval.retriever import HybridRetriever, RetrieverSpec
from ragnor.retrieval.strategies import HybridRetrieval, RetrievalStrategy
from ragnor.stores.base import DocumentStore, VectorStore
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.telemetry.events import EventSink
from ragnor.types import Document, SearchResult

if TYPE_CHECKING:
    from ragnor.db.session import SessionProvider
    from ragnor.db.sql_factory import SQLFactory


DEFAULT_EMBEDDING_MODEL = "local"
"""Embedding model name the builder defaults to."""

DEFAULT_CHUNK_SIZE = 600
"""Default sliding-window chunk size (words per chunk)."""

DEFAULT_OVERLAP = 100
"""Default sliding-window overlap (words shared between chunks)."""


@dataclass
class BuilderSettings:
    """Offline default settings satisfying the :class:`RagSettings` protocol.

    A plain dataclass (no pydantic-settings -- ragnor's core has zero
    third-party dependencies): any object exposing the protocol's
    attributes works structurally, and a host passes its own settings
    through ``RagnorBuilder.local(..., settings=...)`` or
    ``with_settings``.
    """

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rag_distance_threshold: float = 1.3
    vector_store_provider: str = "memory"
    openai_api_key: str = ""
    openai_api_base: str = ""
    embedding_drain_interval_seconds: float = 5.0
    embedding_drain_batch_size: int = 10
    job_stale_after_seconds: float = 300.0
    worker_id: str = "ragnor-builder"
    ingest_on_startup: bool = False
    database_url: str = ""


class InMemoryVersionStore:
    """In-memory :class:`VersionStore` -- the first built-in implementation.

    Tracks the active version, its snapshot, and the chunk ids of
    retired versions in process memory. ``delete_chunks`` deletes a
    retired version's chunks through the wired store's delete
    capability (``await store.delete(filters={version_id_key:
    version_id})``): the pipeline stamps ``version_id`` into chunk
    metadata, so the metadata-filtered delete removes exactly the
    retired version's chunks. The session argument is ignored -- the
    memory store and this version store share no database, so the
    pipeline's owned-session path runs on the null session provider.
    """

    def __init__(
        self,
        store: DocumentStore,
        version_id_key: str = "version_id",
    ) -> None:
        """Initialize the version store wired to ``store``.

        Args:
            store: Store whose delete capability retires chunks.
            version_id_key: Metadata key the pipeline stamps the
                version id under.
        """
        self._store = store
        self._version_id_key = version_id_key
        self._active: VersionRecord | None = None
        self._snapshots: dict[str, IngestionSnapshot] = {}
        self._chunk_ids: dict[str, list[str]] = {}
        self._next_version = 0

    @property
    def active(self) -> VersionRecord | None:
        """The active version, or ``None`` when nothing is ingested."""
        return self._active

    async def find_active(self, session: Any) -> VersionRecord | None:
        """Return the active version (the session is ignored).

        Args:
            session: Pipeline session (unused; storage is in process).

        Returns:
            The active version record, or ``None``.
        """
        return self._active

    async def close_active(self, session: Any) -> None:
        """Retire the active version (the session is ignored).

        Args:
            session: Pipeline session (unused; storage is in process).
        """
        self._active = None

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
        self._next_version += 1
        version_id = f"v{self._next_version}"
        record = VersionRecord(version_id=version_id, snapshot=snapshot)
        self._snapshots[version_id] = snapshot
        self._active = record
        return record

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        """Delete the chunks belonging to ``version_id``.

        The retired version's chunk ids are recorded (read through the
        store's get capability) and the chunks are then deleted through
        the store's delete capability, filtered by the version id the
        pipeline stamps into chunk metadata.

        Args:
            session: Pipeline session (unused; storage is in process).
            version_id: Identifier of the version whose chunks are
                deleted.
        """
        rows = await self._store.get(filters={self._version_id_key: version_id})
        self._chunk_ids[version_id] = [row["id"] for row in rows]
        await self._store.delete(filters={self._version_id_key: version_id})

    def snapshot(self, version_id: str) -> IngestionSnapshot | None:
        """Return the snapshot recorded for ``version_id``.

        Args:
            version_id: Identifier of the version.

        Returns:
            The stored snapshot, or ``None`` when unknown.
        """
        return self._snapshots.get(version_id)

    def chunk_ids(self, version_id: str) -> list[str]:
        """Return the chunk ids recorded for a retired ``version_id``.

        Chunk ids are recorded when a version is retired (its chunks
        are deleted); the active version's chunks are read from the
        store directly.

        Args:
            version_id: Identifier of the retired version.

        Returns:
            The chunk ids that belonged to the version.
        """
        return list(self._chunk_ids.get(version_id, []))


class _NullSession:
    """No-op session: the offline path commits nothing."""

    async def commit(self) -> None:
        """No-op commit (there is no database to commit to)."""


@asynccontextmanager
async def _null_session_factory() -> AsyncIterator[_NullSession]:
    """Yield a no-op session (the builder runs without a database).

    The pipeline's owned-session path calls ``commit()`` on the session
    it opens, so the null session is an object with a no-op commit
    rather than ``None`` itself; the memory store and the in-memory
    version store ignore the session entirely.
    """
    yield _NullSession()


class RagnorBuilder:
    """Builder for the :class:`Ragnor` convenience facade.

    ``RagnorBuilder.local(embedder)`` wires offline defaults -- the
    in-memory vector store, the sliding-window chunker, the in-memory
    version store, and :class:`BuilderSettings` -- and every component
    is overridable before ``build()``.
    """

    def __init__(  # noqa: PLR0913
        self,
        embedder: EmbeddingFunction | None = None,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
        version_store: VersionStore | None = None,
        settings: RagSettings | None = None,
        session_factory: SessionProvider | None = None,
        sql_factory: SQLFactory | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        event_sink: EventSink | None = None,
        production_mode: bool = False,
    ) -> None:
        """Initialize the builder.

        Args:
            embedder: Embedding function for chunk texts and queries.
                ragnor ships no local embedder in core, so this is
                required (``build()`` raises ``ValueError`` without
                one).
            embedding_model: Embedding model name recorded in
                ingestion snapshots.
            embedding_dim: Embedding dimensionality. When given, it is
                registered for ``embedding_model`` via
                :func:`ragnor.embeddings.register_dimension` and the
                default store validates embeddings against it.
            chunker: Chunker that splits document content. Defaults to
                ``SlidingWindowChunker(chunk_size=600, overlap=100)``.
            store: Vector store the chunks are upserted into. Defaults
                to :class:`ragnor.stores.InMemoryVectorStore`.
            version_store: Version storage. Defaults to
                :class:`InMemoryVersionStore` wired to the store.
            settings: Settings for the pipeline and retriever.
                Defaults to :class:`BuilderSettings`.
            session_factory: Host-supplied async session context manager factory.
                If omitted, the builder's local path uses a null session.
            sql_factory: Optional advisory locking adapter for ingestion.
            retrieval_strategy: Optional injected strategy replacing default hybrid retrieval.
            event_sink: Optional host-owned search telemetry sink.
            production_mode: Require explicit persistent components (not a durability proof).
        """
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._chunker = chunker
        self._store = store
        self._version_store = version_store
        self._settings = settings
        self._session_factory = session_factory
        self._sql_factory = sql_factory
        self._retrieval_strategy = retrieval_strategy
        self._event_sink = event_sink
        self._production_mode = production_mode

    @classmethod
    def local(  # noqa: PLR0913
        cls,
        embedder: EmbeddingFunction | None,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int | None = None,
        chunker: Chunker | None = None,
        store: VectorStore | None = None,
        version_store: VersionStore | None = None,
        settings: RagSettings | None = None,
    ) -> RagnorBuilder:
        """Build a builder with offline defaults.

        Args:
            embedder: Embedding function for chunk texts and queries
                (required -- ragnor ships no local embedder in core).
            embedding_model: Embedding model name recorded in
                ingestion snapshots.
            embedding_dim: Embedding dimensionality; registered for
                ``embedding_model`` when given.
            chunker: Chunker override.
            store: Vector store override.
            version_store: Version storage override.
            settings: Settings override.

        Returns:
            A :class:`RagnorBuilder` ready for further customization
            and ``build()``.
        """
        return cls(
            embedder,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunker=chunker,
            store=store,
            version_store=version_store,
            settings=settings,
        )

    def with_embedder(self, embedder: EmbeddingFunction) -> RagnorBuilder:
        """Override the embedder.

        Args:
            embedder: Embedding function for chunk texts and queries.

        Returns:
            This builder (for chaining).
        """
        self._embedder = embedder
        return self

    def with_chunker(self, chunker: Chunker) -> RagnorBuilder:
        """Override the chunker.

        Args:
            chunker: Chunker that splits document content.

        Returns:
            This builder (for chaining).
        """
        self._chunker = chunker
        return self

    def with_store(self, store: VectorStore) -> RagnorBuilder:
        """Override the vector store.

        Args:
            store: Vector store the chunks are upserted into.

        Returns:
            This builder (for chaining).
        """
        self._store = store
        return self

    def with_version_store(self, version_store: VersionStore) -> RagnorBuilder:
        """Override the version storage.

        Args:
            version_store: Version storage.

        Returns:
            This builder (for chaining).
        """
        self._version_store = version_store
        return self

    def with_settings(self, settings: RagSettings) -> RagnorBuilder:
        """Override the settings.

        Args:
            settings: Settings for the pipeline and retriever.

        Returns:
            This builder (for chaining).
        """
        self._settings = settings
        return self

    def with_session_factory(self, session_factory: SessionProvider) -> RagnorBuilder:
        """Use a host-provided session provider for ingestion and retrieval.

        Sessions returned by this factory are controlled by the provider.
        The ingestion pipeline commits its own session, but not a
        session explicitly supplied to `pipeline.run(..., session=...)`.
        """
        self._session_factory = session_factory
        return self

    def with_sql_factory(self, sql_factory: SQLFactory) -> RagnorBuilder:
        """Inject the host's advisory-lock adapter; no SQLAlchemy dependency required."""
        self._sql_factory = sql_factory
        return self

    def with_retrieval_strategy(self, strategy: RetrievalStrategy) -> RagnorBuilder:
        """Use a custom pluggable retrieval strategy instead of the default hybrid one."""
        self._retrieval_strategy = strategy
        return self

    def with_event_sink(self, event_sink: EventSink) -> RagnorBuilder:
        """Forward retrieval telemetry to a host-supplied event sink."""
        self._event_sink = event_sink
        return self

    def with_production_mode(self) -> RagnorBuilder:
        """Refuse known ephemeral defaults when building a persistent application.

        This checks required injection, not whether a custom backend is truly
        durable or safe under concurrency. Validate custom adapters separately.
        """
        self._production_mode = True
        return self

    def _validate_production_dependencies(self) -> None:
        if not self._production_mode:
            return
        required = {
            "store": self._store,
            "version_store": self._version_store,
            "session_factory": self._session_factory,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "production mode requires explicit "
                + ", ".join(missing)
                + "; local defaults are not durable"
            )
        if isinstance(self._store, InMemoryVectorStore) or isinstance(
            self._version_store, InMemoryVersionStore
        ):
            raise ValueError(
                "production mode rejects InMemory stores; "
                "inject and validate durable host implementations"
            )

    async def build(self) -> Ragnor:
        """Wire the components into a :class:`Ragnor` facade.

        The defaults are resolved here (not at ``local()`` time) so a
        ``with_store`` override is also the store the default version
        store is wired to. The retriever is wired through an injected
        :class:`ragnor.retrieval.strategies.HybridRetrieval` strategy
        owning the same store instance the pipeline writes to -- the
        retriever's settings-driven construction would build a second,
        disconnected store and a LiteLLM embedder, which the
        zero-dependency core install cannot do.

        Returns:
            The wired :class:`Ragnor` facade.

        Raises:
            ValueError: If no embedder is configured.
        """
        if self._embedder is None:
            raise ValueError(
                "no embedder configured: pass embedder= to "
                "RagnorBuilder.local() or call with_embedder() "
                "(ragnor ships no local embedder in core)"
            )

        self._validate_production_dependencies()

        settings = self._settings
        if settings is None:
            settings = BuilderSettings(embedding_model=self._embedding_model)

        if self._embedding_dim is not None:
            register_dimension(settings.embedding_model, self._embedding_dim)

        store = self._store
        if store is None:
            store = InMemoryVectorStore(
                embedding_dim=self._embedding_dim,
                embedding_fn=self._embedder,
            )

        chunker = self._chunker
        if chunker is None:
            chunker = SlidingWindowChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)

        version_store = self._version_store
        if version_store is None:
            version_store = InMemoryVersionStore(store)

        session_factory = self._session_factory or _null_session_factory
        pipeline = IngestionPipeline(
            store=store,
            embedder=self._embedder,
            chunker=chunker,
            session_factory=session_factory,
            settings=settings,
            sql_factory=self._sql_factory,
        )
        strategy = self._retrieval_strategy
        if strategy is None:
            strategy = HybridRetrieval(
                vector_index=store,
                lexical_index=store,
                embedder=self._embedder,
            )
        retriever = HybridRetriever(
            spec=RetrieverSpec(
                table_name=getattr(store, "table_name", "documents"),
                metadata_fields=["source", "version_id"],
            ),
            settings=settings,
            session_factory=session_factory,
            strategy=strategy,
            event_sink=self._event_sink,
        )
        return Ragnor(
            store=store,
            pipeline=pipeline,
            retriever=retriever,
            version_store=version_store,
            settings=settings,
        )


class Ragnor:
    """High-level facade over the ragnor seams.

    Exposes the three-line workflow (``ingest`` / ``search`` /
    ``count``) while keeping the low-level seams reachable through
    ``.store``, ``.pipeline``, and ``.retriever``.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        store: VectorStore,
        pipeline: IngestionPipeline,
        retriever: HybridRetriever,
        version_store: VersionStore,
        settings: RagSettings,
    ) -> None:
        """Initialize the facade.

        Args:
            store: Vector store the chunks are upserted into.
            pipeline: Ingestion pipeline the facade runs.
            retriever: Hybrid retriever the facade searches through.
            version_store: Version storage the pipeline records
                snapshots in.
            settings: Settings the pipeline and retriever read.
        """
        self.store = store
        self.pipeline = pipeline
        self.retriever = retriever
        self._version_store = version_store
        self._settings = settings

    async def ingest(
        self,
        documents: list[Document],
        *,
        purge_on_empty: bool = False,
        lock_key: str | None = None,
    ) -> int:
        """Ingest documents and return the stored chunk count.

        The documents are wrapped in a
        :class:`ragnor.ingestion.sources.ListDocumentSource` and run
        through the ingestion pipeline: an unchanged document set is
        skipped via snapshot comparison (idempotent), and changed
        content replaces the active version -- the retired version's
        chunks are deleted from the store.

        Args:
            documents: Documents to ingest.
            purge_on_empty: Treat an empty list as authoritative: the
                active version is retired and its chunks deleted from
                the search index.
            lock_key: Optional scoped ingestion advisory lock identifier.
                Requires a compatible SQL factory and transaction/session
                provider. Never a distributed lock with local defaults.

        Returns:
            The number of chunks stored, or the stored chunk count
            when the documents were unchanged.
        """
        source = ListDocumentSource(documents)
        return await self.pipeline.run(
            source,
            self._version_store,
            purge_on_empty=purge_on_empty,
            lock_key=lock_key,
        )

    async def search(
        self,
        query: str,
        n_results: int = 5,
        **filters: Any,
    ) -> list[SearchResult]:
        """Search for documents matching ``query``.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            **filters: Equality filters applied to both search stages
                (e.g. ``source="policy"``).

        Returns:
            Search results sorted by RRF score.
        """
        return await self.retriever.retrieve(query, n_results=n_results, **filters)

    async def count(self, **filters: Any) -> int:
        """Count documents in the store.

        Args:
            **filters: Equality filters matched against document
                fields.

        Returns:
            Number of matching documents.
        """
        return await self.store.count(**filters)
