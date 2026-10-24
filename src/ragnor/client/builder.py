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

import asyncio
import inspect
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ragnor.chunking import SlidingWindowChunker
from ragnor.chunking.base import Chunker
from ragnor.core.config import DeploymentMode, RagSettings
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.dimensions import (
    has_registered_embedding_dimension,
    register_dimension,
)
from ragnor.core.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    OperationTimeoutError,
    RagnorError,
    sanitize_message,
)
from ragnor.ingestion.pipeline import (
    IngestionPipeline,
    VersionRecord,
    VersionStore,
)
from ragnor.ingestion.sources import ListDocumentSource
from ragnor.ingestion.versioning import IngestionSnapshot
from ragnor.core.lifecycle import AsyncCloseable
from ragnor.retrieval.capabilities import BackendFeature, StoreCapabilities
from ragnor.retrieval.retriever import HybridRetriever, RetrieverSpec
from ragnor.retrieval.strategies import HybridRetrieval, Reranker, RetrievalStrategy
from ragnor.stores.base import DocumentStore, LexicalIndex, VectorIndex, VectorStore
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.telemetry.events import EventSink
from ragnor.core.types import Document, SearchResult

if TYPE_CHECKING:
    from ragnor.db.ports import SessionProvider, SQLFactory
    from ragnor.evidence.assembler import EvidenceCompressor
    from ragnor.evidence.tokenizer import Tokenizer
    from ragnor.evidence.types import ContextBudget, EvidenceBundle, SearchHit
    from ragnor.jobs.outbox import EmbeddingJobStore
    from ragnor.retrieval.query_transform import QueryTransformer
    from ragnor.sync.manifest import ManifestStore
    from ragnor.sync.state import SyncStateStore
    from ragnor.client.sync_client import SyncRagnor
    from ragnor.telemetry.health import HealthReport
    from ragnor.tools.security import ScopeAuthorizer


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

    def __repr__(self) -> str:
        masked_key = "***" if self.openai_api_key else ""
        masked_url = sanitize_message(self.database_url) if self.database_url else ""
        return (
            f"BuilderSettings(embedding_model={self.embedding_model!r}, "
            f"rag_distance_threshold={self.rag_distance_threshold!r}, "
            f"vector_store_provider={self.vector_store_provider!r}, "
            f"openai_api_key={masked_key!r}, "
            f"database_url={masked_url!r})"
        )


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
        reranker: Reranker | None = None,
        query_transformer: QueryTransformer | None = None,
        tokenizer: Tokenizer | None = None,
        evidence_compressor: EvidenceCompressor | None = None,
        manifest_store: ManifestStore | None = None,
        sync_state_store: SyncStateStore | None = None,
        job_store: EmbeddingJobStore | None = None,
        authorizer: ScopeAuthorizer | None = None,
        event_sink: EventSink | None = None,
        production_mode: bool = False,
    ) -> None:
        """Initialize the builder."""
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
        self._reranker = reranker
        self._query_transformer = query_transformer
        self._tokenizer = tokenizer
        self._evidence_compressor = evidence_compressor
        self._manifest_store = manifest_store
        self._sync_state_store = sync_state_store
        self._job_store = job_store
        self._authorizer = authorizer
        self._event_sink = event_sink
        self._production_mode = production_mode or (
            getattr(settings, "deployment_mode", None) == DeploymentMode.PRODUCTION
        )
        self._owned_resources: list[AsyncCloseable] = []
        self._startup_check: Callable[[], Awaitable[None]] | None = None
        self._startup_timeout_seconds = 5.0

    def __repr__(self) -> str:
        return (
            f"RagnorBuilder(embedding_model={self._embedding_model!r}, "
            f"embedding_dim={self._embedding_dim!r}, "
            f"production_mode={self._production_mode!r})"
        )

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
        """Build a builder with offline defaults."""
        return cls(
            embedder,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunker=chunker,
            store=store,
            version_store=version_store,
            settings=settings,
        )

    @classmethod
    def production(  # noqa: PLR0913
        cls,
        embedder: EmbeddingFunction,
        *,
        store: VectorStore,
        version_store: VersionStore,
        session_factory: SessionProvider,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int | None = None,
        chunker: Chunker | None = None,
        settings: RagSettings | None = None,
        sql_factory: SQLFactory | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        event_sink: EventSink | None = None,
    ) -> RagnorBuilder:
        """Build a production-mode builder requiring explicit durable adapters."""
        return cls(
            embedder,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            chunker=chunker,
            store=store,
            version_store=version_store,
            settings=settings,
            session_factory=session_factory,
            sql_factory=sql_factory,
            retrieval_strategy=retrieval_strategy,
            event_sink=event_sink,
            production_mode=True,
        )

    def with_embedder(self, embedder: EmbeddingFunction) -> RagnorBuilder:
        """Override the embedder."""
        self._embedder = embedder
        return self

    def with_chunker(self, chunker: Chunker) -> RagnorBuilder:
        """Override the chunker."""
        self._chunker = chunker
        return self

    def with_store(self, store: VectorStore) -> RagnorBuilder:
        """Override the vector store."""
        self._store = store
        return self

    def with_version_store(self, version_store: VersionStore) -> RagnorBuilder:
        """Override the version storage."""
        self._version_store = version_store
        return self

    def with_settings(self, settings: RagSettings) -> RagnorBuilder:
        """Override the settings."""
        self._settings = settings
        return self

    def with_session_factory(self, session_factory: SessionProvider) -> RagnorBuilder:
        """Use a host-provided session provider for ingestion and retrieval."""
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

    def with_reranker(self, reranker: Reranker) -> RagnorBuilder:
        """Inject a custom reranker into the default hybrid retrieval strategy."""
        self._reranker = reranker
        return self

    def with_query_transformer(self, query_transformer: QueryTransformer) -> RagnorBuilder:
        """Inject a query transformer into the default hybrid retrieval strategy."""
        self._query_transformer = query_transformer
        return self

    def with_tokenizer(self, tokenizer: Tokenizer) -> RagnorBuilder:
        """Inject a custom tokenizer for evidence assembly and token accounting."""
        self._tokenizer = tokenizer
        return self

    def with_evidence_compressor(self, compressor: EvidenceCompressor) -> RagnorBuilder:
        """Inject a custom evidence compressor for budget-constrained assembly."""
        self._evidence_compressor = compressor
        return self

    def with_manifest_store(self, manifest_store: ManifestStore) -> RagnorBuilder:
        """Inject a custom manifest store for sync coordination and health checks."""
        self._manifest_store = manifest_store
        return self

    def with_sync_state_store(self, sync_state_store: SyncStateStore) -> RagnorBuilder:
        """Inject a custom sync state store for lease fencing and health checks."""
        self._sync_state_store = sync_state_store
        return self

    def with_job_store(self, job_store: EmbeddingJobStore) -> RagnorBuilder:
        """Inject a custom durable job outbox store."""
        self._job_store = job_store
        return self

    def with_authorizer(self, authorizer: ScopeAuthorizer) -> RagnorBuilder:
        """Inject a custom scope authorizer for tool/MCP access control."""
        self._authorizer = authorizer
        return self

    def with_event_sink(self, event_sink: EventSink) -> RagnorBuilder:
        """Forward retrieval telemetry to a host-supplied event sink."""
        self._event_sink = event_sink
        return self

    def with_owned_resource(self, resource: AsyncCloseable) -> RagnorBuilder:
        """Transfer ownership of an async-closeable resource to the Ragnor facade.

        Unlike injected stores, embedders and sessions, resources registered
        here are closed by `aclose()` in reverse registration order. A resource
        is registered once by identity even if passed repeatedly.
        """
        if not callable(getattr(resource, "aclose", None)):
            raise TypeError("owned resource must implement async aclose()")
        if all(existing is not resource for existing in self._owned_resources):
            self._owned_resources.append(resource)
        return self

    def with_startup_check(
        self,
        check: Callable[[], Awaitable[None]],
        *,
        timeout_seconds: float = 5.0,
    ) -> RagnorBuilder:
        """Require a bounded host readiness check before returning the facade.

        The check must raise on unavailable or unsuitable infrastructure and
        must cooperate with cancellation. It should perform read-only probes
        using the same clients and authenticated scope as normal operations.
        Owned resources transfer before the check starts and are disposed on
        failure, timeout or cancellation. Structural validation happens first;
        if that fails, ownership remains with the host/builder.
        """
        if not callable(check):
            raise ConfigurationError("startup check must be callable")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("startup timeout_seconds must be positive and finite")
        self._startup_check = check
        self._startup_timeout_seconds = timeout_seconds
        return self

    async def _check_startup(self) -> None:
        if self._startup_check is None:
            return
        try:
            deadline = asyncio.timeout(self._startup_timeout_seconds)
            async with deadline:
                result = self._startup_check()
                if not inspect.isawaitable(result):
                    raise ConfigurationError("startup check must return an awaitable")
                await result
            if deadline.expired():
                raise OperationTimeoutError("Startup readiness check exceeded its timeout")
        except RagnorError:
            raise
        except TimeoutError as exc:
            raise OperationTimeoutError("Startup readiness check timed out") from exc
        except Exception as exc:
            raise BackendUnavailableError(
                f"Startup readiness check failed ({type(exc).__name__})"
            ) from exc

    def with_production_mode(self) -> RagnorBuilder:
        """Refuse known ephemeral defaults when building a persistent application."""
        self._production_mode = True
        return self

    def _validate_production_dependencies(self) -> None:
        if getattr(self._settings, "deployment_mode", None) == DeploymentMode.PRODUCTION:
            self._production_mode = True
        if not self._production_mode:
            return
        required = {
            "store": self._store,
            "version_store": self._version_store,
            "session_factory": self._session_factory,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ConfigurationError(
                "production mode requires explicit "
                + ", ".join(missing)
                + "; local defaults are not durable"
            )
        if isinstance(self._store, InMemoryVectorStore) or isinstance(
            self._version_store, InMemoryVersionStore
        ):
            raise ConfigurationError(
                "production mode rejects InMemory stores; "
                "inject and validate durable host implementations"
            )

        caps = getattr(self._store, "capabilities", None)
        if not isinstance(caps, StoreCapabilities):
            raise ConfigurationError(
                "production mode requires explicit StoreCapabilities from "
                "the host storage adapter; class names and method presence "
                "are not evidence of security or durability"
            )
        if not caps.supports_feature(BackendFeature.ISOLATION_SCOPE):
            raise ConfigurationError(
                "production mode requires declared isolation_scope capability; "
                "configure a backend with verified tenant filtering"
            )

        if self._sql_factory is not None and not callable(
            getattr(self._sql_factory, "advisory_lock", None)
        ):
            raise ConfigurationError(
                "production sql_factory must implement advisory_lock(session, lock_key)"
            )

        model = getattr(self._settings, "embedding_model", self._embedding_model)
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("production embedding_model must be a non-empty string")
        if self._embedding_dim is not None and (
            not isinstance(self._embedding_dim, int)
            or isinstance(self._embedding_dim, bool)
            or self._embedding_dim <= 0
        ):
            raise ConfigurationError("production embedding_dim must be a positive integer")
        if self._embedding_dim is None and not has_registered_embedding_dimension(model):
            raise ConfigurationError(
                "production embedding dimension must be explicitly registered "
                "for an unknown embedding_model; refusing implicit fallback"
            )
        declared = getattr(self._store, "embedding_dim", None)
        if (
            declared is not None
            and self._embedding_dim is not None
            and declared != self._embedding_dim
        ):
            raise ConfigurationError(
                "production store embedding dimension does not match the configured embedding_dim"
            )

        if not callable(self._embedder):
            raise ConfigurationError("production embedder must be callable")
        for method in ("embed_query", "embed_documents"):
            if hasattr(self._embedder, method) and not callable(getattr(self._embedder, method)):
                raise ConfigurationError(f"production embedder {method} must be callable")

        # Validate the executable ports, not just descriptive capability flags.
        for name, component, methods in (
            (
                "version_store",
                self._version_store,
                ("find_active", "close_active", "create_version", "delete_chunks"),
            ),
            ("store", self._store, ("upsert", "count")),
        ):
            missing_methods = [
                method for method in methods if not callable(getattr(component, method, None))
            ]
            if missing_methods:
                raise ConfigurationError(
                    f"production {name} must implement: " + ", ".join(missing_methods)
                )
        if not callable(self._session_factory):
            raise ConfigurationError("production session_factory must be callable")

    async def build(self) -> Ragnor:
        """Wire the components into a :class:`Ragnor` facade.

        Returns:
            The wired :class:`Ragnor` facade.

        Raises:
            ConfigurationError: If no embedder is configured or production validation fails.
        """
        if self._embedder is None:
            raise ConfigurationError(
                "no embedder configured: pass embedder= to "
                "RagnorBuilder.local() or call with_embedder() "
                "(ragnor ships no local embedder in core)"
            )

        self._validate_production_dependencies()

        settings = self._settings
        if settings is None:
            settings = BuilderSettings(embedding_model=self._embedding_model)
        elif hasattr(settings, "validate") and callable(settings.validate):
            settings.validate()

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
            version_store = InMemoryVersionStore(cast(DocumentStore, store))

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
                vector_index=cast(VectorIndex, store),
                lexical_index=cast(LexicalIndex, store),
                embedder=self._embedder,
                reranker=self._reranker,
                query_transformer=self._query_transformer,
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
        # Transfer ownership exactly once: a reusable builder must never
        # accidentally register the same closeable in two SDK clients.
        owned_resources = tuple(self._owned_resources)
        self._owned_resources.clear()
        rag = Ragnor(
            store=store,
            pipeline=pipeline,
            retriever=retriever,
            version_store=version_store,
            settings=settings,
            owned_resources=owned_resources,
            production_mode=self._production_mode,
            tokenizer=self._tokenizer,
            evidence_compressor=self._evidence_compressor,
            manifest_store=self._manifest_store,
            sync_state_store=self._sync_state_store,
            job_store=self._job_store,
            authorizer=self._authorizer,
        )
        try:
            await self._check_startup()
        except BaseException as startup_error:
            try:
                await rag.aclose()
            except asyncio.CancelledError:
                raise
            except Exception as cleanup_error:
                raise BaseExceptionGroup(
                    "SDK startup and resource cleanup failed", [startup_error, cleanup_error]
                ) from None
            raise
        return rag

    def build_sync(self, *, timeout: float | None = 60.0) -> SyncRagnor:
        """Build and return a synchronous :class:`~ragnor.client.sync_client.SyncRagnor` facade."""
        from ragnor.client.sync_client import SyncRagnor  # noqa: PLC0415

        return SyncRagnor.from_builder(self, timeout=timeout)


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
        *,
        owned_resources: tuple[AsyncCloseable, ...] = (),
        production_mode: bool = False,
        tokenizer: Tokenizer | None = None,
        evidence_compressor: EvidenceCompressor | None = None,
        manifest_store: ManifestStore | None = None,
        sync_state_store: SyncStateStore | None = None,
        job_store: EmbeddingJobStore | None = None,
        authorizer: ScopeAuthorizer | None = None,
    ) -> None:
        """Initialize the facade."""
        self.store = store
        self.pipeline = pipeline
        self.retriever = retriever
        self._version_store = version_store
        self._settings = settings
        self._owned_resources = owned_resources
        self._production_mode = production_mode
        self.tokenizer = tokenizer
        self.evidence_compressor = evidence_compressor
        self.manifest_store = manifest_store
        self.sync_state_store = sync_state_store
        self.job_store = job_store
        self.authorizer = authorizer
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._active_ops = 0
        self._operations_drained = asyncio.Event()
        self._operations_drained.set()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Ragnor client is closed")

    @asynccontextmanager
    async def _active_operation(self) -> AsyncIterator[None]:
        """Hold a facade-level operation lease until the call completes."""
        self._ensure_open()
        self._active_ops += 1
        self._operations_drained.clear()
        try:
            yield
        finally:
            self._active_ops -= 1
            if self._active_ops == 0:
                self._operations_drained.set()

    async def __aenter__(self) -> Ragnor:
        """Enter an async context managed facade."""
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close explicitly owned resources, leaving borrowed clients open."""
        await self.aclose()

    async def _close_owned_resources(self) -> None:
        errors: list[BaseException] = []
        for resource in reversed(self._owned_resources):
            try:
                await resource.aclose()
            except (Exception, asyncio.CancelledError) as error:
                errors.append(error)
        if len(errors) > 1:
            raise BaseExceptionGroup("Ragnor resource cleanup failed", errors)
        if errors:
            raise errors[0]

    async def _drain_and_close(self) -> None:
        await self._operations_drained.wait()
        await self._close_owned_resources()

    async def aclose(self) -> None:
        """Close explicitly owned resources exactly once.

        All callers await the same cleanup task. Cancelling a caller does not
        abandon disposal of remaining SDK-owned resources.
        """
        if self._close_task is not None and self._close_task.done():
            return
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._drain_and_close())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if self._close_task.done():
                    self._close_task.result()
                    break
        if cancelled:
            raise asyncio.CancelledError

    async def ingest(
        self,
        documents: list[Document],
        *,
        purge_on_empty: bool = False,
        lock_key: str | None = None,
    ) -> int:
        """Ingest documents and return the stored chunk count."""
        async with self._active_operation():
            if lock_key is not None:
                if not isinstance(lock_key, str) or not lock_key.strip():
                    raise ConfigurationError("lock_key must be a non-empty string")
                if self.pipeline._session_factory is _null_session_factory:
                    raise ConfigurationError(
                        "lock_key requires an explicit session_factory; "
                        "local null session cannot hold advisory locks"
                    )
            source = ListDocumentSource(documents)
            try:
                return await self.pipeline.run(
                    source,
                    self._version_store,
                    purge_on_empty=purge_on_empty,
                    lock_key=lock_key,
                )
            except OperationTimeoutError:
                raise
            except TimeoutError as exc:
                raise OperationTimeoutError(f"Ingestion timed out ({type(exc).__name__})") from exc
            except ConnectionError as exc:
                raise BackendUnavailableError(
                    f"Ingestion backend connection failed ({type(exc).__name__})"
                ) from exc

    async def search(
        self,
        query: str,
        n_results: int = 5,
        **filters: Any,
    ) -> list[SearchResult]:
        """Search for documents matching ``query``."""
        async with self._active_operation():
            search_kwargs = dict(filters)
            if self._production_mode and "strict" not in search_kwargs:
                search_kwargs["strict"] = True
            return await self.retriever.retrieve(query, n_results=n_results, **search_kwargs)

    async def count(self, **filters: Any) -> int:
        """Count documents in the store."""
        async with self._active_operation():
            count_kwargs = dict(filters)
            if self._production_mode and "strict" not in count_kwargs:
                count_kwargs["strict"] = True
            try:
                return await self.store.count(**count_kwargs)
            except OperationTimeoutError:
                raise
            except TimeoutError as exc:
                raise OperationTimeoutError(f"Count timed out ({type(exc).__name__})") from exc
            except ConnectionError as exc:
                raise BackendUnavailableError(
                    f"Count backend connection failed ({type(exc).__name__})"
                ) from exc

    def assemble_evidence(
        self,
        hits: Sequence[SearchHit | dict[str, Any] | SearchResult],
        budget: ContextBudget,
        **kwargs: Any,
    ) -> EvidenceBundle:
        """Assemble retrieved hits into a token-budgeted EvidenceBundle."""
        from ragnor.evidence.assembler import EvidenceAssembler  # noqa: PLC0415

        assembler = EvidenceAssembler(
            tokenizer=self.tokenizer,
            compressor=self.evidence_compressor,
            **kwargs,
        )
        return assembler.assemble(hits, budget)

    async def check_health(
        self,
        *,
        scope: str = "default",
        allow_empty: bool = True,
        timeout_seconds: float = 2.0,
        include_expensive: bool = False,
    ) -> HealthReport:
        """Run bounded readiness and configuration checks across wired components."""
        from ragnor.telemetry.health import HealthChecker  # noqa: PLC0415

        async with self._active_operation():
            checker = HealthChecker(
                settings=self._settings,
                vector_store=self.store,
                manifest_store=self.manifest_store,
                sync_state_store=self.sync_state_store,
                job_store=self.job_store,
                scope=scope,
                allow_empty=allow_empty,
                timeout_seconds=timeout_seconds,
            )
            return await checker.check_all(scope=scope, include_expensive=include_expensive)
