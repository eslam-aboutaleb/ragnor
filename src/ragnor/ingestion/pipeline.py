"""Generic ingestion pipeline.

The pipeline orchestrates: load source, hash the
source manifest, compare the ingestion snapshot,
chunk, embed, and validate -- outside any
transaction -- then retire the old version and
upsert the new chunks inside one short
transaction, serialized by an advisory lock taken
through the SQL factory when a lock key is
supplied.

Splitting the work into a prepare phase and a
short critical section keeps a slow remote
embedding request from holding a transaction and
an advisory lock open: the embedding happens
before any session is opened for the version
mutations, and the critical section re-checks the
snapshot under the lock (double-checked locking)
so a concurrent process that ingested meanwhile
still wins.

Version *storage* is application-side: the pipeline talks
to a :class:`VersionStore` protocol, which the host
application implements against its version tables.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ragnor.chunking.base import Chunker
from ragnor.core.config import RagSettings
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.ingestion.sources import DocumentSource, RawDocument
from ragnor.ingestion.connector_core import LoadResult
from ragnor.ingestion.versioning import (
    IngestionSnapshot,
    build_snapshot,
    compute_source_hash,
    snapshot_matches,
)
from ragnor.stores.base import VectorStore
from ragnor.core.types import Chunk
from ragnor.core.validation import validate_embedding

if TYPE_CHECKING:
    from ragnor.db.session import SessionProvider
    from ragnor.db.sql_factory import SQLFactory

logger = logging.getLogger(__name__)


@dataclass
class VersionRecord:
    """A stored ingestion version and its snapshot.

    Attributes:
        version_id: Identifier of the version row.
        snapshot: The snapshot stored with the version.
        source_id: Optional identifier of the source the
            version belongs to (e.g., a document id).
    """

    version_id: str
    snapshot: IngestionSnapshot
    source_id: str | None = None


@runtime_checkable
class VersionStore(Protocol):
    """Storage for ingestion versions (application-side).

    The host application implements this protocol against
    its version tables. All methods
    receive the pipeline's session so every version
    mutation joins the ingestion transaction.
    """

    async def find_active(self, session: Any) -> VersionRecord | None:
        """Return the active version for the source, or ``None``."""
        ...

    async def close_active(self, session: Any) -> None:
        """Retire the active version (set its end timestamp)."""
        ...

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        """Insert a new version row carrying ``snapshot``."""
        ...

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        """Delete the chunks belonging to ``version_id``."""
        ...


class IngestionPipeline:
    """Chunks, embeds, and stores documents from a source."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        store: VectorStore,
        embedder: EmbeddingFunction,
        chunker: Chunker,
        session_factory: SessionProvider,
        settings: RagSettings,
        version_id_key: str = "version_id",
        source_id_key: str = "source_id",
        sql_factory: SQLFactory | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            store: Vector store the chunks are upserted into.
            embedder: Embedding function for chunk texts.
            chunker: Chunker that splits document content.
            session_factory: Session provider the pipeline
                owns its transactions with.
            settings: Settings carrying the embedding model.
            version_id_key: Metadata key the version id is
                stamped under (the target table's version
                column name).
            source_id_key: Metadata key the source id is
                stamped under (the target table's source
                column name).
            sql_factory: SQL factory used for advisory
                locking and session creation; defaults to
                the SQLAlchemy factory. Inject a custom
                :class:`ragnor.db.sql_factory.SQLFactory`
                to run the pipeline on another stack.
        """
        self._store = store
        self._embedder = embedder
        self._chunker = chunker
        self._session_factory = session_factory
        self._settings = settings
        self._version_id_key = version_id_key
        self._source_id_key = source_id_key
        self._sql_factory = sql_factory
        self._last_load_result = None

    async def run(  # noqa: PLR0913, PLR0911
        self,
        source: DocumentSource,
        version_store: VersionStore,
        lock_key: str | None = None,
        session: Any = None,
        purge_on_empty: bool = False,
    ) -> int:
        """Ingest the source and return the stored chunk count.

        The run has two phases. The *prepare* phase opens
        no transaction and takes no lock: it loads the
        source, hashes the source manifest, compares the
        snapshot (an unchanged source skips without
        embedding), chunks, embeds, and validates. The
        *critical section* is short: it re-checks the
        snapshot under the advisory lock (double-checked
        locking -- a concurrent process may have ingested
        the source while the embedding request was in
        flight), retires the old version, creates the new
        one, and upserts the chunks.

        The pipeline owns its session -- and commits it --
        when a ``lock_key`` is supplied: the advisory lock
        serializes concurrent ingestion of the same source
        across instances. A caller-supplied ``session`` is
        never committed; the caller owns its transaction
        (and its locking).

        Args:
            source: Document source to load from.
            version_store: Application-side version storage.
            lock_key: Advisory-lock key (the source path).
                When supplied, the pipeline opens its own
                session, locks it, and commits it.
            session: Caller-supplied session. When supplied,
                the pipeline runs on it without committing.
            purge_on_empty: Treat an empty load as
                authoritative. When ``True``, an empty
                source retires the active version and
                deletes its chunks from the search index,
                so deleted content stops being searchable;
                when ``False`` (the default), an empty
                load is a historical no-op.

        Returns:
            The number of chunks stored, or the stored
            chunk count when the source was unchanged.
        """
        if hasattr(source, "load"):
            loaded = await source.load()
        else:
            loaded = source
        if isinstance(loaded, LoadResult):
            documents = list(loaded)
            self._last_load_result = loaded
            if not loaded.authoritative:
                logger.warning(
                    "Non-authoritative source scan; skipping ingestion without mutation."
                )
                return 0
        else:
            documents = list(loaded) if not isinstance(loaded, list) else loaded
            self._last_load_result = None

        sql_factory = None
        if lock_key is not None:
            sql_factory = self._sql_factory
            if sql_factory is None:
                # Import the default only when no host adapter was supplied:
                # SQLAlchemy is an optional dependency.
                from ragnor.db.sql_factory import SQLAlchemyFactory  # noqa: PLC0415

                sql_factory = SQLAlchemyFactory()

        if not documents:
            if not purge_on_empty:
                logger.warning("No documents loaded from source; skipping ingestion.")
                return 0
            # An authoritative empty dataset: retire the
            # active version so deleted content stops being
            # searchable.
            async with self._transacting_session(lock_key, sql_factory, session) as purge_session:
                await self._retire_active(purge_session, version_store)
            return 0

        source_hash = compute_source_hash(documents)
        extractor = getattr(source, "extractor", None)
        if extractor is None and documents and documents[0].extractor:
            first_ext = documents[0].extractor
            if all(doc.extractor == first_ext for doc in documents):
                extractor = first_ext
        current_snapshot = build_snapshot(
            source_hash,
            self._settings,
            self._chunker,
            extractor=extractor,
            embedding_profile=getattr(self._embedder, "profile", None),
        )

        # Prepare phase: no transaction, no lock. The
        # read-only snapshot check runs on a short-lived
        # session (or read-only on the caller's session),
        # so an unchanged source skips without embedding.
        if session is not None:
            # Caller-supplied session: the caller owns the
            # transaction, so the pipeline never commits it.
            active = await version_store.find_active(session)
            if snapshot_matches(current_snapshot, active.snapshot if active else None):
                return await self._skip_unchanged(session, current_snapshot)
        else:
            async with self._session_factory() as check_session:
                active = await version_store.find_active(check_session)
                if snapshot_matches(current_snapshot, active.snapshot if active else None):
                    return await self._skip_unchanged(check_session, current_snapshot)

        chunks = self._chunk(documents)
        if not chunks:
            # Abort before touching the version table or the
            # search index: a version row carrying the current
            # snapshot would make every later run skip this
            # source forever (the snapshot would compare equal),
            # leaving it permanently un-ingested. Aborting here
            # keeps the previous version active so the index
            # still reflects it and the next run retries.
            logger.warning("No chunks extracted from source. Ingestion aborted.")
            return 0

        # Embedding and validation run outside any transaction
        # and outside the advisory lock: a slow remote call
        # must not hold either open.
        embeddings = await self._embedder([chunk.text for chunk in chunks])
        embedding_dim = current_snapshot.embedding_dim
        documents_to_upsert: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            validate_embedding(embedding, expected_dim=embedding_dim, label="chunk embedding")
            documents_to_upsert.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

        # Critical section: short. The snapshot is re-checked
        # under the lock -- a concurrent process may have
        # ingested this source while the embedding request
        # was in flight.
        async with self._transacting_session(lock_key, sql_factory, session) as critical_session:
            return await self._commit_ingestion(
                critical_session, current_snapshot, documents_to_upsert, version_store
            )

    @asynccontextmanager
    async def _transacting_session(
        self,
        lock_key: str | None,
        sql_factory: SQLFactory | None,
        session: Any,
    ) -> AsyncIterator[Any]:
        """Yield the session a critical section transacts on.

        A caller-supplied session is yielded as-is -- the
        caller owns its transaction, so it is never
        committed -- and the advisory lock is taken on it
        when a lock key is supplied. Otherwise the
        pipeline opens its own session, takes the advisory
        lock on it when a lock key is supplied, and
        commits it when the block succeeds.

        Args:
            lock_key: Advisory-lock key, or ``None``.
            sql_factory: SQL factory for the advisory lock,
                or ``None`` when no lock key was supplied.
            session: Caller-supplied session, or ``None``.

        Yields:
            The session the critical section runs on.
        """
        if session is not None:
            # Caller-supplied session: the caller owns the
            # transaction, so the pipeline never commits it.
            # The advisory lock is still taken on the
            # caller's session when a lock key is supplied.
            if sql_factory is not None:
                assert lock_key is not None
                async with sql_factory.advisory_lock(session, lock_key):
                    yield session
            else:
                yield session
            return

        if sql_factory is not None:
            assert lock_key is not None
            async with (
                self._session_factory() as owned_session,
                sql_factory.advisory_lock(owned_session, lock_key),
            ):
                yield owned_session
                await owned_session.commit()
            return

        async with self._session_factory() as owned_session:
            yield owned_session
            await owned_session.commit()

    async def _retire_active(
        self,
        session: Any,
        version_store: VersionStore,
    ) -> None:
        """Close the active version and delete its chunks (no commit).

        Args:
            session: Session to run the retirement on.
            version_store: Application-side version storage.
        """
        active = await version_store.find_active(session)
        if active is None:
            return
        await version_store.close_active(session)
        await version_store.delete_chunks(session, active.version_id)

    async def _skip_unchanged(
        self,
        session: Any,
        current_snapshot: IngestionSnapshot,
    ) -> int:
        """Skip ingestion of an unchanged source (no commit).

        Args:
            session: Session the stored chunk count is read on.
            current_snapshot: The snapshot that matched.

        Returns:
            The stored chunk count.
        """
        count = await self._store.count(session=session)
        logger.info(
            "Source unchanged (snapshot %s); skipping ingestion. %d chunks already stored.",
            current_snapshot,
            count,
        )
        return count

    async def _commit_ingestion(  # noqa: PLR0913
        self,
        session: Any,
        current_snapshot: IngestionSnapshot,
        documents_to_upsert: list[dict[str, Any]],
        version_store: VersionStore,
    ) -> int:
        """Run the version-table mutations and the upsert on ``session`` (no commit).

        The snapshot is re-checked first (double-checked
        locking): a concurrent process may have ingested
        this source while the embedding request was in
        flight, in which case the critical section skips
        without touching the version table or the index.

        Args:
            session: Session to run the transaction on.
            current_snapshot: The snapshot to record.
            documents_to_upsert: Validated chunk documents
                (id, text, metadata, embedding).
            version_store: Application-side version storage.

        Returns:
            The number of chunks stored, or the stored chunk
            count when a concurrent run ingested the source.
        """
        active = await version_store.find_active(session)
        if snapshot_matches(current_snapshot, active.snapshot if active else None):
            return await self._skip_unchanged(session, current_snapshot)

        if active is not None:
            # Close the old active version and remove its
            # chunks from the search index. chunk_id is
            # position-based, so the new version's chunks
            # would violate the unique constraint on chunk_id
            # if the retired version's chunks stayed. The
            # index always reflects the current version;
            # version history itself remains in the version
            # table.
            await version_store.close_active(session)
            await version_store.delete_chunks(session, active.version_id)

        version = await version_store.create_version(session, current_snapshot)

        for document in documents_to_upsert:
            document["metadata"][self._version_id_key] = version.version_id
            if version.source_id is not None:
                document["metadata"][self._source_id_key] = version.source_id

        await self._store.upsert(documents_to_upsert, session=session)
        return len(documents_to_upsert)

    def _chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        """Chunk the loaded documents, stamping each with its source name.

        The source name comes from the loaded document, not
        from the chunker, so a multi-document source stamps
        every chunk with the document it actually came from.
        """
        chunks: list[Chunk] = []
        for index, document in enumerate(documents):
            document_chunks = self._chunker.chunk(document.content)
            for chunk in document_chunks:
                if len(documents) > 1:
                    chunk.id = f"document_{index}:{chunk.id}"
                chunk.metadata = {**document.metadata, **chunk.metadata}
                evidence = chunk.metadata.get("page_extraction")
                if isinstance(evidence, dict) and isinstance(evidence.get("evidence_sha256"), str):
                    # RawDocument retains full geometry; avoid replicating a whole
                    # page's word/table payload into every vector-store chunk.
                    compact = dict(evidence)
                    for key in ("words", "regions", "tables"):
                        if key in compact:
                            compact[key + "_count"] = len(compact.pop(key))
                    chunk.metadata["page_extraction"] = compact
                chunk.metadata["source"] = document.source_name
                if document.external_id is not None:
                    chunk.metadata["external_id"] = document.external_id
                if document.provenance:
                    chunk.metadata["provenance"] = dict(document.provenance)
                if document.location is not None:
                    chunk.metadata["location"] = document.location
                if document.extractor is not None:
                    chunk.metadata["extractor"] = document.extractor
            chunks.extend(document_chunks)
        return chunks
