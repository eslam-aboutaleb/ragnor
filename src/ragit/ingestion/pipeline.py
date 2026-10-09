"""Generic ingestion pipeline.

The pipeline orchestrates: load source, hash content, compare
the ingestion snapshot, retire the old version, chunk,
embed, validate, and upsert -- inside one transaction,
serialized by an advisory lock taken through the SQL
factory when a lock key is supplied.

Version *storage* is application-side: the pipeline talks
to a :class:`VersionStore` protocol, which the host
application implements against its version tables.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ragit.chunking.base import Chunker
from ragit.config import RagSettings
from ragit.embeddings.base import EmbeddingFunction
from ragit.embeddings.dimensions import get_embedding_dimension
from ragit.ingestion.sources import DocumentSource, RawDocument
from ragit.ingestion.versioning import (
    IngestionSnapshot,
    build_snapshot,
    snapshot_matches,
)
from ragit.stores.base import VectorStore
from ragit.types import Chunk
from ragit.validation import validate_embedding

if TYPE_CHECKING:
    from ragit.db.session import SessionProvider
    from ragit.db.sql_factory import SQLFactory

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
                :class:`ragit.db.sql_factory.SQLFactory`
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

    async def run(  # noqa: PLR0913
        self,
        source: DocumentSource,
        version_store: VersionStore,
        lock_key: str | None = None,
        session: Any = None,
    ) -> int:
        """Ingest the source and return the stored chunk count.

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

        Returns:
            The number of chunks stored, or the stored
            chunk count when the source was unchanged.
        """
        documents = await source.load()
        if not documents:
            logger.warning("No documents loaded from source; skipping ingestion.")
            return 0

        content = "".join(doc.content for doc in documents)
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        sql_factory = None
        if lock_key is not None:
            # Imported lazily so importing the pipeline does
            # not require a database driver; the default is
            # the SQLAlchemy factory.
            from ragit.db.sql_factory import SQLAlchemyFactory  # noqa: PLC0415

            sql_factory = self._sql_factory or SQLAlchemyFactory()

        if session is not None:
            # Caller-supplied session: the caller owns the
            # transaction, so the pipeline never commits it.
            # The advisory lock is still taken on the
            # caller's session when a lock key is supplied.
            if sql_factory is not None:
                async with sql_factory.advisory_lock(session, lock_key):
                    return await self._ingest(session, documents, source_hash, version_store)
            return await self._ingest(session, documents, source_hash, version_store)

        if sql_factory is not None:
            async with (
                self._session_factory() as owned_session,
                sql_factory.advisory_lock(owned_session, lock_key),
            ):
                count = await self._ingest(owned_session, documents, source_hash, version_store)
                await owned_session.commit()
                return count

        async with self._session_factory() as owned_session:
            count = await self._ingest(owned_session, documents, source_hash, version_store)
            await owned_session.commit()
            return count

    async def _ingest(
        self,
        session: Any,
        documents: list[RawDocument],
        source_hash: str,
        version_store: VersionStore,
    ) -> int:
        """Run the ingestion steps on ``session`` (no commit).

        Args:
            session: Session to run the transaction on.
            documents: Loaded documents.
            source_hash: SHA-256 hex digest of the content.
            version_store: Application-side version storage.

        Returns:
            The number of chunks stored, or the stored chunk
            count when the source was unchanged.
        """
        current_snapshot = build_snapshot(source_hash, self._settings, self._chunker)

        active = await version_store.find_active(session)
        if snapshot_matches(current_snapshot, active.snapshot if active else None):
            count = await self._store.count(session=session)
            logger.info(
                "Source unchanged (snapshot %s); skipping ingestion. %d chunks already stored.",
                current_snapshot,
                count,
            )
            return count

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

        embeddings = await self._embedder([chunk.text for chunk in chunks])
        embedding_dim = get_embedding_dimension(self._settings.embedding_model)
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
        for document in documents:
            document_chunks = self._chunker.chunk(document.content)
            for chunk in document_chunks:
                chunk.metadata["source"] = document.source_name
            chunks.extend(document_chunks)
        return chunks
