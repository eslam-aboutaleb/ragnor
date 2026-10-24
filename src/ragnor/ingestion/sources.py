"""Document source abstractions for ingestion.

A :class:`DocumentSource` loads
:class:`RawDocument` objects; the ingestion pipeline
hashes their content, chunks it, and embeds it.

Sources are the extension point for backends ragnor
does not ship (S3, databases, APIs): implement the
protocol and pass the source to
:meth:`ragnor.ingestion.pipeline.IngestionPipeline.run`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragnor.ingestion.records import RawDocument
from ragnor.core.types import Document

if TYPE_CHECKING:
    from ragnor.ingestion.connector_core import DiscoveryPage, LoadResult, SourceConnector

logger = logging.getLogger(__name__)


@runtime_checkable
class DocumentSource(Protocol):
    """Loads documents for ingestion.

    Contract: ``load`` **raises** on failure (network
    error, permission denied, malformed response) and
    returns an empty list **only** for an authoritative
    empty dataset -- a source that genuinely has no
    documents. A sync process relies on the distinction:
    an empty load is a no-op by default, and
    ``purge_on_empty=True`` on
    :meth:`ragnor.ingestion.pipeline.IngestionPipeline.run`
    treats it as authoritative, retiring the active
    version so deleted content stops being searchable.
    """

    async def load(self) -> list[RawDocument]:
        """Return the documents this source provides."""
        ...


class FileDocumentSource:
    """Document source reading a single file from disk."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        """Initialize the source.

        Args:
            path: Path to the document file.
            encoding: File encoding. Defaults to UTF-8.
        """
        self._path = path
        self._encoding = encoding

    async def load(self) -> list[RawDocument]:
        """Load the file as one :class:`RawDocument`.

        A missing file is not an error: ingestion of an
        absent document is a no-op, so the source yields
        no documents and logs a warning (the historical
        behavior for a missing document file).

        The read runs in a worker thread so a large file
        never blocks the event loop.

        Returns:
            One document named after the file, or an empty
            list when the file does not exist.
        """
        try:
            content = await asyncio.to_thread(self._read)
        except FileNotFoundError:
            logger.warning("Document not found at '%s'; skipping.", self._path)
            return []
        return [RawDocument(source_name=Path(self._path).name, content=content)]

    def _read(self) -> str:
        """Read the file contents (runs in a worker thread)."""
        with open(self._path, encoding=self._encoding) as f:
            return f.read()


class ListDocumentSource:
    """Document source over a fixed list of documents.

    Accepts :class:`ragnor.core.types.Document` objects (mapping
    ``Document.id`` to the raw document's source name and external
    id) or pre-built :class:`RawDocument` objects, which pass
    through unchanged. Useful beyond the builder -- tests and
    one-shot ingestion of in-memory documents.
    """

    def __init__(self, documents: Sequence[Document | RawDocument]) -> None:
        """Initialize the source with a fixed document list.

        Args:
            documents: Documents to serve; ``Document`` objects are
                converted to :class:`RawDocument` on load.
        """
        self._documents = list(documents)

    async def load(self) -> list[RawDocument]:
        """Return the fixed documents as :class:`RawDocument` objects.

        Returns:
            The documents; a ``Document``'s id is recorded as both the
            source name and the external id.
        """
        return [
            document
            if isinstance(document, RawDocument)
            else RawDocument(
                source_name=document.id,
                content=document.text,
                external_id=document.id,
                metadata=dict(document.metadata),
            )
            for document in self._documents
        ]


class DocumentSourceConnector:
    """Adapt a legacy source to connector pagination.

    Legacy empty lists are ambiguous: the adapter must trust their historical
    meaning, except FileDocumentSource whose missing-file behavior is detected
    separately. The source is loaded once per traversal so pages share a snapshot.
    """

    def __init__(self, source: DocumentSource) -> None:
        self.source = source
        self._documents: list[RawDocument] = []
        self._scan: LoadResult | None = None

    async def _start(self) -> LoadResult:
        from ragnor.ingestion.connector_core import (  # noqa: PLC0415
            LoadResult,
            SourceStatus,
            _error_record,
        )

        try:
            if isinstance(self.source, FileDocumentSource):
                # Read directly: legacy load() deliberately hides FileNotFoundError.
                content = await asyncio.to_thread(self.source._read)
                loaded = [RawDocument(Path(self.source._path).name, content)]
            else:
                loaded = await self.source.load()
        except FileNotFoundError as exc:
            return LoadResult(
                status=SourceStatus.NOT_FOUND, errors=[_error_record(exc)], complete=False
            )
        except PermissionError as exc:
            return LoadResult(
                status=SourceStatus.INACCESSIBLE, errors=[_error_record(exc)], complete=False
            )
        except Exception as exc:
            return LoadResult(
                status=SourceStatus.ERROR, errors=[_error_record(exc)], complete=False
            )
        return loaded if isinstance(loaded, LoadResult) else LoadResult(loaded)

    async def load_page(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        from ragnor.ingestion.connector_core import (  # noqa: PLC0415
            LoadResult,
            SourceStatus,
            _error_record,
        )

        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if cursor is None:
            result = await self._start()
            if not result.authoritative:
                return result
            self._documents = list(result)
            self._scan = result
        try:
            offset = 0 if cursor is None else int(cursor)
            if offset < 0 or offset > len(self._documents):
                raise ValueError("Invalid adapter cursor")
        except (TypeError, ValueError) as exc:
            return LoadResult(
                status=SourceStatus.ERROR, errors=[_error_record(exc)], complete=False
            )
        end = len(self._documents) if limit is None else offset + limit
        documents = self._documents[offset:end]
        has_more = end < len(self._documents)
        return LoadResult(
            documents,
            cursor=cursor,
            has_more=has_more,
            next_cursor=str(end) if has_more else None,
            total_items=len(documents),
            skipped=self._scan.skipped if self._scan is not None and cursor is None else [],
            metadata=self._scan.metadata if self._scan is not None else {},
            checkpoint=self._scan.checkpoint if self._scan is not None else None,
        )

    async def discover(self, *, cursor: Any = None, limit: int | None = None) -> DiscoveryPage:
        from ragnor.ingestion.connector_core import DiscoveryPage, SourceItem  # noqa: PLC0415

        page = await self.load_page(cursor=cursor, limit=limit)
        return DiscoveryPage(
            [SourceItem(source_name=doc.source_name, external_id=doc.external_id) for doc in page],
            cursor=cursor,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
            complete=page.complete,
            status=page.status,
            errors=page.errors,
            skipped=page.skipped,
        )

    async def load(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        from ragnor.ingestion.connector_core import collect_pages  # noqa: PLC0415

        if cursor is not None:
            return await self.load_page(cursor=cursor, limit=limit)
        return await collect_pages(self, page_size=limit)


class ConnectorDocumentSource:
    """Preserve LoadResult authority information at the legacy source boundary."""

    def __init__(self, connector: SourceConnector) -> None:
        self.connector = connector

    @property
    def extractor(self) -> Any:
        return getattr(self.connector, "extractor", None)

    async def load(self) -> LoadResult:
        return await self.connector.load()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connector, name)
