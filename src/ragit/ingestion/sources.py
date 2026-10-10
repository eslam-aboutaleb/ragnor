"""Document source abstractions for ingestion.

A :class:`DocumentSource` loads
:class:`RawDocument` objects; the ingestion pipeline
hashes their content, chunks it, and embeds it.

Sources are the extension point for backends ragit
does not ship (S3, databases, APIs): implement the
protocol and pass the source to
:meth:`ragit.ingestion.pipeline.IngestionPipeline.run`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragit.types import Document

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """A loaded document, before chunking.

    Attributes:
        source_name: Identifier recorded in chunk metadata
            (typically the filename).
        content: The full document text.
        external_id: Optional identifier from the source
            system (e.g., an object key or row id).
    """

    source_name: str
    content: str
    external_id: str | None = None


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
    :meth:`ragit.ingestion.pipeline.IngestionPipeline.run`
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

    Accepts :class:`ragit.types.Document` objects (mapping
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
            )
            for document in self._documents
        ]
