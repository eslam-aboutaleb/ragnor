"""LlamaIndex readers and documents as a ragnor ``DocumentSource``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ragnor.ingestion.records import RawDocument


class LlamaIndexDocumentSource:
    """Adapt a LlamaIndex reader (or ready ``Document`` list) to ragnor.

    ``load`` follows the ragnor contract: reader failures propagate, and an
    empty list is returned only when the reader itself returned no documents.
    Reader keyword arguments go through ``reader_kwargs`` to ``load_data``.
    Readers that implement ``aload_data`` are awaited; others run in a thread.
    """

    def __init__(
        self,
        reader: BaseReader | Sequence[Document],
        *,
        reader_kwargs: dict[str, Any] | None = None,
        source_name: str = "llamaindex",
    ) -> None:
        self._reader = reader
        self._kwargs = dict(reader_kwargs or {})
        self._source_name = source_name

    async def _documents(self) -> list[Document]:
        if isinstance(self._reader, BaseReader):
            return list(await self._reader.aload_data(**self._kwargs))
        return list(self._reader)

    async def load(self) -> list[RawDocument]:
        """Return the reader's documents as :class:`RawDocument`."""
        docs = await self._documents()
        return [self._convert(doc, index) for index, doc in enumerate(docs)]

    def _convert(self, doc: Document, index: int) -> RawDocument:
        metadata = dict(doc.metadata or {})
        name = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("source")
            or doc.doc_id
            or f"{self._source_name}-{index}"
        )
        return RawDocument(
            source_name=str(name),
            content=doc.get_content(),
            external_id=doc.doc_id,
            metadata=metadata,
            extractor="llamaindex",
        )


__all__ = ["LlamaIndexDocumentSource"]
