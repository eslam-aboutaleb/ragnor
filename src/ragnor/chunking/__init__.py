"""Chunking seam: chunker protocol, sliding-window and Markdown strategies."""

from __future__ import annotations

from ragnor.chunking.base import Chunker
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.registry import chunker_registry
from ragnor.chunking.sliding_window import SlidingWindowChunker, sliding_window_chunk
from ragnor.chunking.structured import (
    OversizeUnitError,
    StructuredChunkingError,
    StructuredDocument,
    StructuredDocumentChunker,
    StructuredRegion,
    TableRow,
    markdown_to_structured,
    structured_from_pages,
)
from ragnor.chunking.token_aware import TokenAwareChunker

__all__ = [
    "Chunker",
    "OversizeUnitError",
    "StructuredChunkingError",
    "StructuredDocument",
    "StructuredDocumentChunker",
    "StructuredRegion",
    "TableRow",
    "markdown_to_structured",
    "structured_from_pages",
    "MarkdownSectionChunker",
    "SlidingWindowChunker",
    "TokenAwareChunker",
    "chunker_registry",
    "sliding_window_chunk",
]
