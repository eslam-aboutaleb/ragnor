"""Chunking seam: chunker protocol, sliding-window and Markdown strategies."""

from __future__ import annotations

from ragit.chunking.base import Chunker
from ragit.chunking.markdown import MarkdownSectionChunker
from ragit.chunking.registry import chunker_registry
from ragit.chunking.sliding_window import SlidingWindowChunker, sliding_window_chunk

__all__ = [
    "Chunker",
    "MarkdownSectionChunker",
    "SlidingWindowChunker",
    "chunker_registry",
    "sliding_window_chunk",
]
