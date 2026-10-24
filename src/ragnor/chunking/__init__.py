"""Chunking seam: chunker protocol, sliding-window and Markdown strategies."""

from __future__ import annotations

from ragnor.chunking.base import Chunker
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.registry import chunker_registry
from ragnor.chunking.sliding_window import SlidingWindowChunker, sliding_window_chunk
from ragnor.chunking.token_aware import TokenAwareChunker

__all__ = [
    "Chunker",
    "MarkdownSectionChunker",
    "SlidingWindowChunker",
    "TokenAwareChunker",
    "chunker_registry",
    "sliding_window_chunk",
]
