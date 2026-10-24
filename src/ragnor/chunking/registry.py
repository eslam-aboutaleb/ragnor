"""Chunker provider registry.

Registers the chunking strategies under their provider
names. Registration happens here (rather than via decorators
in the class modules) so importing this module alone yields a
fully populated registry without import cycles.
"""

from __future__ import annotations

from ragnor.chunking.base import Chunker
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.core.registry import Registry

chunker_registry: Registry[Chunker] = Registry()

chunker_registry.register("sliding-window")(SlidingWindowChunker)
chunker_registry.register("markdown")(MarkdownSectionChunker)
