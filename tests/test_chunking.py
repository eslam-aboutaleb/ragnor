"""Tests for ragit.chunking.

Covers the sliding-window chunker -- including the word-count
semantics pinned by the original ingest implementation -- the
Markdown section chunker (metadata keys, preamble handling,
title extraction), and the chunker registry.
"""

from __future__ import annotations

import pytest

from ragit.chunking.base import Chunker
from ragit.chunking.markdown import MarkdownSectionChunker
from ragit.chunking.registry import chunker_registry
from ragit.chunking.sliding_window import (
    SlidingWindowChunker,
    sliding_window_chunk,
)
from ragit.types import Chunk

SAMPLE_DOCUMENT = """# Sample Policy

This is the preamble.

## Coverage

Coverage details go here.

## Exclusions

Exclusion details go here.
"""


class TestSlidingWindowChunk:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert sliding_window_chunk("") == []
        assert sliding_window_chunk("   ") == []

    def test_short_text_yields_a_single_chunk(self) -> None:
        assert sliding_window_chunk("hello world") == ["hello world"]

    def test_1300_word_text_yields_three_chunks_with_100_word_overlap(
        self,
    ) -> None:
        """Spot-check: the move preserves the original word-count semantics.

        With chunk_size=600 and overlap=100 the step is 500 words, so a
        1300-word text splits into 600 + 600 + 300 words and consecutive
        chunks share exactly 100 words.
        """
        text = " ".join(f"w{i}" for i in range(1300))

        chunks = sliding_window_chunk(text)

        assert len(chunks) == 3
        first, second, third = (chunk.split() for chunk in chunks)
        assert len(first) == 600
        assert len(second) == 600
        assert len(third) == 300
        assert first[-100:] == second[:100]
        assert second[-100:] == third[:100]

    def test_custom_chunk_size_and_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))

        chunks = sliding_window_chunk(text, chunk_size=4, overlap=2)

        assert chunks == [
            "w0 w1 w2 w3",
            "w2 w3 w4 w5",
            "w4 w5 w6 w7",
            "w6 w7 w8 w9",
        ]

    def test_zero_chunk_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            sliding_window_chunk("hello", chunk_size=0)

    def test_negative_chunk_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            sliding_window_chunk("hello", chunk_size=-3)

    def test_overlap_equal_to_chunk_size_is_rejected(self) -> None:
        """overlap == chunk_size makes the step zero (range() error)."""
        with pytest.raises(ValueError, match="overlap must satisfy"):
            sliding_window_chunk("hello world", chunk_size=4, overlap=4)

    def test_overlap_greater_than_chunk_size_is_rejected(self) -> None:
        """overlap > chunk_size would silently yield no chunks."""
        with pytest.raises(ValueError, match="overlap must satisfy"):
            sliding_window_chunk("hello world", chunk_size=4, overlap=5)

    def test_negative_overlap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap must satisfy"):
            sliding_window_chunk("hello world", chunk_size=4, overlap=-1)

    def test_zero_overlap_is_allowed(self) -> None:
        chunks = sliding_window_chunk("a b c d", chunk_size=2, overlap=0)

        assert chunks == ["a b", "c d"]


class TestSlidingWindowChunker:
    def test_version(self) -> None:
        assert SlidingWindowChunker().version == "sliding-window-v1"

    def test_constructor_rejects_overlap_equal_to_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must satisfy"):
            SlidingWindowChunker(chunk_size=100, overlap=100)

    def test_constructor_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            SlidingWindowChunker(chunk_size=0)

    def test_config_exposes_chunk_size_and_overlap(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=600, overlap=100)

        assert chunker.config == {"chunk_size": 600, "overlap": 100}

    def test_chunk_matches_sliding_window_chunk_output(self) -> None:
        chunker = SlidingWindowChunker()
        text = " ".join(f"w{i}" for i in range(1300))

        chunks = chunker.chunk(text)

        assert [chunk.text for chunk in chunks] == sliding_window_chunk(text)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.id for chunk in chunks)

    def test_satisfies_the_chunker_protocol(self) -> None:
        assert isinstance(SlidingWindowChunker(), Chunker)


class TestMarkdownSectionChunker:
    def test_splits_on_h2_headers_and_extracts_the_preamble_title(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(), source_name="policy.md")

        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        assert {chunk.metadata["section"] for chunk in chunks} == {
            "Sample Policy",
            "Coverage",
            "Exclusions",
        }

    def test_metadata_keys_are_exactly_the_frozen_set(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(), source_name="policy.md")

        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        for chunk in chunks:
            assert set(chunk.metadata) == {
                "chunk_id",
                "section",
                "source",
                "chunk_index",
                "sub_chunk_index",
            }

    def test_chunk_id_pattern_indices_and_source(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(), source_name="policy.md")

        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        for index, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == f"chunk_{index}"
            assert chunk.metadata["chunk_index"] == index
            assert chunk.metadata["sub_chunk_index"] == 0
            assert chunk.metadata["source"] == "policy.md"

    def test_section_text_is_reconstructed_with_its_h2_header(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(), source_name="policy.md")

        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        coverage = next(chunk for chunk in chunks if chunk.metadata["section"] == "Coverage")
        # The H2 header is reconstructed before sliding-window chunking,
        # which normalizes whitespace, so the title leads the chunk text.
        assert coverage.text.startswith("## Coverage")
        assert "Coverage details go here." in coverage.text

    def test_preamble_without_a_title_defaults_to_introduction(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(), source_name="p.md")

        chunks = chunker.chunk("Just preamble text without any header.")

        assert chunks[0].metadata["section"] == "Introduction"

    def test_version(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker())

        assert chunker.version == "section-aware-v1"

    def test_config_delegates_to_the_inner_chunker(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker(chunk_size=600, overlap=100))

        assert chunker.config == {"chunk_size": 600, "overlap": 100}

    def test_multiple_sub_chunks_per_section(self) -> None:
        inner = SlidingWindowChunker(chunk_size=10, overlap=2)
        chunker = MarkdownSectionChunker(inner=inner, source_name="p.md")
        text = "## Big Section\n\n" + " ".join(f"w{i}" for i in range(30))

        chunks = chunker.chunk(text)

        assert len(chunks) == 4
        assert [chunk.metadata["sub_chunk_index"] for chunk in chunks] == [0, 1, 2, 3]
        assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2, 3]

    def test_satisfies_the_chunker_protocol(self) -> None:
        chunker = MarkdownSectionChunker(inner=SlidingWindowChunker())

        assert isinstance(chunker, Chunker)


class TestChunkerRegistry:
    def test_chunkers_are_registered(self) -> None:
        assert chunker_registry.get("sliding-window") is SlidingWindowChunker
        assert chunker_registry.get("markdown") is MarkdownSectionChunker

    def test_lookup_is_case_insensitive(self) -> None:
        assert chunker_registry.get("Sliding-Window") is SlidingWindowChunker
        assert chunker_registry.get("MARKDOWN") is MarkdownSectionChunker

    def test_unknown_chunker_raises_key_error_listing_registered(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            chunker_registry.get("nope")
