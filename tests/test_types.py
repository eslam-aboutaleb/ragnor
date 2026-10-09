"""Tests for ragit.types.

The dataclasses in :mod:`ragit.types` are the canonical boundary
across chunking, embedding, storage, and retrieval. These tests pin
their defaults, mutability semantics, and the typed retrieval error.
"""

from __future__ import annotations

import pytest

from ragit.types import (
    Chunk,
    Document,
    RetrievalError,
    SearchResult,
)


class TestChunk:
    def test_defaults(self) -> None:
        chunk = Chunk(id="c1", text="hello")

        assert chunk.id == "c1"
        assert chunk.text == "hello"
        assert chunk.metadata == {}

    def test_metadata_is_independent_per_instance(self) -> None:
        """The ``default_factory`` must not share a dict across chunks."""
        first = Chunk(id="c1", text="a")
        second = Chunk(id="c2", text="b")

        first.metadata["section"] = "Coverage"

        assert second.metadata == {}

    def test_metadata_is_preserved(self) -> None:
        chunk = Chunk(id="c1", text="a", metadata={"section": "Coverage"})

        assert chunk.metadata == {"section": "Coverage"}

    def test_equality_is_value_based(self) -> None:
        assert Chunk(id="c1", text="a") == Chunk(id="c1", text="a")
        assert Chunk(id="c1", text="a") != Chunk(id="c2", text="a")


class TestDocument:
    def test_defaults(self) -> None:
        document = Document(id="d1", text="hello")

        assert document.id == "d1"
        assert document.text == "hello"
        assert document.metadata == {}
        assert document.embedding is None

    def test_embedding_is_optional(self) -> None:
        document = Document(id="d1", text="hello", embedding=[0.1, 0.2])

        assert document.embedding == [0.1, 0.2]

    def test_metadata_is_independent_per_instance(self) -> None:
        first = Document(id="d1", text="a")
        second = Document(id="d2", text="b")

        first.metadata["owner_id"] = "u1"

        assert second.metadata == {}


class TestSearchResult:
    def test_defaults(self) -> None:
        result = SearchResult(id="r1", document="text")

        assert result.id == "r1"
        assert result.document == "text"
        assert result.metadata == {}
        assert result.distance is None
        assert result.keyword_score is None
        assert result.rrf_score == 0.0

    def test_vector_only_hit_has_no_keyword_score(self) -> None:
        """A vector-only hit carries a distance but no keyword score."""
        result = SearchResult(id="r1", document="text", distance=0.25)

        assert result.distance == 0.25
        assert result.keyword_score is None

    def test_keyword_only_hit_has_no_distance(self) -> None:
        """A keyword-only hit carries a keyword score but no distance."""
        result = SearchResult(id="r1", document="text", keyword_score=0.75)

        assert result.distance is None
        assert result.keyword_score == 0.75

    def test_hybrid_hit_carries_both_scores(self) -> None:
        result = SearchResult(
            id="r1",
            document="text",
            distance=0.25,
            keyword_score=0.75,
            rrf_score=0.0322,
        )

        assert result.distance == 0.25
        assert result.keyword_score == 0.75
        assert result.rrf_score == 0.0322

    def test_metadata_defaults_to_empty_dict(self) -> None:
        result = SearchResult(id="r1", document="text")

        assert result.metadata == {}
        assert result.metadata is not SearchResult(id="r2", document="t").metadata


class TestRetrievalError:
    def test_is_an_exception(self) -> None:
        assert issubclass(RetrievalError, Exception)

    def test_carries_the_failure_message(self) -> None:
        error = RetrievalError("Hybrid search failed on documents")

        assert str(error) == "Hybrid search failed on documents"

    def test_is_raised_and_caught_as_retrieval_error(self) -> None:
        with pytest.raises(RetrievalError, match="failed"):
            raise RetrievalError("search failed")
