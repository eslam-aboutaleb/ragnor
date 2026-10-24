"""SDK-07 token-aware chunking: opt-in exactness and stable re-index fingerprint."""

from __future__ import annotations

import pytest

from ragnor.chunking import SlidingWindowChunker, TokenAwareChunker
from ragnor.chunking.base import Chunker
from ragnor.evidence.tokenizer import ApproximateTokenizer, ExactTokenizerAdapter
from ragnor.ingestion.versioning import build_snapshot, snapshot_matches


class CharacterTokenizer:
    is_exact = True
    name = "model-char-v1"

    def count_tokens(self, text: str) -> int:
        return len(text)


class FakeSettings:
    embedding_model = "text-embedding-3-small"


def test_opt_in_exact_chunker_preserves_source_and_token_cap() -> None:
    text = "Arabic: مرحبا,中文 test 👋🏽\nmore code={x:2}."
    chunker = TokenAwareChunker(CharacterTokenizer(), max_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk(text)

    assert isinstance(chunker, Chunker)
    assert len(chunks) >= 2
    assert all(0 < len(x.text) <= 10 for x in chunks)
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[-1].metadata["end_offset"] == len(text)
    assert all(
        x.text == text[x.metadata["start_offset"] : x.metadata["end_offset"]] for x in chunks
    )
    assert all(x.metadata["start_offset"] < x.metadata["end_offset"] for x in chunks)
    assert all(
        nxt.metadata["start_offset"] > prev.metadata["start_offset"]
        for prev, nxt in zip(chunks, chunks[1:], strict=False)
    )
    # every character of the source is in at least one generated chunk
    covered = set().union(
        *(set(range(x.metadata["start_offset"], x.metadata["end_offset"])) for x in chunks)
    )
    assert covered == set(range(len(text)))


def test_strict_chunker_rejects_inexact_token_counter() -> None:
    with pytest.raises(ValueError, match="exact"):
        TokenAwareChunker(ApproximateTokenizer(), max_tokens=8, exact_required=True)


def test_chunker_rejects_impossible_and_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        TokenAwareChunker(CharacterTokenizer(), max_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens"):
        TokenAwareChunker(CharacterTokenizer(), max_tokens=4, overlap_tokens=4)


def test_token_chunker_empty_text_returns_empty() -> None:
    assert TokenAwareChunker(CharacterTokenizer(), max_tokens=4).chunk("   ") == []


def test_legacy_word_chunker_unchanged() -> None:
    assert [x.text for x in SlidingWindowChunker(3, 1).chunk("a b c d")] == [
        "a b c",
        "c d",
    ]


def test_snapshot_invalidates_when_tokenizer_or_chunking_changes() -> None:
    first = TokenAwareChunker(CharacterTokenizer(), max_tokens=8)
    changed_tokens = TokenAwareChunker(
        ExactTokenizerAdapter(len, name="other-model"),
        max_tokens=8,
    )
    changed_budget = TokenAwareChunker(CharacterTokenizer(), max_tokens=9)
    a = build_snapshot("same-document", FakeSettings(), first)
    b = build_snapshot("same-document", FakeSettings(), changed_tokens)
    c = build_snapshot("same-document", FakeSettings(), changed_budget)
    assert a.chunker_config_hash
    assert a.chunker_config_hash != b.chunker_config_hash
    assert a.chunker_config_hash != c.chunker_config_hash
    assert not snapshot_matches(a, b)
    assert not snapshot_matches(a, c)
    assert snapshot_matches(a, build_snapshot("same-document", FakeSettings(), first))
