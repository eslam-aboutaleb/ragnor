"""Real tokenizer integration, only in the installed tiktoken-extra CI job."""

from __future__ import annotations

import pytest

from ragnor.chunking import TokenAwareChunker
from ragnor.evidence.tiktoken_adapter import TiktokenTokenizer


def test_real_model_tokenizer_caps_every_chunk_in_multilingual_text() -> None:
    tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")
    text = '# Policy\n\nArabic: مرحبا بكم. English refund rules. 中文文档和代码: {"a": 1}\n' * 15
    chunker = TokenAwareChunker(tokenizer, max_tokens=40, overlap_tokens=7)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(tokenizer.count_tokens(chunk.text) <= 40 for chunk in chunks)
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[-1].metadata["end_offset"] == len(text)


def test_unknown_tiktoken_encoding_is_not_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown"):
        TiktokenTokenizer(encoding_name="unknown-demo-encoding")


def test_model_and_encoding_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TiktokenTokenizer(encoding_name="cl100k_base", model="gpt-4o")
