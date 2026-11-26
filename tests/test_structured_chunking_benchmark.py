"""The chunk-level citation benchmark must show the documented improvement."""

from __future__ import annotations

from benchmarks import structured_chunking as bench
from ragnor.chunking import StructuredDocumentChunker
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.sliding_window import SlidingWindowChunker


def test_structured_beats_sliding_window_on_headers_rows_and_citations() -> None:
    md = bench.build_markdown(3, 40)
    old = MarkdownSectionChunker(SlidingWindowChunker(chunk_size=bench.BUDGET, overlap=5)).chunk(md)
    new = StructuredDocumentChunker(bench.TOKENIZER, max_tokens=bench.BUDGET, strict=True).chunk(md)
    a = bench.evaluate(old, tables=3, rows=40, structured=False)
    b = bench.evaluate(new, tables=3, rows=40, structured=True)
    assert b["row_chunks_with_header"] == 1.0 > a["row_chunks_with_header"]
    assert b["rows_split_across_chunks"] == 0
    assert b["rows_with_exact_source_span"] == 1.0 and a["rows_with_exact_source_span"] == 0.0
    # documented trade-off: more, smaller chunks because header/caption repeat
    assert b["chunks"] > a["chunks"]
