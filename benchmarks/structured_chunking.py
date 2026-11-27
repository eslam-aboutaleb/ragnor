"""Chunk-level citation benchmark: structured vs Markdown/sliding-window chunking.

Deterministic and offline. For a synthetic report with large tables it measures,
per chunker: header retention (chunks containing table rows that also carry the
table header), rows split across chunks, and row attribution accuracy (a
queried row is found in a chunk whose exact source span names that row).

This is a chunk-level proxy. It is NOT the EX-02 retrieval benchmark: there is
no embedding model, so Recall@K is not measured here (see docs/structured-chunking.md).
Run: python -m benchmarks.structured_chunking
"""

from __future__ import annotations

import json

from ragnor.chunking import StructuredDocumentChunker
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.evidence.tokenizer import ExactTokenizerAdapter

BUDGET = 60
TOKENIZER = ExactTokenizerAdapter(lambda t: len(t.split()), name="whitespace-words")


def build_markdown(tables: int = 4, rows: int = 60) -> str:
    parts = ["# Annual report"]
    for t in range(tables):
        body = "\n".join(f"| item-{t}-{r} | {r * 3} | {r}.25 |" for r in range(rows))
        head = "| Item | Qty | Price |\n| --- | --- | --- |"
        parts.append(f"## Table {t}\n\nTable {t}: stock levels.\n\n{head}\n{body}\n")
    return "\n".join(parts)


def evaluate(chunks, *, tables: int, rows: int, structured: bool) -> dict[str, float]:
    row_chunks = [c for c in chunks if "| item-" in c.text]
    with_header = [c for c in row_chunks if "| Item | Qty | Price |" in c.text]
    split_rows = 0
    found = 0
    total = tables * rows
    for t in range(tables):
        for r in range(rows):
            needle = f"| item-{t}-{r} | {r * 3} | {r}.25 |"
            hits = [c for c in chunks if needle in c.text]
            if not hits:
                split_rows += 1  # the full row never appears intact in any chunk
                continue
            if structured and any(
                s.get("row") == r and s.get("kind") == "table_row"
                for c in hits
                for s in c.metadata.get("source_spans", [])
            ):
                found += 1
    return {
        "chunks": len(chunks),
        "row_chunks_with_header": round(len(with_header) / max(1, len(row_chunks)), 3),
        "rows_split_across_chunks": split_rows,
        "rows_with_exact_source_span": round(found / total, 3),
    }


def main() -> None:
    tables, rows = 4, 60
    md = build_markdown(tables, rows)
    old = MarkdownSectionChunker(SlidingWindowChunker(chunk_size=BUDGET, overlap=5)).chunk(md)
    new = StructuredDocumentChunker(TOKENIZER, max_tokens=BUDGET, strict=True).chunk(md)
    report = {
        "budget_tokens": BUDGET,
        "markdown_sliding_window": evaluate(old, tables=tables, rows=rows, structured=False),
        "structured": evaluate(new, tables=tables, rows=rows, structured=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
