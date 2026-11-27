# Structure-aware chunking (EX-06)

`StructuredDocumentChunker` chunks reading-ordered regions (headings, paragraphs,
tables, figures, formulas, captions) instead of raw text. It complements
`MarkdownSectionChunker` and the token-aware chunker from #27; it does not add a
tokenizer. Token counting uses the existing `Tokenizer` protocol.

## What it guarantees

- A table with its caption, a figure with its caption, and a formula are atomic
  while they fit the budget.
- An oversized table is split only on complete row boundaries. Every part repeats
  the section path, caption and header, and carries a continuation marker
  (`continuation` metadata plus `(part n of m)` text).
- A single row (with header and caption) that cannot fit is never split. Strict
  mode raises `OversizeUnitError`. Non-strict mode emits it alone with
  `exceeds_budget: True`.
- Headings H1-H6 form a nested `section_path`. A heading directly followed by a
  deeper heading is represented through its child's path.
- Strict mode requires an exact tokenizer and hard-limits every chunk. Without
  strict mode the default `ApproximateTokenizer` is allowed, and every chunk says
  `token_count_exact: False`. An estimate is never presented as a guarantee.

## Provenance

Each chunk carries two separate lists:

- `source_spans`: the exact sources of the chunk body. One entry per region, or
  per table row (`kind: table_row`, `row`, `page`, `bbox`, optional per-cell
  boxes). Spans from several regions or pages are never merged into one box.
- `context_spans`: repeated retrieval context (section headings, table header,
  caption). These are not extra sources.

Other metadata: `section_path`, `pages`, `retrieval_context`,
`retrieval_projection` (tables), `storage_refs` (HTML/JSON/crop references only,
never image bytes), `access` (tenant/authorization mapping copied unchanged),
`token_count`, `token_count_exact`.

## Identity and re-indexing

Chunk ids are `sdc-<sha256 prefix>` of the source fingerprint, document id,
chunker version, chunker config (budget, tokenizer, strict flag) and the ordered
boundary (region ids, row ranges, span keys). The same input gives the same ids.
Changing any option, the tokenizer, the chunker version or the source
fingerprint changes the ids, and `build_snapshot()` changes
`chunker_config_hash`, so the existing manifest/snapshot flow re-ingests and
retires the earlier generation as it does for other chunkers. Access metadata is
not part of identity.

## Usage

```python
from ragnor.chunking import StructuredDocumentChunker, structured_from_pages
from ragnor.evidence.tokenizer import ExactTokenizerAdapter

tokenizer = ExactTokenizerAdapter(my_exact_count_fn, name="my-model")
chunker = StructuredDocumentChunker(tokenizer, max_tokens=512, strict=True)

# Basic documents (Markdown text): works with today's Chunker.chunk(text) API.
chunks = chunker.chunk(markdown_text)

# Structured PDFs: adapt EX-01 page outcomes (tables keep their cell boxes).
document = structured_from_pages(pages, document_id="doc-1", fingerprint=evidence_hash)
chunks = chunker.chunk_document(document)               # or token_budget=/tokenizer=
```

See `examples/structured_chunking.py`.

## Migration and fallback

1. Keep `MarkdownSectionChunker` as the fallback. `StructuredDocumentChunker.chunk(text)`
   parses Markdown (headings, pipe tables, paragraphs, fenced code) and is
   registered as `structured` in `chunker_registry`.
2. Switching chunkers changes `chunker_version` and config, so affected sources
   re-ingest once. Plan this as a re-index.
3. Structured regions and `source_spans` need a structured input. Until the typed
   document model of #61 (EX-03) lands, build `StructuredDocument` directly or
   with `structured_from_pages`. An EX-03 document is adapted into
   `StructuredDocument` (the input types are intentionally small).

## Benchmark and known gaps

`python -m benchmarks.structured_chunking` compares chunkers on a synthetic report
(chunk-level proxy, offline): header retention, rows split, and exact source-span
coverage. Trade-off: more, smaller chunks, because header and caption repeat.

Honest gaps: this does not report EX-02 retrieval Recall@K or end-to-end citation
accuracy, since that needs an embedding model and the EX-02 harness. The ingestion
pipeline still calls `chunker.chunk(text)`, so passing a `StructuredDocument`
through the pipeline automatically needs EX-03.

## Real-corpus benchmark (lexical retrieval)

`python -m benchmarks.structured_chunking_real --budget N` runs the real
`PdfPlumberExtractor` over the EX-02 fixture PDFs, chunks each page with
`SlidingWindowChunker` (page text as ingestion produces it today, tables inline
as HTML) and with `StructuredDocumentChunker`, indexes every chunk in one BM25
index and asks 125 table-cell questions derived from the corpus annotations
(row label plus column header). A chunk answers a question when row label,
column header and value are all in it, on the right page. Contract probes, no
embedding model, so this is lexical retrieval only, on a small synthetic corpus.

| Budget (words) | Sliding Recall@5 | Structured Recall@5 | Sliding / structured chunks |
|---|---|---|---|
| 20 | 0.160 | 0.472 | 77 / 86 |
| 40 | 0.552 | 0.736 | 39 / 53 |
| 80 | 1.000 | 1.000 | 25 / 28 |
| 200 | 1.000 | 1.000 | 24 / 20 |

Structured chunking helps when the budget is smaller than a table, because it
repeats the header and caption on every partition. At budgets of 80 words or
more every table fits whole in these documents and the two tie (0.992 Recall@1
both, MRR 0.996 vs 0.995). At small budgets structured still cannot answer 33
of 125 questions at 40 words and 66 at 20, because a row plus its header does
not always fit the budget. It costs more chunks at small budgets (53 vs 39 at
40 words). This corpus has few and short tables, so the tie at larger budgets
does not show the benefit on long real-world tables.
