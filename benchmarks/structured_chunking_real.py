"""Real-corpus chunking benchmark: structured vs sliding-window, BM25 retrieval.

Pipeline, all real components: the EX-02 fixture PDFs are extracted by the real
``PdfPlumberExtractor``; the page output is chunked by (a) ``SlidingWindowChunker``
over the page text exactly as ingestion produces it today (tables inline as
HTML) and (b) ``StructuredDocumentChunker`` over ``structured_from_pages``;
every chunk of the whole corpus goes into one BM25 index (``rank_bm25``); and
table-cell questions are answered by the top-K chunks.

Questions come from the corpus annotations (label of a table row + column
header) and are contract probes, not curated human QA. A chunk *answers* a
question when it is from the same document and page and contains the row
label, the column header and the cell value together. Recall@K is the share of
questions with an answering chunk in the top K. No embedding model is used
(lexical only), so this does not measure semantic retrieval.

Run: python -m benchmarks.structured_chunking_real [--budget 40]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from rank_bm25 import BM25Plus

from ragnor.chunking import StructuredDocumentChunker
from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.chunking.structured import structured_from_pages
from ragnor.evidence.tokenizer import ExactTokenizerAdapter
from ragnor.ingestion import PdfPlumberExtractor

FIXTURES = Path(__file__).parent / "pdf_extraction" / "fixtures"
TOKENIZER = ExactTokenizerAdapter(lambda t: len(t.split()), name="whitespace-words")


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"\w[\w.,-]*", re.sub(r"<[^>]+>", " ", text).lower())


def load_questions() -> list[dict]:
    questions = []
    for line in (FIXTURES / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line)
        for table in rec.get("tables", []):
            cells = {(c["row"], c["col"]): c["text"] for c in table["cells"]}
            headers = {c: t for (r, c), t in cells.items() if r == 0 and t}
            for (r, c), value in sorted(cells.items()):
                label = cells.get((r, 0), "")
                if r == 0 or c == 0 or not value or not label or c not in headers:
                    continue
                questions.append(
                    {
                        "doc": rec["document_id"],
                        "page": rec["page"],
                        "query": f"{label} {headers[c]}",
                        "need": [label, headers[c], value],
                    }
                )
    return questions


def corpus_pages() -> tuple[dict[tuple[str, int], object], dict[str, bytes]]:
    pages, raw = {}, {}
    extractor = PdfPlumberExtractor(policy="allow_partial")
    seen = set()
    for line in (FIXTURES / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("document_id") in seen or "file" not in rec:
            continue
        seen.add(rec["document_id"])
        data = (FIXTURES / rec["file"]).read_bytes()
        try:
            outcomes = extractor.extract_outcomes(data)
        except Exception:  # corrupt/oversized fixtures are expected to fail
            continue
        raw[rec["document_id"]] = data
        for outcome in outcomes:
            pages[(rec["document_id"], outcome.page)] = outcome
    return pages, raw


def build_chunks(pages, budget: int, strategy: str) -> list[dict]:
    chunks = []
    if strategy == "sliding":
        chunker = SlidingWindowChunker(chunk_size=budget, overlap=budget // 5)
        for (doc, page), outcome in pages.items():
            chunks += [
                {"doc": doc, "pages": {page}, "text": c.text} for c in chunker.chunk(outcome.text)
            ]
        return chunks
    chunker = StructuredDocumentChunker(TOKENIZER, max_tokens=budget)
    by_doc: dict[str, list] = {}
    for (doc, _), outcome in pages.items():
        by_doc.setdefault(doc, []).append(outcome)
    for doc, outcomes in by_doc.items():
        document = structured_from_pages(outcomes, document_id=doc, fingerprint=doc)
        for c in chunker.chunk_document(document):
            spans = c.metadata.get("source_spans", [])
            pages_in = {s.get("page") for s in spans}
            ctx = c.metadata.get("retrieval_context", "")
            chunks.append({"doc": doc, "pages": pages_in, "text": f"{ctx} {c.text}"})
    return chunks


def answers(chunk: dict, q: dict) -> bool:
    if chunk["doc"] != q["doc"] or q["page"] not in chunk["pages"]:
        return False
    text = " ".join(_norm_tokens(chunk["text"]))
    return all(" ".join(_norm_tokens(n)) in text for n in q["need"])


def evaluate(chunks: list[dict], questions: list[dict], ks=(1, 3, 5)) -> dict:
    index = BM25Plus([_norm_tokens(c["text"]) for c in chunks])
    ranks, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        scores = index.get_scores(_norm_tokens(q["query"]))
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        latencies.append((time.perf_counter() - start) * 1000)
        rank = next((n for n, i in enumerate(order, 1) if answers(chunks[i], q)), None)
        ranks.append(rank)
    n = len(questions)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in ks}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    out["unanswerable_in_any_chunk"] = sum(
        1 for q in questions if not any(answers(c, q) for c in chunks)
    )
    out["chunks"] = len(chunks)
    out["query_ms_p50"] = round(statistics.median(latencies), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=40, help="words per chunk")
    parser.add_argument("--out")
    args = parser.parse_args()
    pages, _ = corpus_pages()
    questions = load_questions()
    report = {"budget_words": args.budget, "questions": len(questions), "results": {}}
    for strategy in ("sliding", "structured"):
        report["results"][strategy] = evaluate(
            build_chunks(pages, args.budget, strategy), questions
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
