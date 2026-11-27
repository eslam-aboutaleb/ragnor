"""Real-corpus chunking benchmark: real extractor, real chunkers, real BM25."""

from __future__ import annotations

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("rank_bm25")

from benchmarks import structured_chunking_real as bench  # noqa: E402


@pytest.fixture(scope="module")
def pages():
    return bench.corpus_pages()[0]


@pytest.fixture(scope="module")
def questions():
    return bench.load_questions()


def test_questions_come_from_annotated_tables(questions):
    assert len(questions) >= 100
    assert all(len(q["need"]) == 3 and q["query"] for q in questions)


def test_every_question_is_answerable_by_some_chunk_when_chunks_are_large(pages, questions):
    for strategy in ("sliding", "structured"):
        result = bench.evaluate(bench.build_chunks(pages, 200, strategy), questions)
        assert result["unanswerable_in_any_chunk"] == 0, strategy
        assert result["recall@5"] >= 0.99, strategy


def test_structured_chunking_keeps_more_table_answers_at_small_budgets(pages, questions):
    sliding = bench.evaluate(bench.build_chunks(pages, 40, "sliding"), questions)
    structured = bench.evaluate(bench.build_chunks(pages, 40, "structured"), questions)
    assert structured["recall@5"] > sliding["recall@5"]
    assert structured["unanswerable_in_any_chunk"] < sliding["unanswerable_in_any_chunk"]


def test_answer_check_requires_label_header_and_value_in_one_chunk():
    q = {"doc": "d", "page": 1, "query": "x", "need": ["Total due", "Amount", "1,250.00"]}
    ok = {"doc": "d", "pages": {1}, "text": "Amount Total due 1,250.00"}
    assert bench.answers(ok, q)
    assert not bench.answers({**ok, "text": "Total due 1,250.00"}, q)
    assert not bench.answers({**ok, "pages": {2}}, q)
    assert not bench.answers({**ok, "doc": "other"}, q)
