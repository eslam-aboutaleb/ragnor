"""Tests for retrieval score semantics and non-interchangeability."""

from __future__ import annotations


from ragnor.retrieval.scores import RawScore, ScoreKind
from ragnor.retrieval.types import RetrievedDocument


def test_score_kinds_and_ordering_semantics() -> None:
    d = RawScore.distance(0.15, metric="l2")
    assert d.kind == ScoreKind.DISTANCE
    assert d.value == 0.15
    assert d.lower_is_better is True
    assert d.metric == "l2"

    s = RawScore.similarity(0.85, metric="cosine")
    assert s.kind == ScoreKind.SIMILARITY
    assert s.value == 0.85
    assert s.lower_is_better is False
    assert s.metric == "cosine"

    r = RawScore.rrf(0.016)
    assert r.kind == ScoreKind.RRF
    assert r.lower_is_better is False

    lex = RawScore.lexical(12.4, metric="bm25")
    assert lex.kind == ScoreKind.LEXICAL
    assert lex.value == 12.4
    assert lex.lower_is_better is False

    rerank = RawScore.reranker(0.98, model="cohere/rerank-v3.5")
    assert rerank.kind == ScoreKind.RERANKER
    assert rerank.value == 0.98
    assert rerank.lower_is_better is False


def test_scores_are_not_interchangeable() -> None:
    # A distance of 0.2 is very good (closer), but a similarity of 0.2 is very poor
    dist = RawScore.distance(0.2)
    sim = RawScore.similarity(0.2)

    assert dist.lower_is_better != sim.lower_is_better
    assert dist.kind != sim.kind


def test_retrieved_document_tracks_multiple_score_kinds() -> None:
    doc = RetrievedDocument(
        id="chunk_42",
        document="Some text",
        distance=0.12,
        keyword_score=5.0,
        rrf_score=0.03,
        reranker_score=0.91,
    )

    # Document auto-populates all 4 scores
    assert doc.scores[ScoreKind.DISTANCE].value == 0.12
    assert doc.scores[ScoreKind.DISTANCE].lower_is_better is True

    assert doc.scores[ScoreKind.LEXICAL].value == 5.0
    assert doc.scores[ScoreKind.LEXICAL].lower_is_better is False

    assert doc.scores[ScoreKind.RRF].value == 0.03
    assert doc.scores[ScoreKind.RERANKER].value == 0.91
