"""Tests for typed retrieval requests and result wrappers."""

from __future__ import annotations


from ragnor.retrieval.scores import ScoreKind
from ragnor.retrieval.types import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedDocument,
    ScopeViolationError,
)
from ragnor.types import SearchResult


def test_retrieval_request_creation_and_defaults() -> None:
    req = RetrievalRequest(query="search terms")
    assert req.query == "search terms"
    assert req.final_k == 5
    assert req.candidate_k is None
    assert req.get_candidate_k() == 5
    assert req.filters == {}
    assert req.mandatory_capabilities == set()
    assert req.isolation_scope is None
    assert req.scope_field == "tenant_id"


def test_retrieval_request_candidate_k_effective() -> None:
    req = RetrievalRequest(query="test", final_k=3, candidate_k=20)
    assert req.get_candidate_k() == 20

    # candidate_k smaller than final_k gets clamped to final_k
    req2 = RetrievalRequest(query="test", final_k=10, candidate_k=4)
    assert req2.get_candidate_k() == 10


def test_retrieval_request_effective_filters() -> None:
    req = RetrievalRequest(
        query="test",
        filters={"category": "tech"},
        isolation_scope="org_123",
        scope_field="tenant_id",
        expected_generation=4,
    )
    effective = req.effective_filters()
    assert effective == {
        "category": "tech",
        "tenant_id": "org_123",
        "generation": 4,
    }


def test_retrieval_request_to_dict_and_from_query() -> None:
    req = RetrievalRequest.from_query(
        "sample query",
        n_results=10,
        candidate_k=30,
        threshold=0.8,
        strict=True,
        isolation_scope="tenant_a",
        category="news",
    )
    assert req.query == "sample query"
    assert req.final_k == 10
    assert req.candidate_k == 30
    assert req.threshold == 0.8
    assert req.strict is True
    assert req.isolation_scope == "tenant_a"
    assert req.filters == {"category": "news"}

    as_dict = req.to_dict()
    assert as_dict["query"] == "sample query"
    assert as_dict["final_k"] == 10
    assert as_dict["candidate_k"] == 30
    assert as_dict["threshold"] == 0.8
    assert as_dict["category"] == "news"


def test_retrieved_document_from_and_to_dict() -> None:
    legacy_row = {
        "id": "doc_1",
        "document": "Content of doc 1",
        "metadata": {"source": "manual", "tenant_id": "tenant_1"},
        "distance": 0.25,
        "keyword_score": 3.5,
        "_rrf_score": 0.032,
        "_reranker_score": 0.95,
    }

    doc = RetrievedDocument.from_dict(legacy_row)
    assert doc.id == "doc_1"
    assert doc.document == "Content of doc 1"
    assert doc.metadata == {"source": "manual", "tenant_id": "tenant_1"}
    assert doc.distance == 0.25
    assert doc.keyword_score == 3.5
    assert doc.rrf_score == 0.032
    assert doc.reranker_score == 0.95

    # Check explicit scores dictionary
    assert ScoreKind.DISTANCE in doc.scores
    assert doc.scores[ScoreKind.DISTANCE].value == 0.25
    assert doc.scores[ScoreKind.DISTANCE].lower_is_better is True

    assert ScoreKind.LEXICAL in doc.scores
    assert doc.scores[ScoreKind.LEXICAL].value == 3.5
    assert doc.scores[ScoreKind.LEXICAL].lower_is_better is False

    assert ScoreKind.RRF in doc.scores
    assert doc.scores[ScoreKind.RRF].value == 0.032

    assert ScoreKind.RERANKER in doc.scores
    assert doc.scores[ScoreKind.RERANKER].value == 0.95

    # Bridge back to legacy dict
    converted = doc.to_dict()
    assert converted["id"] == "doc_1"
    assert converted["document"] == "Content of doc 1"
    assert converted["distance"] == 0.25
    assert converted["keyword_score"] == 3.5
    assert converted["_rrf_score"] == 0.032
    assert converted["_reranker_score"] == 0.95

    # Bridge to SearchResult
    search_res = doc.to_search_result()
    assert isinstance(search_res, SearchResult)
    assert search_res.id == "doc_1"
    assert search_res.document == "Content of doc 1"
    assert search_res.rrf_score == 0.032


def test_retrieval_response_bidirectional_bridge() -> None:
    rows = [
        {"id": "d1", "document": "First", "metadata": {}, "distance": 0.1, "_rrf_score": 0.05},
        {"id": "d2", "document": "Second", "metadata": {}, "distance": 0.2, "_rrf_score": 0.03},
    ]
    resp = RetrievalResponse.from_dicts(rows, query="q", candidate_k=10, final_k=2)
    assert len(resp) == 2
    assert resp[0].id == "d1"
    assert resp[1].id == "d2"
    assert resp.query == "q"
    assert resp.candidate_k == 10
    assert resp.final_k == 2

    # Iteration
    ids = [item.id for item in resp]
    assert ids == ["d1", "d2"]

    # to_dicts
    back = resp.to_dicts()
    assert len(back) == 2
    assert back[0]["id"] == "d1"
    assert back[1]["id"] == "d2"


def test_scope_violation_error_inheritance() -> None:
    from ragnor.types import RetrievalError

    err = ScopeViolationError("Scope mismatch")
    assert isinstance(err, RetrievalError)
