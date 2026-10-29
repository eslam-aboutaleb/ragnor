"""Real SDK retrieval/evidence/citation checks on parser outputs."""

import pytest

pytest.importorskip("rank_bm25")

from benchmarks.quality_providers import QualityPage
from benchmarks.rag_probes import evaluate_rag


def corpus():
    return {
        "documents": [
            {
                "document_id": "spec",
                "source_hash": "revision-one",
                "family": "public",
                "pages": [
                    {
                        "page": 1,
                        "regions": [
                            {
                                "id": "r",
                                "category": "Table",
                                "text": "copper torque is 42 units today",
                            }
                        ],
                    }
                ],
            },
            {
                "document_id": "other",
                "source_hash": "revision-two",
                "family": "public",
                "pages": [{"page": 1, "regions": []}],
            },
        ],
        "qa_probes": [
            {"id": "q", "query": "copper torque", "answer": "42", "document_id": "spec", "page": 1}
        ],
    }


@pytest.mark.asyncio
async def test_sdk_rag_resolves_scoped_immutable_citations():
    predicted = {
        ("spec", 1): QualityPage(1, "copper torque is 42 units today", "extracted"),
        ("other", 1): QualityPage(1, "aluminum temperature is 99 degrees", "extracted"),
    }
    result = await evaluate_rag(corpus(), predicted)
    assert result["recall_at_k"] == result["mrr"] == 1
    assert result["citation_resolution_rate"] == 1
    assert result["source_answer_hit_rate"] == 1
    assert result["authorization_and_revision"]["passed"]
    assert result["authorization_and_revision"]["checks"] == 4
    assert result["authorization_and_revision"]["retrieval_cross_tenant_denied"]
    assert "copper torque" not in str(result)


@pytest.mark.asyncio
async def test_omitted_answer_is_not_an_unlabelled_success():
    result = await evaluate_rag(
        corpus(), {("spec", 1): QualityPage(1, "copper torque missing", "partial")}
    )
    assert result["recall_at_k"] == result["mrr"] == 0
    assert result["source_answer_hit_rate"] == 0
    # A resolvable citation is separate from recovering the answer.
    assert result["citation_resolution_rate"] == 1


@pytest.mark.asyncio
async def test_duplicate_curated_queries_cannot_overwrite_the_oracle():
    manifest = corpus()
    manifest["qa_probes"].append({**manifest["qa_probes"][0], "id": "another"})
    with pytest.raises(ValueError, match="unique"):
        await evaluate_rag(manifest, {})
