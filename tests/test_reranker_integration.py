"""Tests for optional reranker integration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ragnor.retrieval.rerankers import LiteLLMReranker
from ragnor.retrieval.strategies import Reranker


def test_litellm_reranker_satisfies_reranker_protocol() -> None:
    reranker = LiteLLMReranker(model="cohere/rerank-v3.5")
    assert isinstance(reranker, Reranker)


@pytest.mark.asyncio
async def test_litellm_reranker_reranks_and_attaches_scores() -> None:
    reranker = LiteLLMReranker(model="cohere/rerank-v3.5")

    mock_litellm = MagicMock()
    # Mock litellm.rerank returning results
    mock_litellm.rerank.return_value = [
        {"index": 1, "relevance_score": 0.98},
        {"index": 0, "relevance_score": 0.45},
    ]

    candidates = [
        {"id": "doc0", "document": "First candidate", "distance": 0.2},
        {"id": "doc1", "document": "Second candidate", "distance": 0.4},
    ]

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        reranked = await reranker.rerank("my search query", candidates)

    assert len(reranked) == 2
    # Document index 1 is now ranked first
    assert reranked[0]["id"] == "doc1"
    assert reranked[0]["_reranker_score"] == 0.98
    assert reranked[1]["id"] == "doc0"
    assert reranked[1]["_reranker_score"] == 0.45


@pytest.mark.asyncio
async def test_litellm_reranker_missing_extra_raises_import_error() -> None:
    reranker = LiteLLMReranker()

    # Simulate litellm missing
    with patch.dict(sys.modules, {"litellm": None}):
        with pytest.raises(ImportError) as exc_info:
            await reranker.rerank("query", [{"id": "1", "document": "test"}])
    assert "ragnor[rerank]" in str(exc_info.value)
