"""Tests for ragnor.retrieval.

Covers the :class:`HybridRetriever` orchestrator:
the distance-threshold fallback, filter passthrough,
the ``SearchResult`` conversion, and -- the key
plan-04 assertion -- retrieval against the
:class:`ragnor.stores.InMemoryVectorStore`, which
proves the retriever has no pgvector dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any
from unittest.mock import patch

import pytest

from ragnor.retrieval import (
    HybridRetriever,
    RetrieverSpec,
    as_dicts,
)
from ragnor.retrieval.strategies import (
    DenseRetrieval,
    HybridRetrieval,
    KeywordRetrieval,
    RetrievalStrategy,
    Reranker,
)
from ragnor.stores.base import (
    DocumentStore,
    LexicalIndex,
    VectorIndex,
)
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.types import RetrievalError, SearchResult


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self) -> None:
        self.embedding_model = "text-embedding-3-small"
        self.rag_distance_threshold = 1.3
        self.vector_store_provider = "memory"
        self.openai_api_key = "test-key"
        self.openai_api_base = ""
        self.embedding_drain_interval_seconds = 5.0
        self.embedding_drain_batch_size = 10
        self.job_stale_after_seconds = 300.0
        self.worker_id = "test-worker"
        self.ingest_on_startup = True
        self.database_url = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"


class _HashingEmbedder:
    """Deterministic embedder: tokens hash into vector dimensions.

    Shared tokens land in shared dimensions, so texts that
    share vocabulary get close L2 distance -- bag-of-words
    similarity without any network calls.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class _CapturingStore:
    """Store that records the arguments hybrid_search received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def hybrid_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "embedding": embedding,
                "n_results": n_results,
                "threshold": threshold,
                "text_field": text_field,
                "metadata_fields": metadata_fields,
                "filters": filters,
            }
        )
        return [
            {
                "id": "1",
                "document": "captured document",
                "metadata": {"section": "Captured"},
                "distance": 0.5,
                "keyword_score": None,
                "_rrf_score": 0.8,
            }
        ]


def _spec(**kwargs: Any) -> RetrieverSpec:
    return RetrieverSpec(table_name="documents", **kwargs)


@pytest.mark.asyncio
async def test_retrieve_returns_search_results() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        results = await retriever.retrieve("water damage")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].document == "captured document"
    assert results[0].distance == 0.5
    assert results[0].rrf_score == 0.8


@pytest.mark.asyncio
async def test_distance_threshold_none_falls_back_to_settings() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query")

    assert store.calls[0]["threshold"] == 1.3


@pytest.mark.asyncio
async def test_explicit_distance_threshold_is_passed_through() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query", distance_threshold=0.7)

    assert store.calls[0]["threshold"] == 0.7


@pytest.mark.asyncio
async def test_filters_and_n_results_are_passed_through() -> None:
    store = _CapturingStore()
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        await retriever.retrieve("query", n_results=3, owner_id="owner-1", status="open")

    assert store.calls[0]["n_results"] == 3
    assert store.calls[0]["filters"] == {"owner_id": "owner-1", "status": "open"}


@pytest.mark.asyncio
async def test_retriever_retrieves_from_in_memory_store() -> None:
    """The retriever works against the in-memory store -- no pgvector.

    Seeds three documents and retrieves by keyword (token overlap)
    and by vector similarity (shared-token L2 distance), proving
    the orchestrator carries no pgvector dependency.
    """
    embedder = _HashingEmbedder(dim=8)
    store = InMemoryVectorStore(
        table_name="documents",
        id_field="id",
        embedding_dim=8,
        embedding_fn=embedder,
    )
    await store.upsert(
        [
            {
                "id": "1",
                "text": "water damage coverage",
                "metadata": {"section": "Water"},
                "embedding": embedder.embed("water damage coverage"),
            },
            {
                "id": "2",
                "text": "fire damage coverage",
                "metadata": {"section": "Fire"},
                "embedding": embedder.embed("fire damage coverage"),
            },
            {
                "id": "3",
                "text": "the deductible amount",
                "metadata": {"section": "Deductible"},
                "embedding": embedder.embed("the deductible amount"),
            },
        ]
    )

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(
            spec=_spec(metadata_fields=["section"]),
            settings=_FakeSettings(),
        )

        # Keyword + vector: "water damage" shares both tokens with
        # document 1 and only "damage" with document 2.
        results = await retriever.retrieve("water damage", n_results=5)
        assert results, "expected results for a seeded query"
        assert all(isinstance(result, SearchResult) for result in results)
        assert results[0].document == "water damage coverage"
        assert results[0].metadata == {"section": "Water"}

        # Keyword-only: "water" appears in document 1 alone.
        keyword_results = await retriever.retrieve("water", n_results=5)
        assert keyword_results[0].document == "water damage coverage"

        # Vector similarity: the query embedding of "water damage
        # coverage" is closest to document 1's embedding.
        vector_results = await retriever.retrieve("water damage coverage", n_results=5)
        assert vector_results[0].document == "water damage coverage"

        # Dict consumers keep working through as_dicts().
        as_dict = as_dicts(results)
        assert as_dict[0]["id"] == "1"
        assert as_dict[0]["document"] == "water damage coverage"
        assert as_dict[0]["metadata"] == {"section": "Water"}
        assert as_dict[0]["distance"] is not None
        assert as_dict[0]["_rrf_score"] > 0.0


@pytest.mark.asyncio
async def test_retrieve_propagates_store_errors() -> None:
    store = _CapturingStore()

    async def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("search failed")

    store.hybrid_search = _fail  # type: ignore[method-assign]
    embedder = _HashingEmbedder()

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ),
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        with pytest.raises(RuntimeError, match="search failed"):
            await retriever.retrieve("query")


def test_as_dicts_maps_every_field() -> None:
    results = [
        SearchResult(
            id="1",
            document="doc",
            metadata={"section": "S"},
            distance=0.5,
            keyword_score=0.25,
            rrf_score=0.8,
        ),
        SearchResult(id="2", document="doc2"),
    ]
    dicts = as_dicts(results)
    assert dicts[0] == {
        "id": "1",
        "document": "doc",
        "metadata": {"section": "S"},
        "distance": 0.5,
        "keyword_score": 0.25,
        "_rrf_score": 0.8,
    }
    assert dicts[1]["distance"] is None
    assert dicts[1]["keyword_score"] is None
    assert dicts[1]["_rrf_score"] == 0.0


def test_retriever_spec_validates_identifiers() -> None:
    with pytest.raises(ValueError, match="Unsafe table_name"):
        RetrieverSpec(table_name="bad-name")
    with pytest.raises(ValueError, match="Unsafe metadata_field"):
        RetrieverSpec(table_name="documents", metadata_fields=["bad-field"])
    spec = _spec(id_field="id", text_field="text", metadata_fields=["section"])
    assert spec.table_name == "documents"


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


class _FakeVectorIndex:
    """Minimal VectorIndex double recording its calls."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[dict[str, Any]] = []

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "embedding": embedding,
                "n_results": n_results,
                "threshold": threshold,
                "text_field": text_field,
                "metadata_fields": metadata_fields,
                "filters": filters,
            }
        )
        return self.rows


class _FakeLexicalIndex:
    """Minimal LexicalIndex double recording its calls."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls: list[dict[str, Any]] = []

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "n_results": n_results,
                "text_field": text_field,
                "metadata_fields": metadata_fields,
                "filters": filters,
            }
        )
        return self.rows


class _FailingVectorIndex:
    """VectorIndex whose vector_search always raises."""

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("index down")


class _FailingLexicalIndex:
    """LexicalIndex whose keyword_search always raises."""

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("index down")


class _RecordingReranker:
    """Reranker that records the query and reverses the results."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return list(reversed(results))


class _VectorOnlyCustomBackend:
    """A custom backend implementing only VectorIndex + DocumentStore.

    It deliberately implements no keyword search -- a host
    with a vector-only engine composes it with DenseRetrieval.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [
            {
                "id": "1",
                "document": "custom doc",
                "metadata": {"section": "Custom"},
                "distance": 0.25,
                "keyword_score": None,
            }
        ]

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        return self.rows

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        return len(self.rows)

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: Any = None,
        session: Any = None,
    ) -> int:
        return 0

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: Any = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {"id": row["id"], "document": row["document"], "metadata": row["metadata"]}
            for row in self.rows
        ]


def test_strategies_satisfy_the_strategy_protocol() -> None:
    vector_index = _FakeVectorIndex()
    lexical_index = _FakeLexicalIndex()
    embedder = _HashingEmbedder()

    assert isinstance(DenseRetrieval(vector_index, embedder), RetrievalStrategy)
    assert isinstance(KeywordRetrieval(lexical_index), RetrievalStrategy)
    assert isinstance(
        HybridRetrieval(vector_index, lexical_index, embedder),
        RetrievalStrategy,
    )
    assert isinstance(_RecordingReranker(), Reranker)


@pytest.mark.asyncio
async def test_dense_retrieval_embeds_and_searches() -> None:
    rows = [
        {
            "id": "1",
            "document": "doc",
            "metadata": {},
            "distance": 0.25,
            "keyword_score": None,
        }
    ]
    vector_index = _FakeVectorIndex(rows)
    embedder = _HashingEmbedder()

    strategy = DenseRetrieval(vector_index, embedder)
    results = await strategy.retrieve(
        "water damage", n_results=3, threshold=1.3, owner_id="owner-1"
    )

    assert results == rows
    assert vector_index.calls[0]["embedding"] == embedder.embed("water damage")
    assert vector_index.calls[0]["n_results"] == 3
    assert vector_index.calls[0]["threshold"] == 1.3
    assert vector_index.calls[0]["filters"] == {"owner_id": "owner-1"}


@pytest.mark.asyncio
async def test_dense_retrieval_requires_threshold() -> None:
    strategy = DenseRetrieval(_FakeVectorIndex(), _HashingEmbedder())
    with pytest.raises(ValueError, match="threshold"):
        await strategy.retrieve("query")


@pytest.mark.asyncio
async def test_keyword_retrieval_needs_no_embedder() -> None:
    """Keyword-only retrieval has no embedding dependency."""
    rows = [
        {
            "id": "1",
            "document": "doc",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.75,
        }
    ]
    lexical_index = _FakeLexicalIndex(rows)

    # No embedder is constructed -- the strategy takes only the
    # lexical index, so a keyword-only application (e.g.
    # low-latency game NPC retrieval) needs no embedding-provider
    # configuration at all.
    strategy = KeywordRetrieval(lexical_index)
    results = await strategy.retrieve("refund policy", n_results=5, tenant="a")

    assert results == rows
    assert lexical_index.calls[0]["query"] == "refund policy"
    assert lexical_index.calls[0]["n_results"] == 5
    assert lexical_index.calls[0]["filters"] == {"tenant": "a"}


@pytest.mark.asyncio
async def test_hybrid_retrieval_runs_both_stages_and_fuses() -> None:
    vector_rows = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {},
            "distance": 0.1,
            "keyword_score": None,
        }
    ]
    keyword_rows = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.9,
        },
        {
            "id": "b",
            "document": "doc b",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.5,
        },
    ]
    vector_index = _FakeVectorIndex(vector_rows)
    lexical_index = _FakeLexicalIndex(keyword_rows)
    embedder = _HashingEmbedder()

    strategy = HybridRetrieval(vector_index, lexical_index, embedder)
    results = await strategy.retrieve(
        "water damage", n_results=5, threshold=1.3, owner_id="owner-1"
    )

    # "a" is found by both stages and outranks the keyword-only "b".
    assert [row["id"] for row in results] == ["a", "b"]
    assert results[0]["_rrf_score"] > results[1]["_rrf_score"]
    assert results[0]["distance"] == 0.1
    assert results[0]["keyword_score"] == 0.9
    assert results[1]["distance"] is None
    assert results[1]["keyword_score"] == 0.5
    # Both stages received the query, the embedder's vector, and the filters.
    assert vector_index.calls[0]["embedding"] == embedder.embed("water damage")
    assert vector_index.calls[0]["threshold"] == 1.3
    assert vector_index.calls[0]["filters"] == {"owner_id": "owner-1"}
    assert lexical_index.calls[0]["query"] == "water damage"
    assert lexical_index.calls[0]["filters"] == {"owner_id": "owner-1"}


@pytest.mark.asyncio
async def test_hybrid_retrieval_applies_reranker() -> None:
    vector_rows = [
        {
            "id": "a",
            "document": "doc a",
            "metadata": {},
            "distance": 0.1,
            "keyword_score": None,
        }
    ]
    keyword_rows = [
        {
            "id": "b",
            "document": "doc b",
            "metadata": {},
            "distance": None,
            "keyword_score": 0.5,
        }
    ]
    reranker = _RecordingReranker()
    strategy = HybridRetrieval(
        _FakeVectorIndex(vector_rows),
        _FakeLexicalIndex(keyword_rows),
        _HashingEmbedder(),
        reranker=reranker,
    )

    results = await strategy.retrieve("water damage", n_results=5, threshold=1.3)

    # The reranker reversed the fused results.
    assert [row["id"] for row in results] == ["b", "a"]
    assert reranker.queries == ["water damage"]


@pytest.mark.asyncio
async def test_hybrid_retrieval_over_two_different_capability_objects() -> None:
    """The two stages may live in different systems."""
    vector_index = _FakeVectorIndex(
        [
            {
                "id": "pg-1",
                "document": "pgvector doc",
                "metadata": {},
                "distance": 0.1,
                "keyword_score": None,
            }
        ]
    )
    lexical_index = _FakeLexicalIndex(
        [
            {
                "id": "ext-1",
                "document": "external engine doc",
                "metadata": {},
                "distance": None,
                "keyword_score": 0.8,
            }
        ]
    )

    strategy = HybridRetrieval(
        vector_index=vector_index,
        lexical_index=lexical_index,
        embedder=_HashingEmbedder(),
    )
    results = await strategy.retrieve("water damage", n_results=5, threshold=1.3)

    # A pgvector VectorIndex plus an external engine adapted to
    # LexicalIndex: both stages contribute, fused by RRF.
    assert {row["id"] for row in results} == {"pg-1", "ext-1"}
    assert results[0]["_rrf_score"] == pytest.approx(1.0 / 61)
    assert results[1]["_rrf_score"] == pytest.approx(1.0 / 61)


@pytest.mark.asyncio
async def test_vector_only_custom_backend_with_dense_retrieval() -> None:
    """A custom backend implementing only VectorIndex + DocumentStore.

    It satisfies both Protocols structurally and composes with
    DenseRetrieval -- no keyword stage exists, so hybrid retrieval
    is not an option, but dense retrieval is.
    """
    backend = _VectorOnlyCustomBackend()
    assert isinstance(backend, VectorIndex)
    assert isinstance(backend, DocumentStore)
    assert not isinstance(backend, LexicalIndex)

    strategy = DenseRetrieval(backend, _HashingEmbedder())
    results = await strategy.retrieve("water damage", n_results=5, threshold=1.3)

    assert [row["id"] for row in results] == ["1"]
    assert results[0]["document"] == "custom doc"
    assert results[0]["metadata"] == {"section": "Custom"}


@pytest.mark.asyncio
async def test_strategy_failures_are_swallowed_by_default() -> None:
    dense = DenseRetrieval(_FailingVectorIndex(), _HashingEmbedder())
    assert await dense.retrieve("query", n_results=5, threshold=1.3) == []

    keyword = KeywordRetrieval(_FailingLexicalIndex())
    assert await keyword.retrieve("query", n_results=5) == []

    hybrid = HybridRetrieval(
        _FailingVectorIndex(),
        _FakeLexicalIndex(),
        _HashingEmbedder(),
    )
    assert await hybrid.retrieve("query", n_results=5, threshold=1.3) == []


@pytest.mark.asyncio
async def test_strategy_failures_raise_under_strict() -> None:
    dense = DenseRetrieval(_FailingVectorIndex(), _HashingEmbedder())
    with pytest.raises(RetrievalError) as caught:
        await dense.retrieve("query", n_results=5, threshold=1.3, strict=True)

    assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
    assert str(caught.value.__cause__) == "index down"
    assert "index down" not in str(caught.value)

    keyword = KeywordRetrieval(_FailingLexicalIndex())
    with pytest.raises(RetrievalError) as caught:
        await keyword.retrieve("query", n_results=5, strict=True)

    assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
    assert str(caught.value.__cause__) == "index down"
    assert "index down" not in str(caught.value)

    hybrid = HybridRetrieval(
        _FakeVectorIndex(),
        _FailingLexicalIndex(),
        _HashingEmbedder(),
    )
    with pytest.raises(RetrievalError) as caught:
        await hybrid.retrieve("query", n_results=5, threshold=1.3, strict=True)

    assert caught.value.code == "ERR_BACKEND_UNAVAILABLE"
    assert str(caught.value.__cause__) == "index down"
    assert "index down" not in str(caught.value)


@pytest.mark.asyncio
async def test_retriever_with_injected_strategy_builds_nothing() -> None:
    """Strategy injection bypasses eager embedder/store construction.

    The settings carry no usable embedding provider, so building
    the embedder would fail -- with a strategy injected, the
    constructor builds nothing and retrieval delegates to it.
    """
    lexical_index = _FakeLexicalIndex(
        [
            {
                "id": "1",
                "document": "keyword doc",
                "metadata": {},
                "distance": None,
                "keyword_score": 0.9,
            }
        ]
    )
    strategy = KeywordRetrieval(lexical_index)

    class _NoEmbeddingSettings(_FakeSettings):
        embedding_provider = "nonexistent-provider"

    retriever = HybridRetriever(
        spec=_spec(),
        settings=_NoEmbeddingSettings(),
        strategy=strategy,
    )

    results = await retriever.retrieve("refund policy", n_results=5)

    assert results[0].document == "keyword doc"
    assert lexical_index.calls[0]["query"] == "refund policy"
    assert lexical_index.calls[0]["n_results"] == 5


@pytest.mark.asyncio
async def test_retriever_passes_strict_through_to_strategy() -> None:
    captured: dict[str, Any] = {}

    class _StrictCapturingStrategy:
        async def retrieve(
            self,
            query: str,
            n_results: int = 5,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return [
                {
                    "id": "1",
                    "document": "doc",
                    "metadata": {},
                    "distance": None,
                    "keyword_score": None,
                }
            ]

    retriever = HybridRetriever(
        spec=_spec(),
        settings=_FakeSettings(),
        strategy=_StrictCapturingStrategy(),
    )
    await retriever.retrieve("query", n_results=5, strict=True, owner_id="o1")

    assert captured["strict"] is True
    assert captured["threshold"] == _FakeSettings().rag_distance_threshold
    assert captured["text_field"] == "text"
    assert captured["metadata_fields"] == []
    assert captured["owner_id"] == "o1"


@pytest.mark.asyncio
async def test_retriever_default_path_still_builds_embedder_and_store() -> None:
    """Without a strategy, the constructor builds the embedder and store."""
    embedder = _HashingEmbedder()
    store = InMemoryVectorStore(
        table_name="documents",
        id_field="id",
        embedding_dim=8,
        embedding_fn=embedder,
    )

    with (
        patch(
            "ragnor.retrieval.retriever.get_embedding_function",
            return_value=embedder,
        ),
        patch(
            "ragnor.retrieval.retriever.get_vector_store",
            return_value=store,
        ) as get_store,
    ):
        retriever = HybridRetriever(spec=_spec(), settings=_FakeSettings())
        assert retriever._embed_fn is embedder
        assert retriever._store is store
        assert get_store.called

    await store.upsert(
        [
            {
                "id": "1",
                "text": "water damage coverage",
                "embedding": embedder.embed("water damage coverage"),
            }
        ]
    )
    results = await retriever.retrieve("water damage", n_results=5)
    assert results[0].document == "water damage coverage"
