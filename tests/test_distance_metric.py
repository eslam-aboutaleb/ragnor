"""Tests for the selectable distance metric.

Covers metric resolution (an explicit argument beats the host
settings, the settings beat the default, values are normalized
case-insensitively, unknown values raise), the cosine math
helpers (``cosine_distance == 1 - cosine_similarity`` with
hand-computed fixtures, the zero-vector guard, the length check),
``normalize``, and the backends that support (or refuse) the
cosine metric.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from ragit.config import RagSettings
from ragit.metrics import (
    DistanceMetric,
    cosine_distance,
    cosine_similarity,
    normalize,
)
from ragit.registry import Registry
from ragit.stores import (
    DEFAULT_DISTANCE_METRIC,
    SUPPORTED_DISTANCE_METRICS,
    VectorStore,
    get_vector_store,
    resolve_distance_metric,
)
from ragit.stores.chroma import ChromaStore
from ragit.stores.memory import InMemoryVectorStore
from ragit.stores.qdrant import QdrantStore
from ragit.stores.store_factory import RegistryVectorStoreFactory


class _Settings:
    """Settings double carrying an optional metric field."""

    def __init__(self, metric: str | None = None) -> None:
        if metric is not None:
            self.rag_distance_metric = metric


class _NoneMetricSettings:
    """Settings double declaring the metric field as ``None``."""

    rag_distance_metric: str | None = None


class _ProtocolSettings:
    """The smallest object that structurally satisfies ``RagSettings``."""

    embedding_model = "text-embedding-3-small"
    rag_distance_threshold = 0.5
    vector_store_provider = "memory"
    openai_api_key = "test-key"
    openai_api_base = ""
    embedding_drain_interval_seconds = 5.0
    embedding_drain_batch_size = 10
    job_stale_after_seconds = 300.0
    worker_id = "test-worker"
    ingest_on_startup = False
    database_url = "sqlite+aiosqlite:///:memory:"


class _ProtocolSettingsWithMetric(_ProtocolSettings):
    """A host that opts into cosine by adding the optional field."""

    rag_distance_metric = "cosine"


class TestResolveDistanceMetric:
    def test_explicit_argument_wins_over_settings(self) -> None:
        assert resolve_distance_metric("l2", _Settings("cosine")) == "l2"
        assert resolve_distance_metric("cosine", _Settings("l2")) == "cosine"

    def test_settings_win_over_the_default(self) -> None:
        assert resolve_distance_metric(None, _Settings("cosine")) == "cosine"

    def test_default_when_nothing_is_configured(self) -> None:
        assert resolve_distance_metric() == DEFAULT_DISTANCE_METRIC
        assert resolve_distance_metric(None, None) == "l2"
        assert resolve_distance_metric(None, _Settings()) == "l2"
        assert resolve_distance_metric(None, _NoneMetricSettings()) == "l2"

    def test_normalization_is_case_insensitive(self) -> None:
        assert resolve_distance_metric("COSINE") == "cosine"
        assert resolve_distance_metric("L2") == "l2"
        assert resolve_distance_metric("  Cosine  ") == "cosine"

    def test_unknown_metric_raises_listing_supported_names(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance metric") as exc_info:
            resolve_distance_metric("dot")
        message = str(exc_info.value)
        assert "'dot'" in message
        for supported in SUPPORTED_DISTANCE_METRICS:
            assert supported in message

    def test_unknown_metric_from_settings_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance metric"):
            resolve_distance_metric(None, _Settings("manhattan"))

    def test_supported_metrics_are_l2_and_cosine(self) -> None:
        assert SUPPORTED_DISTANCE_METRICS == ("l2", "cosine")

    def test_default_metric_is_l2(self) -> None:
        assert DEFAULT_DISTANCE_METRIC == "l2"
        metric: DistanceMetric = DEFAULT_DISTANCE_METRIC
        assert metric in SUPPORTED_DISTANCE_METRICS


class TestCosineMath:
    def test_hand_computed_similarity(self) -> None:
        # dot = 1*4 + 2*5 + 3*6 = 32; norms sqrt(14) and sqrt(77).
        similarity = cosine_similarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert similarity == pytest.approx(0.9746318461970762)

    def test_cosine_distance_is_one_minus_similarity(self) -> None:
        left = [1.0, 2.0, 3.0]
        right = [4.0, 5.0, 6.0]
        assert cosine_distance(left, right) == pytest.approx(1.0 - cosine_similarity(left, right))
        assert cosine_distance(left, right) == pytest.approx(0.02536815380292384)

    def test_identical_unit_vectors_score_zero_distance(self) -> None:
        unit = [1.0, 0.0, 0.0]
        assert cosine_similarity(unit, unit) == pytest.approx(1.0)
        assert cosine_distance(unit, unit) == pytest.approx(0.0)

    def test_orthogonal_vectors_score_one_distance(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_vectors_score_two_distance(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
        assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_zero_vector_scores_one_distance(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_distance([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)
        assert cosine_distance([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="same length"):
            cosine_distance([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_similarity_is_scale_invariant(self) -> None:
        assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)
        assert cosine_distance([2.0, 0.0], [5.0, 0.0]) == pytest.approx(0.0)
        # 45 degrees between the two axes.
        assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 / math.sqrt(2.0))


class TestNormalize:
    def test_returns_a_unit_vector(self) -> None:
        assert normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
        assert normalize([1.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])

    def test_result_norm_is_one(self) -> None:
        normalized = normalize([1.0, 2.0, 3.0, 4.0])
        norm = math.sqrt(sum(value * value for value in normalized))
        assert norm == pytest.approx(1.0)

    def test_zero_vector_is_returned_unchanged(self) -> None:
        assert normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_accepts_any_sequence_and_does_not_mutate_it(self) -> None:
        source = [3.0, 4.0]
        assert normalize(source) == pytest.approx([0.6, 0.8])
        assert source == [3.0, 4.0]
        assert normalize((3.0, 4.0)) == pytest.approx([0.6, 0.8])


class TestStoreMetricWiring:
    def test_memory_store_defaults_to_l2(self) -> None:
        store = InMemoryVectorStore(embedding_dim=3)
        assert store.metric == "l2"

    def test_memory_store_reads_the_optional_settings_field(self) -> None:
        store = InMemoryVectorStore(embedding_dim=3, settings=_ProtocolSettingsWithMetric())
        assert store.metric == "cosine"

    def test_memory_store_argument_beats_settings(self) -> None:
        store = InMemoryVectorStore(
            embedding_dim=3, settings=_ProtocolSettingsWithMetric(), metric="l2"
        )
        assert store.metric == "l2"

    def test_memory_store_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance metric"):
            InMemoryVectorStore(embedding_dim=3, metric="dot")

    def test_optional_metric_field_keeps_the_settings_protocol(self) -> None:
        assert isinstance(_ProtocolSettings(), RagSettings)
        assert isinstance(_ProtocolSettingsWithMetric(), RagSettings)

    def test_qdrant_rejects_the_cosine_metric(self) -> None:
        with pytest.raises(ValueError, match="unsupported by the Qdrant backend"):
            QdrantStore(table_name="docs", embedding_dim=3, metric="cosine")

    def test_qdrant_accepts_the_l2_metric(self) -> None:
        store = QdrantStore(table_name="docs", embedding_dim=3)
        assert store.metric == "l2"

    def test_chroma_rejects_the_cosine_metric(self) -> None:
        with pytest.raises(ValueError, match="unsupported by the Chroma backend"):
            ChromaStore(table_name="docs", embedding_dim=3, metric="cosine")

    def test_chroma_accepts_the_l2_metric(self) -> None:
        store = ChromaStore(table_name="docs", embedding_dim=3)
        assert store.metric == "l2"


class _RecordingStore(VectorStore):
    """Host backend that records the kwargs the factory passed it."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        return 0

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        return None


class _StrictStore(VectorStore):
    """Host backend whose constructor has no ``metric`` parameter."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str,
        id_field: str,
        embedding_dim: int | None = None,
        session_factory: Any = None,
        settings: Any = None,
        embedding_fn: Any = None,
    ) -> None:
        self.table_name = table_name
        self.id_field = id_field
        self.embedding_dim = embedding_dim

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def count(self, session: Any = None, **kwargs: Any) -> int:
        return 0

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        return None


class TestMetricThroughTheFactories:
    """``get_vector_store`` and the registry factory thread the metric through."""

    def test_get_vector_store_forwards_an_explicit_metric(self) -> None:
        store = get_vector_store(table_name="docs", id_field="id", embedding_dim=3, metric="cosine")

        assert isinstance(store, InMemoryVectorStore)
        assert store.metric == "cosine"

    def test_get_vector_store_leaves_the_metric_to_the_store_by_default(self) -> None:
        store = get_vector_store(table_name="docs", id_field="id", embedding_dim=3)

        assert isinstance(store, InMemoryVectorStore)
        assert store.metric == DEFAULT_DISTANCE_METRIC

    def test_get_vector_store_argument_beats_the_settings_field(self) -> None:
        store = get_vector_store(
            table_name="docs",
            id_field="id",
            embedding_dim=3,
            settings=_ProtocolSettingsWithMetric(),
            metric="l2",
        )

        assert store.metric == "l2"

    def test_get_vector_store_with_an_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance metric"):
            get_vector_store(table_name="docs", id_field="id", embedding_dim=3, metric="dot")

    def test_factory_forwards_the_metric_when_it_is_set(self) -> None:
        registry = Registry[VectorStore]()

        @registry.register("recording-backend")
        class RecordingBackend(_RecordingStore):
            pass

        factory = RegistryVectorStoreFactory(registry=registry)
        settings = SimpleNamespace(vector_store_provider="recording-backend")
        store = factory.create(
            settings=settings, table_name="chunks", embedding_dim=4, metric="cosine"
        )

        assert isinstance(store, RecordingBackend)
        assert store.kwargs["metric"] == "cosine"

    def test_factory_omits_the_metric_when_it_is_none(self) -> None:
        registry = Registry[VectorStore]()

        @registry.register("recording-backend")
        class RecordingBackend(_RecordingStore):
            pass

        factory = RegistryVectorStoreFactory(registry=registry)
        settings = SimpleNamespace(vector_store_provider="recording-backend")
        store = factory.create(settings=settings, table_name="chunks", embedding_dim=4)

        assert isinstance(store, RecordingBackend)
        assert "metric" not in store.kwargs

    def test_custom_backend_without_the_metric_kwarg_keeps_working(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = Registry[VectorStore]()

        @registry.register("strict-backend")
        class StrictBackend(_StrictStore):
            pass

        monkeypatch.setattr(
            "ragit.stores.registry.RegistryVectorStoreFactory",
            lambda: RegistryVectorStoreFactory(
                registry=registry, default_provider="strict-backend"
            ),
        )

        # The metric kwarg is forwarded only when it is not None, so a
        # host backend whose constructor never declared it still builds
        # through get_vector_store.
        store = get_vector_store(table_name="chunks", id_field="id", embedding_dim=4)

        assert isinstance(store, StrictBackend)
        assert store.table_name == "chunks"

    def test_metric_surface_is_exported_from_the_stores_package(self) -> None:
        assert DEFAULT_DISTANCE_METRIC == "l2"
        assert SUPPORTED_DISTANCE_METRICS == ("l2", "cosine")
        assert resolve_distance_metric("COSINE") == "cosine"
