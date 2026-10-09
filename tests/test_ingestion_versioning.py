"""Tests for ragit.ingestion.versioning."""

from __future__ import annotations

import pytest

from ragit.chunking import MarkdownSectionChunker, SlidingWindowChunker
from ragit.embeddings.dimensions import register_dimension
from ragit.ingestion import (
    IngestionSnapshot,
    build_snapshot,
    snapshot_matches,
)


class _FakeSettings:
    """Minimal settings object satisfying the RagSettings protocol."""

    def __init__(self, embedding_model: str = "text-embedding-3-small") -> None:
        self.embedding_model = embedding_model
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


def _snapshot(**overrides: object) -> IngestionSnapshot:
    fields: dict[str, object] = {
        "source_hash": "hash",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "chunker_version": "v1",
        "chunk_size": 600,
        "overlap": 100,
        "retrieval_schema_version": "v1",
    }
    fields.update(overrides)
    return IngestionSnapshot(**fields)  # type: ignore[arg-type]


def test_snapshot_matches_none_is_false() -> None:
    assert snapshot_matches(_snapshot(), None) is False


def test_snapshot_matches_equal_snapshots() -> None:
    assert snapshot_matches(_snapshot(), _snapshot()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_hash": "other-hash"},
        {"embedding_model": "text-embedding-3-large"},
        {"embedding_dim": 3072},
        {"chunker_version": "section-aware-v1"},
        {"chunk_size": 800},
        {"overlap": 200},
        {"retrieval_schema_version": "v2"},
    ],
)
def test_snapshot_matches_detects_any_changed_field(overrides: dict) -> None:
    assert snapshot_matches(_snapshot(), _snapshot(**overrides)) is False


def test_build_snapshot_reads_settings_and_chunker() -> None:
    chunker = MarkdownSectionChunker(
        SlidingWindowChunker(chunk_size=600, overlap=100),
        source_name="policy.md",
    )
    snapshot = build_snapshot("the-hash", _FakeSettings(), chunker)

    assert snapshot.source_hash == "the-hash"
    assert snapshot.embedding_model == "text-embedding-3-small"
    assert snapshot.embedding_dim == 1536
    assert snapshot.chunker_version == chunker.version
    assert snapshot.chunk_size == 600
    assert snapshot.overlap == 100
    assert snapshot.retrieval_schema_version == "v1"


def test_build_snapshot_resolves_the_dimension_for_the_model() -> None:
    register_dimension("ragit-test-dimension-model", 7)
    chunker = MarkdownSectionChunker(SlidingWindowChunker(), source_name="policy.md")

    snapshot = build_snapshot("the-hash", _FakeSettings("ragit-test-dimension-model"), chunker)

    assert snapshot.embedding_dim == 7


def test_snapshots_compare_by_value() -> None:
    assert _snapshot() == _snapshot()
    assert _snapshot() != _snapshot(source_hash="other")
