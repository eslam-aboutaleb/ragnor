"""Tests for EmbeddingProfile: canonical hash, compatibility, and validation.

Covers:
* Canonical hash changes when any semantic field changes.
* Same model name / different service_url → different hash → incompatible.
* Task-mode difference alone is compatible for bi-encoder retrieval.
* profiles_require_reindex reflects hash changes correctly.
* validate_profile_at_startup warns / raises as appropriate.
* Secrets in preprocessing_config are stripped from the hash.
* IngestionSnapshot includes embedding_profile_hash and build_snapshot wires it.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from ragnor.embeddings.dimensions import register_dimension
from ragnor.embeddings.profile import (
    DistanceMetric,
    EmbeddingProfile,
    NormalizationMode,
    ProfileValidationError,
    TaskMode,
    profiles_compatible_for_query,
    profiles_require_reindex,
    validate_profile_at_startup,
)
from ragnor.ingestion.versioning import IngestionSnapshot, build_snapshot


def _make_profile(**overrides) -> EmbeddingProfile:
    defaults = {
        "provider_id": "openai",
        "model_id": "text-embedding-3-small",
        "dimension": 1536,
    }
    defaults.update(overrides)
    return EmbeddingProfile(**defaults)


class TestCanonicalHash:
    def test_identical_profiles_produce_same_hash(self) -> None:
        p1 = _make_profile()
        p2 = _make_profile()
        assert p1.canonical_hash() == p2.canonical_hash()

    def test_hash_is_stable_across_calls(self) -> None:
        p = _make_profile()
        assert p.canonical_hash() == p.canonical_hash()

    def test_provider_id_change_changes_hash(self) -> None:
        p1 = _make_profile(provider_id="openai")
        p2 = _make_profile(provider_id="azure")
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_model_id_change_changes_hash(self) -> None:
        p1 = _make_profile(model_id="model-a")
        p2 = _make_profile(model_id="model-b")
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_dimension_change_changes_hash(self) -> None:
        p1 = _make_profile(dimension=1536)
        p2 = _make_profile(dimension=3072)
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_normalization_change_changes_hash(self) -> None:
        p1 = _make_profile(normalization=NormalizationMode.NONE)
        p2 = _make_profile(normalization=NormalizationMode.L2)
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_distance_metric_change_changes_hash(self) -> None:
        p1 = _make_profile(distance_metric=DistanceMetric.COSINE)
        p2 = _make_profile(distance_metric=DistanceMetric.L2)
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_task_mode_change_changes_hash(self) -> None:
        p1 = _make_profile(task_mode=TaskMode.DOCUMENT)
        p2 = _make_profile(task_mode=TaskMode.QUERY)
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_model_revision_change_changes_hash(self) -> None:
        p1 = _make_profile(model_revision="2024-01")
        p2 = _make_profile(model_revision="2024-06")
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_service_url_change_changes_hash(self) -> None:
        p1 = _make_profile(service_url="https://api.openai.com/v1")
        p2 = _make_profile(service_url="https://my-proxy.example.com/v1")
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_same_model_different_service_url_is_different_hash(self) -> None:
        """Same model name through a different service is not automatically equivalent."""
        p1 = _make_profile(provider_id="openai", service_url=None)
        p2 = _make_profile(provider_id="openai", service_url="https://proxy.example.com")
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_preprocessing_config_change_changes_hash(self) -> None:
        p1 = _make_profile(preprocessing_config={"truncation": "right"})
        p2 = _make_profile(preprocessing_config={"truncation": "left"})
        assert p1.canonical_hash() != p2.canonical_hash()

    def test_secrets_are_stripped_from_hash(self) -> None:
        """api_key in preprocessing_config must not feed the hash."""
        p_clean = _make_profile(preprocessing_config={"truncation": "right"})
        p_secret = _make_profile(
            preprocessing_config={"truncation": "right", "api_key": "sk-secret"}
        )
        assert p_clean.canonical_hash() == p_secret.canonical_hash()

    def test_hash_is_hex_string(self) -> None:
        h = _make_profile().canonical_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # must be valid hex


class TestCompatibility:
    def test_identical_profiles_are_compatible(self) -> None:
        p = _make_profile()
        assert profiles_compatible_for_query(p, p)

    def test_different_task_modes_are_compatible(self) -> None:
        """Bi-encoder: document and query task modes are intentionally compatible."""
        doc = _make_profile(task_mode=TaskMode.DOCUMENT)
        qry = _make_profile(task_mode=TaskMode.QUERY)
        assert profiles_compatible_for_query(doc, qry)

    def test_different_provider_is_incompatible(self) -> None:
        doc = _make_profile(provider_id="openai")
        qry = _make_profile(provider_id="cohere")
        assert not profiles_compatible_for_query(doc, qry)

    def test_different_model_is_incompatible(self) -> None:
        doc = _make_profile(model_id="model-a")
        qry = _make_profile(model_id="model-b")
        assert not profiles_compatible_for_query(doc, qry)

    def test_different_dimension_is_incompatible(self) -> None:
        doc = _make_profile(dimension=1536)
        qry = _make_profile(dimension=768)
        assert not profiles_compatible_for_query(doc, qry)

    def test_different_service_url_is_incompatible(self) -> None:
        doc = _make_profile(service_url="https://a.example.com")
        qry = _make_profile(service_url="https://b.example.com")
        assert not profiles_compatible_for_query(doc, qry)

    def test_different_model_revision_is_incompatible(self) -> None:
        doc = _make_profile(model_revision="v1")
        qry = _make_profile(model_revision="v2")
        assert not profiles_compatible_for_query(doc, qry)

    def test_different_normalization_is_incompatible(self) -> None:
        doc = _make_profile(normalization=NormalizationMode.NONE)
        qry = _make_profile(normalization=NormalizationMode.L2)
        assert not profiles_compatible_for_query(doc, qry)


class TestProfilesRequireReindex:
    def test_same_profile_does_not_require_reindex(self) -> None:
        p = _make_profile()
        assert not profiles_require_reindex(p, p)

    def test_dimension_change_requires_reindex(self) -> None:
        old = _make_profile(dimension=1536)
        new = _make_profile(dimension=3072)
        assert profiles_require_reindex(old, new)

    def test_model_change_requires_reindex(self) -> None:
        old = _make_profile(model_id="small")
        new = _make_profile(model_id="large")
        assert profiles_require_reindex(old, new)

    def test_task_mode_change_requires_reindex(self) -> None:
        old = _make_profile(task_mode=TaskMode.SYMMETRIC)
        new = _make_profile(task_mode=TaskMode.DOCUMENT)
        assert profiles_require_reindex(old, new)


class TestValidation:
    def test_known_model_matching_dim_passes_silently(self) -> None:
        register_dimension("test-known-model-v1", 512)
        p = _make_profile(model_id="test-known-model-v1", dimension=512)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_profile_at_startup(p, strict=False)

    def test_unknown_model_warns_in_lenient_mode(self) -> None:
        p = _make_profile(model_id="unknown-model-xyz", dimension=1234)
        with pytest.warns(UserWarning, match="unknown-model-xyz"):
            validate_profile_at_startup(p, strict=False)

    def test_unknown_model_raises_in_strict_mode(self) -> None:
        p = _make_profile(model_id="unknown-model-strict", dimension=999)
        with pytest.raises(ProfileValidationError, match="unknown-model-strict"):
            validate_profile_at_startup(p, strict=True)

    def test_dimension_mismatch_warns_in_lenient_mode(self) -> None:
        register_dimension("test-mismatch-model", 768)
        p = _make_profile(model_id="test-mismatch-model", dimension=512)
        with pytest.warns(UserWarning, match="test-mismatch-model"):
            validate_profile_at_startup(p, strict=False)

    def test_dimension_mismatch_raises_in_strict_mode(self) -> None:
        register_dimension("test-mismatch-model-strict", 768)
        p = _make_profile(model_id="test-mismatch-model-strict", dimension=512)
        with pytest.raises(ProfileValidationError, match="test-mismatch-model-strict"):
            validate_profile_at_startup(p, strict=True)


class TestIngestionSnapshotIntegration:
    def test_snapshot_has_embedding_profile_hash_field(self) -> None:
        snap = IngestionSnapshot(
            source_hash="abc",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
            chunker_version="v1",
            chunk_size=200,
            overlap=20,
            retrieval_schema_version="v1",
        )
        assert snap.embedding_profile_hash == ""

    def test_build_snapshot_records_profile_hash(self) -> None:
        profile = _make_profile()

        settings = MagicMock()
        settings.embedding_model = "text-embedding-3-small"

        chunker = MagicMock()
        chunker.config = {"chunk_size": 200, "overlap": 20}
        chunker.version = "v1"

        snap = build_snapshot(
            source_hash="deadbeef",
            settings=settings,
            chunker=chunker,
            embedding_profile=profile,
        )
        assert snap.embedding_profile_hash == profile.canonical_hash()

    def test_build_snapshot_without_profile_has_empty_hash(self) -> None:
        settings = MagicMock()
        settings.embedding_model = "text-embedding-3-small"

        chunker = MagicMock()
        chunker.config = {"chunk_size": 200, "overlap": 20}
        chunker.version = "v1"

        snap = build_snapshot(
            source_hash="deadbeef",
            settings=settings,
            chunker=chunker,
        )
        assert snap.embedding_profile_hash == ""
