"""Tests for EmbeddingCache: isolation, LRU eviction, TTL, and profile invalidation.

Covers:
* Cache hit returns the stored vector (fresh copy).
* Cache miss returns None.
* Scope A cannot read scope B's vectors (cross-tenant isolation).
* allow_cross_tenant=True shares entries across scopes.
* LRU eviction removes oldest entries when max_entries is reached.
* TTL expiry causes a miss after the entry expires.
* invalidate_profile removes matching entries only.
* invalidate_scope removes all entries for a scope.
* Profile change produces a different cache key (automatic invalidation).
* Policy validation rejects invalid parameters.
"""

from __future__ import annotations

import time

import pytest

from ragit.embeddings.cache import EmbeddingCache, EmbeddingCacheConfig
from ragit.embeddings.profile import EmbeddingProfile, TaskMode


def _make_profile(model_id: str = "model-a", dimension: int = 3) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider_id="test",
        model_id=model_id,
        dimension=dimension,
    )


def _vec(value: float = 1.0, dim: int = 3) -> list[float]:
    return [value] * dim


class TestCacheHitAndMiss:
    def test_miss_returns_none(self) -> None:
        cache = EmbeddingCache()
        result = cache.get("scope-a", "hash-abc", _make_profile(), TaskMode.DOCUMENT)
        assert result is None

    def test_hit_returns_vector(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        vec = _vec(1.0)
        cache.put("scope-a", "hash-abc", profile, TaskMode.DOCUMENT, vec)
        result = cache.get("scope-a", "hash-abc", profile, TaskMode.DOCUMENT)
        assert result == vec

    def test_hit_returns_fresh_copy(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        vec = _vec(1.0)
        cache.put("scope-a", "hash-abc", profile, TaskMode.DOCUMENT, vec)
        r1 = cache.get("scope-a", "hash-abc", profile, TaskMode.DOCUMENT)
        r1[0] = 999.0  # mutate the returned copy
        r2 = cache.get("scope-a", "hash-abc", profile, TaskMode.DOCUMENT)
        assert r2 == _vec(1.0)  # original unchanged

    def test_different_task_mode_is_different_key(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        cache.put("scope-a", "hash", profile, TaskMode.DOCUMENT, _vec(1.0))
        assert cache.get("scope-a", "hash", profile, TaskMode.QUERY) is None

    def test_different_content_hash_is_different_key(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        cache.put("scope-a", "hash-1", profile, TaskMode.DOCUMENT, _vec(1.0))
        assert cache.get("scope-a", "hash-2", profile, TaskMode.DOCUMENT) is None


class TestScopeIsolation:
    def test_scope_a_cannot_read_scope_b_entries(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        cache.put("scope-b", "hash-abc", profile, TaskMode.DOCUMENT, _vec(2.0))
        result = cache.get("scope-a", "hash-abc", profile, TaskMode.DOCUMENT)
        assert result is None

    def test_same_scope_can_read_own_entries(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        vec = _vec(3.0)
        cache.put("tenant-x", "hash-abc", profile, TaskMode.DOCUMENT, vec)
        assert cache.get("tenant-x", "hash-abc", profile, TaskMode.DOCUMENT) == vec

    def test_allow_cross_tenant_shares_across_scopes(self) -> None:
        config = EmbeddingCacheConfig(allow_cross_tenant=True)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()
        cache.put("scope-a", "hash-abc", profile, TaskMode.DOCUMENT, _vec(5.0))
        # scope-b can read what scope-a wrote
        result = cache.get("scope-b", "hash-abc", profile, TaskMode.DOCUMENT)
        assert result == _vec(5.0)

    def test_default_config_never_shares_across_scopes(self) -> None:
        """Ensure cross-tenant=False is the default (critical privacy requirement)."""
        cache = EmbeddingCache()
        assert cache._config.allow_cross_tenant is False


class TestLRUEviction:
    def test_oldest_entry_evicted_on_overflow(self) -> None:
        config = EmbeddingCacheConfig(max_entries=2)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()

        cache.put("s", "h1", profile, TaskMode.DOCUMENT, _vec(1.0))
        cache.put("s", "h2", profile, TaskMode.DOCUMENT, _vec(2.0))
        cache.put("s", "h3", profile, TaskMode.DOCUMENT, _vec(3.0))  # evicts h1

        assert cache.get("s", "h1", profile, TaskMode.DOCUMENT) is None
        assert cache.get("s", "h2", profile, TaskMode.DOCUMENT) == _vec(2.0)
        assert cache.get("s", "h3", profile, TaskMode.DOCUMENT) == _vec(3.0)

    def test_access_promotes_entry_to_recent(self) -> None:
        config = EmbeddingCacheConfig(max_entries=2)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()

        cache.put("s", "h1", profile, TaskMode.DOCUMENT, _vec(1.0))
        cache.put("s", "h2", profile, TaskMode.DOCUMENT, _vec(2.0))
        # Access h1 to make it recent.
        cache.get("s", "h1", profile, TaskMode.DOCUMENT)
        # h2 is now the least recently used.
        cache.put("s", "h3", profile, TaskMode.DOCUMENT, _vec(3.0))  # evicts h2

        assert cache.get("s", "h2", profile, TaskMode.DOCUMENT) is None
        assert cache.get("s", "h1", profile, TaskMode.DOCUMENT) == _vec(1.0)

    def test_size_reflects_evictions(self) -> None:
        config = EmbeddingCacheConfig(max_entries=3)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()
        for i in range(5):
            cache.put("s", f"h{i}", profile, TaskMode.DOCUMENT, _vec(float(i)))
        assert cache.size() == 3


class TestTTLExpiry:
    def test_entry_available_before_expiry(self, monkeypatch) -> None:
        config = EmbeddingCacheConfig(ttl_seconds=10.0)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()
        fake_now = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

        cache.put("s", "h", profile, TaskMode.DOCUMENT, _vec(1.0))
        fake_now[0] = 5.0
        assert cache.get("s", "h", profile, TaskMode.DOCUMENT) == _vec(1.0)

    def test_entry_expired_after_ttl(self, monkeypatch) -> None:
        config = EmbeddingCacheConfig(ttl_seconds=10.0)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()
        fake_now = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

        cache.put("s", "h", profile, TaskMode.DOCUMENT, _vec(1.0))
        fake_now[0] = 11.0  # past TTL
        assert cache.get("s", "h", profile, TaskMode.DOCUMENT) is None

    def test_expired_entry_removed_from_size(self, monkeypatch) -> None:
        config = EmbeddingCacheConfig(ttl_seconds=5.0)
        cache = EmbeddingCache(config=config)
        profile = _make_profile()
        fake_now = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

        cache.put("s", "h", profile, TaskMode.DOCUMENT, _vec(1.0))
        assert cache.size() == 1
        fake_now[0] = 10.0
        cache.get("s", "h", profile, TaskMode.DOCUMENT)  # triggers cleanup
        assert cache.size() == 0


class TestInvalidation:
    def test_invalidate_profile_removes_matching_entries(self) -> None:
        cache = EmbeddingCache()
        p1 = _make_profile(model_id="model-1")
        p2 = _make_profile(model_id="model-2")
        cache.put("s", "h1", p1, TaskMode.DOCUMENT, _vec(1.0))
        cache.put("s", "h2", p2, TaskMode.DOCUMENT, _vec(2.0))

        removed = cache.invalidate_profile("s", p1.canonical_hash())
        assert removed == 1
        assert cache.get("s", "h1", p1, TaskMode.DOCUMENT) is None
        assert cache.get("s", "h2", p2, TaskMode.DOCUMENT) == _vec(2.0)

    def test_invalidate_scope_removes_all_scope_entries(self) -> None:
        cache = EmbeddingCache()
        profile = _make_profile()
        cache.put("scope-a", "h1", profile, TaskMode.DOCUMENT, _vec(1.0))
        cache.put("scope-a", "h2", profile, TaskMode.DOCUMENT, _vec(2.0))
        cache.put("scope-b", "h1", profile, TaskMode.DOCUMENT, _vec(3.0))

        removed = cache.invalidate_scope("scope-a")
        assert removed == 2
        assert cache.get("scope-a", "h1", profile, TaskMode.DOCUMENT) is None
        assert cache.get("scope-a", "h2", profile, TaskMode.DOCUMENT) is None
        # scope-b untouched
        assert cache.get("scope-b", "h1", profile, TaskMode.DOCUMENT) == _vec(3.0)

    def test_profile_change_produces_different_key(self) -> None:
        """Profile change automatically prevents reuse — different key, cache miss."""
        cache = EmbeddingCache()
        old_profile = _make_profile(model_id="v1")
        new_profile = _make_profile(model_id="v2")
        cache.put("s", "h", old_profile, TaskMode.DOCUMENT, _vec(1.0))
        # Same scope + content_hash but different profile → miss
        assert cache.get("s", "h", new_profile, TaskMode.DOCUMENT) is None


class TestConfigValidation:
    def test_max_entries_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_entries"):
            EmbeddingCacheConfig(max_entries=0)

    def test_negative_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            EmbeddingCacheConfig(ttl_seconds=-1.0)

    def test_zero_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            EmbeddingCacheConfig(ttl_seconds=0.0)
