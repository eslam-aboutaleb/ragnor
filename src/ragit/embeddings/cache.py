"""Embedding cache with scoped isolation and profile-keyed invalidation.

:class:`EmbeddingCache` stores vectors keyed on a combination of
isolation scope, content hash, embedding profile, and task mode.
Cross-tenant or cross-scope cache sharing is **never enabled by
default** and requires an explicit opt-in flag.

Privacy and retention policy
-----------------------------
* Each cache entry is scoped to an *isolation scope* (typically a
  tenant or collection ID).  An entry written under scope A is
  never returned to a request under scope B unless
  ``allow_cross_tenant=True`` is set on the
  :class:`EmbeddingCacheConfig`.
* ``allow_cross_tenant`` must be ``False`` (the default) in any
  deployment where scopes carry independent access-control policies.
* Entries expire after ``ttl_seconds`` when set.  The TTL is enforced
  lazily on read: an expired entry is removed and a cache miss is
  reported.
* Profile changes (different :meth:`~ragit.embeddings.profile.EmbeddingProfile.canonical_hash`)
  automatically produce different cache keys, so reindexing after a
  profile change does not require manual cache invalidation.
* The cache is an in-process LRU structure.  It is not persisted across
  restarts and carries no encryption.  Do not cache vectors that contain
  user PII in shared-memory environments without additional controls.

Usage::

    config = EmbeddingCacheConfig(max_entries=1024, ttl_seconds=3600)
    cache = EmbeddingCache(config=config)

    vector = cache.get("tenant-a", content_hash, profile, TaskMode.DOCUMENT)
    if vector is None:
        vector = await embed(text)
        cache.put("tenant-a", content_hash, profile, TaskMode.DOCUMENT, vector)
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from ragit.embeddings.profile import EmbeddingProfile, TaskMode


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCacheConfig:
    """Configuration for :class:`EmbeddingCache`.

    Attributes:
        max_entries: Maximum number of vectors held in the LRU cache.
            Older entries are evicted when the limit is reached.
        ttl_seconds: Optional time-to-live in seconds.  ``None`` means
            entries never expire.
        allow_cross_tenant: When ``True``, the isolation scope is
            omitted from the cache key, allowing reuse across tenants.
            **Never enable this unless all scopes share the same
            access-control domain.**  Defaults to ``False``.
    """

    max_entries: int = 4096
    ttl_seconds: float | None = None
    allow_cross_tenant: bool = False

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {self.ttl_seconds}")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    vector: list[float]
    expires_at: float | None  # monotonic clock value, or None for no expiry
    isolation_scope: str
    profile_hash: str


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Bounded, scoped, profile-keyed in-process embedding cache.

    Stores embedding vectors keyed on isolation scope, content hash,
    :class:`~ragit.embeddings.profile.EmbeddingProfile`, and task mode.
    Provides LRU eviction and optional TTL expiry.

    This class is **not thread-safe**.  Use it from a single async event
    loop; do not share instances across threads without external locking.

    Args:
        config: Cache configuration governing capacity, TTL, and cross-tenant rules.
    """

    def __init__(self, config: EmbeddingCacheConfig | None = None) -> None:
        self._config = config or EmbeddingCacheConfig()
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        isolation_scope: str,
        content_hash: str,
        profile: EmbeddingProfile,
        task_mode: TaskMode,
    ) -> list[float] | None:
        """Return the cached vector, or ``None`` on a miss.

        Args:
            isolation_scope: Tenant / collection identifier.  Scopes
                are never mixed unless ``allow_cross_tenant=True``.
            content_hash: SHA-256 hex digest of the normalised content.
            profile: Embedding profile used to embed the content.
            task_mode: Task mode used when the vector was embedded.

        Returns:
            A fresh copy of the cached vector, or ``None`` on a miss or
            expiry.
        """
        key = self._make_key(isolation_scope, content_hash, profile, task_mode)
        entry = self._store.get(key)
        if entry is None:
            return None

        if self._is_expired(entry):
            del self._store[key]
            return None

        self._store.move_to_end(key)
        return list(entry.vector)

    def put(
        self,
        isolation_scope: str,
        content_hash: str,
        profile: EmbeddingProfile,
        task_mode: TaskMode,
        vector: list[float],
    ) -> None:
        """Store a vector in the cache.

        Args:
            isolation_scope: Tenant / collection identifier.
            content_hash: SHA-256 hex digest of the normalised content.
            profile: Embedding profile used to embed the content.
            task_mode: Task mode used when the vector was embedded.
            vector: The embedding vector to cache (copied internally).
        """
        key = self._make_key(isolation_scope, content_hash, profile, task_mode)
        expires_at: float | None = None
        if self._config.ttl_seconds is not None:
            expires_at = time.monotonic() + self._config.ttl_seconds

        profile_hash = profile.canonical_hash()
        self._store[key] = _CacheEntry(
            vector=list(vector),
            expires_at=expires_at,
            isolation_scope=isolation_scope,
            profile_hash=profile_hash,
        )
        self._store.move_to_end(key)
        self._evict_if_needed()

    def invalidate_profile(self, isolation_scope: str, profile_hash: str) -> int:
        """Remove all entries for *profile_hash* under *isolation_scope*.

        Useful when a profile is retired and any cached vectors would be
        misaligned with the new index generation.

        Args:
            isolation_scope: Scope whose entries should be invalidated.
            profile_hash: :meth:`EmbeddingProfile.canonical_hash` of the
                profile to invalidate.

        Returns:
            Number of entries removed.
        """
        to_delete = [
            k
            for k, entry in self._store.items()
            if (self._config.allow_cross_tenant or entry.isolation_scope == isolation_scope)
            and entry.profile_hash == profile_hash
        ]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)

    def invalidate_scope(self, isolation_scope: str) -> int:
        """Remove all entries for *isolation_scope*.

        Args:
            isolation_scope: The scope to purge.

        Returns:
            Number of entries removed.
        """
        to_delete = [
            k for k, entry in self._store.items() if entry.isolation_scope == isolation_scope
        ]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)

    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(
        self,
        isolation_scope: str,
        content_hash: str,
        profile: EmbeddingProfile,
        task_mode: TaskMode,
    ) -> str:
        """Compute the cache key.

        The key is a SHA-256 digest of the concatenation of (when not
        cross-tenant) the scope, the content hash, the profile's canonical
        hash, and the task mode.  All components are separated by null bytes
        to prevent collisions from concatenating adjacent fields.
        """
        profile_hash = profile.canonical_hash()
        if self._config.allow_cross_tenant:
            raw = f"{content_hash}\x00{profile_hash}\x00{task_mode}"
        else:
            raw = f"{isolation_scope}\x00{content_hash}\x00{profile_hash}\x00{task_mode}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _is_expired(self, entry: _CacheEntry) -> bool:
        """Return ``True`` when *entry* is past its TTL."""
        return entry.expires_at is not None and time.monotonic() > entry.expires_at

    def _evict_if_needed(self) -> None:
        """Evict oldest entries until the cache is within ``max_entries``."""
        while len(self._store) > self._config.max_entries:
            self._store.popitem(last=False)
