"""Embedding profile: identity, configuration, and compatibility rules.

An :class:`EmbeddingProfile` captures every dimension of an embedding
configuration that matters for compatibility: the provider, service
endpoint, model, dimension, normalization, distance metric, task mode,
and non-secret preprocessing options.  Two profiles that produce
incompatible vector spaces have a different :meth:`EmbeddingProfile.canonical_hash`,
so storing the hash alongside indexed vectors makes compatibility
checks unambiguous.

The same model name served through a different provider or service URL
is *not* automatically equivalent -- ``provider_id`` and
``service_url`` both feed the canonical hash.

Usage::

    profile = EmbeddingProfile(
        provider_id="openai",
        model_id="text-embedding-3-small",
        dimension=1536,
    )
    h = profile.canonical_hash()  # stable, secrets-free SHA-256 hex

    # Validate at startup (warns for unknown dimensions by default):
    validate_profile_at_startup(profile)

    # Compatible asymmetric pair for bi-encoder retrieval:
    assert profiles_compatible_for_query(doc_profile, query_profile)
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ragnor.embeddings._safe_config import safe_config
from ragnor.embeddings.dimensions import FALLBACK_DIMENSION, _DIMENSIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskMode(StrEnum):
    """Embedding task mode.

    Some providers (e.g. Cohere, Voyage) use different model weights or
    instruction prefixes for document and query encoding; encoding documents
    and queries with the same mode may reduce retrieval quality.

    ``SYMMETRIC`` is the fallback for models that do not distinguish.
    """

    DOCUMENT = "document"
    QUERY = "query"
    SYMMETRIC = "symmetric"


class DistanceMetric(StrEnum):
    """Distance / similarity metric used by the vector index."""

    COSINE = "cosine"
    L2 = "l2"
    DOT_PRODUCT = "dot_product"


class NormalizationMode(StrEnum):
    """Vector normalization applied by the provider before returning."""

    NONE = "none"
    L2 = "l2"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProfileValidationError(ValueError):
    """Raised when an :class:`EmbeddingProfile` fails validation.

    Distinct from ``ValueError`` so callers can catch profile problems
    separately from generic value errors.
    """


# ---------------------------------------------------------------------------
# EmbeddingProfile dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingProfile:
    """Describes a concrete embedding configuration.

    All attributes that affect vector-space geometry feed
    :meth:`canonical_hash`.  Secrets (API keys, tokens) must **never**
    appear here; use ``preprocessing_config`` only for non-secret options
    and pass it through :func:`~ragnor.ingestion.versioning.safe_config`
    before storing.

    Attributes:
        provider_id: Stable identifier for the embedding provider
            (e.g. ``"openai"``, ``"cohere"``, ``"voyage"``).  Distinct
            from the model name: the same model hosted by two providers
            is *not* automatically equivalent.
        model_id: Embedding model identifier forwarded to the provider
            (e.g. ``"text-embedding-3-small"``).
        dimension: Output vector dimension.
        normalization: Normalization the provider applies before returning
            the vector.  Affects dot-product equivalence.
        distance_metric: Metric the downstream vector index uses.
        task_mode: Default task mode for this profile.  Per-call overrides
            are supported by :class:`~ragnor.embeddings.executor.EmbeddingExecutor`.
        model_revision: Optional provider-reported model revision / snapshot
            tag.  When present, a revision change is treated as a model
            change (different canonical hash).
        service_url: Optional provider endpoint URL (without secrets).
            ``None`` means the provider's default endpoint.  Two profiles
            that differ only in ``service_url`` are not equivalent.
        preprocessing_config: Non-secret preprocessing options forwarded
            to the provider (e.g. instruction prefixes, truncation strategy).
            Secrets are stripped automatically by :meth:`canonical_hash`.
    """

    provider_id: str
    model_id: str
    dimension: int
    normalization: NormalizationMode = NormalizationMode.NONE
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    task_mode: TaskMode = TaskMode.SYMMETRIC
    model_revision: str | None = None
    service_url: str | None = None
    preprocessing_config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Canonical hash
    # ------------------------------------------------------------------

    def canonical_hash(self) -> str:
        """Return the stable, secrets-free SHA-256 hex digest of this profile.

        The hash covers all semantically relevant fields.  ``safe_config``
        is applied to ``preprocessing_config`` to strip secrets before
        hashing.  The result is stable across Python interpreter restarts
        (no hash-randomisation).

        Returns:
            40-character lower-case hex string.
        """
        payload: dict[str, Any] = {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "normalization": str(self.normalization),
            "distance_metric": str(self.distance_metric),
            "task_mode": str(self.task_mode),
            "model_revision": self.model_revision,
            "service_url": self.service_url,
            "preprocessing_config": safe_config(self.preprocessing_config) or {},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_known_dimension(self) -> bool:
        """Return ``True`` when the dimension matches the registry."""
        registered = _DIMENSIONS.get(self.model_id)
        return registered is None or registered == self.dimension

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EmbeddingProfile(provider={self.provider_id!r}, model={self.model_id!r}, "
            f"dim={self.dimension}, task={self.task_mode}, "
            f"hash={self.canonical_hash()[:8]}...)"
        )


# ---------------------------------------------------------------------------
# Compatibility rules
# ---------------------------------------------------------------------------


def profiles_compatible_for_query(
    doc_profile: EmbeddingProfile,
    query_profile: EmbeddingProfile,
) -> bool:
    """Return whether ``query_profile`` is compatible with ``doc_profile``.

    Asymmetric task modes (document vs. query) are *intentionally*
    compatible -- bi-encoder retrieval requires this.  Two profiles are
    compatible when they share the same provider, service endpoint, model,
    model revision, dimension, normalization, and distance metric.  Task
    mode differences are allowed.

    Args:
        doc_profile: Profile used to embed stored documents.
        query_profile: Profile used to embed the query.

    Returns:
        ``True`` when the profiles can be used together safely.
    """
    return (
        doc_profile.provider_id == query_profile.provider_id
        and doc_profile.service_url == query_profile.service_url
        and doc_profile.model_id == query_profile.model_id
        and doc_profile.model_revision == query_profile.model_revision
        and doc_profile.dimension == query_profile.dimension
        and doc_profile.normalization == query_profile.normalization
        and doc_profile.distance_metric == query_profile.distance_metric
    )


def profiles_require_reindex(
    old_profile: EmbeddingProfile,
    new_profile: EmbeddingProfile,
) -> bool:
    """Return whether switching from ``old_profile`` to ``new_profile`` requires reindexing.

    Any change that alters the vector space geometry (provider, model,
    dimension, normalization, distance metric, or service URL) requires
    a new index generation.  Metadata-only changes (e.g. renaming
    ``provider_id`` to an alias that produces identical vectors) are not
    detected here and must be handled explicitly by the operator.

    Args:
        old_profile: Profile active on the existing index.
        new_profile: Profile being evaluated for adoption.

    Returns:
        ``True`` when the active index cannot serve the new profile.
    """
    return old_profile.canonical_hash() != new_profile.canonical_hash()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def validate_profile_at_startup(
    profile: EmbeddingProfile,
    *,
    strict: bool = False,
) -> None:
    """Validate ``profile`` at application startup.

    Warns about unknown dimensions in lenient mode (the default);
    raises :class:`ProfileValidationError` in strict mode.  The
    legacy :data:`~ragnor.embeddings.dimensions.FALLBACK_DIMENSION`
    is not treated as an error unless the profile dimension diverges
    from it *and* the model is known.

    Args:
        profile: Profile to validate.
        strict: When ``True``, unknown dimensions or mismatched
            registered dimensions raise instead of warning.

    Raises:
        ProfileValidationError: If ``strict=True`` and validation fails.
    """
    registered = _DIMENSIONS.get(profile.model_id)

    if registered is None:
        msg = (
            f"No dimension registered for model '{profile.model_id}'; "
            f"profile declares dimension={profile.dimension}. "
            f"Register it with ragnor.embeddings.dimensions.register_dimension() "
            f"if this is wrong (falling back to FALLBACK_DIMENSION={FALLBACK_DIMENSION} "
            f"is unsafe in production)."
        )
        if strict:
            raise ProfileValidationError(msg)
        warnings.warn(msg, stacklevel=2)
        return

    if registered != profile.dimension:
        msg = (
            f"Profile dimension {profile.dimension} does not match the registered "
            f"dimension {registered} for model '{profile.model_id}'. "
            f"This will cause vector dimension mismatches at insert time."
        )
        if strict:
            raise ProfileValidationError(msg)
        warnings.warn(msg, stacklevel=2)

    if profile.dimension <= 0:
        raise ProfileValidationError(
            f"Embedding dimension must be positive, got {profile.dimension}."
        )
