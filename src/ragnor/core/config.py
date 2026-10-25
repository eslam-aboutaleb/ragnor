"""Configuration protocols and validated config models for ragnor.

ragnor reads its configuration through the :class:`RagSettings`
protocol. Any settings object (typically a pydantic ``Settings``
class in the host application, or :class:`RagnorConfig`) that exposes
these attributes works structurally, so ragnor never depends on the
host's config module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ragnor.core.exceptions import ConfigurationError, sanitize_message as _sanitize_message


def sanitize_message(message: str) -> str:
    """Keep the historical config-module redaction import available."""
    return _sanitize_message(message)


class DeploymentMode(StrEnum):
    """Target deployment mode for the Ragnor SDK."""

    LOCAL = "local"
    PRODUCTION = "production"


@runtime_checkable
class RagSettings(Protocol):
    """Settings that ragnor reads at runtime."""

    embedding_model: str
    rag_distance_threshold: float
    vector_store_provider: str
    openai_api_key: str
    openai_api_base: str
    embedding_drain_interval_seconds: float
    embedding_drain_batch_size: int
    job_stale_after_seconds: float
    worker_id: str
    ingest_on_startup: bool


@runtime_checkable
class SettingsProvider(Protocol):
    """Accessor protocol for settings (a ``get_settings()``-style factory)."""

    def get_settings(self) -> RagSettings:
        """Return settings supplied by the host application."""
        raise NotImplementedError


@dataclass
class RagnorConfig:
    """Validated stdlib configuration satisfying :class:`RagSettings`.

    Redacts credentials and connection URLs in string/repr output and
    validates operational bounds deterministically.
    """

    embedding_model: str = "local"
    rag_distance_threshold: float = 1.3
    vector_store_provider: str = "memory"
    openai_api_key: str = ""
    openai_api_base: str = ""
    embedding_drain_interval_seconds: float = 5.0
    embedding_drain_batch_size: int = 10
    job_stale_after_seconds: float = 300.0
    worker_id: str = "ragnor-worker"
    ingest_on_startup: bool = False
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    embedding_dim: int | None = None

    def validate(self) -> None:
        """Validate configuration bounds, raising ConfigurationError on invalid fields."""
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ConfigurationError("embedding_model must be a non-empty string")
        if self.embedding_dim is not None and (
            not isinstance(self.embedding_dim, int)
            or isinstance(self.embedding_dim, bool)
            or self.embedding_dim <= 0
        ):
            raise ConfigurationError("embedding_dim must be a positive integer")
        if self.rag_distance_threshold <= 0:
            raise ConfigurationError("rag_distance_threshold must be positive")
        if self.embedding_drain_interval_seconds <= 0:
            raise ConfigurationError("embedding_drain_interval_seconds must be positive")
        if (
            not isinstance(self.embedding_drain_batch_size, int)
            or isinstance(self.embedding_drain_batch_size, bool)
            or self.embedding_drain_batch_size <= 0
        ):
            raise ConfigurationError("embedding_drain_batch_size must be a positive integer")
        if self.job_stale_after_seconds <= 0:
            raise ConfigurationError("job_stale_after_seconds must be positive")
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ConfigurationError("worker_id must be a non-empty string")

    def __repr__(self) -> str:
        masked_key = "***" if self.openai_api_key else ""
        return (
            f"RagnorConfig(embedding_model={self.embedding_model!r}, "
            f"rag_distance_threshold={self.rag_distance_threshold!r}, "
            f"vector_store_provider={self.vector_store_provider!r}, "
            f"openai_api_key={masked_key!r}, "
            f"deployment_mode={self.deployment_mode!r}, "
            f"embedding_dim={self.embedding_dim!r})"
        )
