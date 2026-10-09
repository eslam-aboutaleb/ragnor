"""Configuration protocols for ragit.

ragit reads its configuration through the :class:`RagSettings`
protocol. Any settings object (typically a pydantic ``Settings``
class in the host application) that exposes these attributes works
structurally, so ragit never depends on the host's config module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RagSettings(Protocol):
    """Settings that ragit reads at runtime."""

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
    database_url: str


@runtime_checkable
class SettingsProvider(Protocol):
    """Accessor protocol for settings (a ``get_settings()``-style factory)."""

    def get_settings(self) -> RagSettings: ...
