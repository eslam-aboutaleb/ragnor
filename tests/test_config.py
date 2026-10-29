"""Structural check that a settings object satisfies RagSettings.

The standalone library ships no settings class of its own: host
applications pass their own settings object (typically a pydantic
``Settings``) and ragnor reads it through the :class:`RagSettings`
protocol. This test pins the protocol's required attribute set so a
settings class that drifts away from the protocol fails here instead
of at runtime inside a store or retriever.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragnor.config import RagSettings, SettingsProvider


class _MinimalSettings:
    """The smallest object that structurally satisfies ``RagSettings``."""

    embedding_model: str = "text-embedding-3-small"
    rag_distance_threshold: float = 0.5
    vector_store_provider: str = "memory"
    openai_api_key: str = "test-key"
    openai_api_base: str = ""
    embedding_drain_interval_seconds: float = 5.0
    embedding_drain_batch_size: int = 10
    job_stale_after_seconds: float = 300.0
    worker_id: str = "test-worker"
    ingest_on_startup: bool = False


class _MinimalProvider:
    """The smallest object that structurally satisfies ``SettingsProvider``."""

    def get_settings(self) -> _MinimalSettings:
        return _MinimalSettings()


def test_minimal_settings_satisfy_the_rag_settings_protocol() -> None:
    assert isinstance(_MinimalSettings(), RagSettings)


def test_rag_settings_is_runtime_checkable() -> None:
    """The protocol is decorated with ``runtime_checkable`` so
    ``isinstance`` checks work for host applications that want to
    validate their settings at start-up."""
    assert RagSettings._is_runtime_protocol  # type: ignore[attr-defined]


def test_settings_provider_satisfies_its_protocol() -> None:
    assert isinstance(_MinimalProvider(), SettingsProvider)


def test_settings_provider_returns_rag_settings() -> None:
    provider = _MinimalProvider()
    assert isinstance(provider.get_settings(), RagSettings)


def test_rag_settings_declares_every_required_field() -> None:
    """The protocol's attribute set is exactly the documented one.

    Adding a field to ``RagSettings`` is a breaking change for every
    host application; this test makes the field set explicit so the
    change is deliberate.
    """
    required = {
        "embedding_model",
        "rag_distance_threshold",
        "vector_store_provider",
        "openai_api_key",
        "openai_api_base",
        "embedding_drain_interval_seconds",
        "embedding_drain_batch_size",
        "job_stale_after_seconds",
        "worker_id",
        "ingest_on_startup",
    }
    # ``__protocol_attrs__`` exists on Python 3.12+; on 3.11 the
    # protocol's declared annotations carry the same attribute set.
    declared = getattr(RagSettings, "__protocol_attrs__", None)
    if declared is None:
        declared = RagSettings.__annotations__
    assert set(declared) == required


def test_a_pydantic_style_settings_class_satisfies_the_protocol() -> None:
    """Host applications typically use pydantic ``BaseSettings``.

    A plain class with the same attributes is structurally identical
    for the protocol's purposes; this pins that duck typing works
    without ragnor importing pydantic at all.
    """

    @runtime_checkable
    class _PydanticLike(Protocol):
        embedding_model: str

    class _HostSettings:
        embedding_model = "text-embedding-3-small"
        rag_distance_threshold = 0.5
        vector_store_provider = "pgvector"
        openai_api_key = "key"
        openai_api_base = ""
        embedding_drain_interval_seconds = 5.0
        embedding_drain_batch_size = 10
        job_stale_after_seconds = 300.0
        worker_id = "worker-1"
        ingest_on_startup = False

    assert isinstance(_HostSettings(), _PydanticLike)
    assert isinstance(_HostSettings(), RagSettings)
