"""Production errors, emitted logs, and SDK repr cannot disclose host secrets."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.capabilities import BackendFeature, FilterOperator, StoreCapabilities

SECRET = "ragnor_private_provider_api_key_do_not_log_9911"


class PrivateStore:
    capabilities = StoreCapabilities(
        filter_operators=frozenset({FilterOperator.EQ}),
        features=frozenset({BackendFeature.ISOLATION_SCOPE}),
    )
    embedding_dim = 2

    def __init__(self) -> None:
        self.api_key = SECRET
        self.write_calls = 0
        self.close_calls = 0

    def __repr__(self) -> str:
        return "PrivateStore(api_key=" + self.api_key + ")"

    async def upsert(self, documents, **kwargs) -> None:
        self.write_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1


class Embedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@asynccontextmanager
async def sessions():
    yield object()


@pytest.mark.asyncio
async def test_failed_production_validation_does_not_include_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = PrivateStore()
    builder = RagnorBuilder(
        embedder=Embedder(),
        embedding_model="safe-model",
        embedding_dim=3,  # incompatible with injected store
        store=store,
        version_store=object(),
        session_factory=sessions,
        production_mode=True,
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as caught:
        await builder.build()
    assert store.write_calls == 0
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(builder)
    assert SECRET not in caplog.text


@pytest.mark.asyncio
async def test_host_owned_secret_client_survives_facade_shutdown() -> None:
    store = PrivateStore()
    rag = await RagnorBuilder(
        embedder=Embedder(),
        embedding_model="safe-model",
        embedding_dim=2,
        store=store,
        version_store=object(),
        session_factory=sessions,
        production_mode=True,
    ).build()
    await rag.aclose()
    assert store.close_calls == 0


def test_error_details_redact_sensitive_keys_recursively_without_mutating_host_data() -> None:
    from ragnor.exceptions import RagnorError

    details = {
        "api_key": SECRET,
        "attempt": 2,
        "provider": {"password": SECRET, "region": "local"},
        "events": [{"prompt_content": SECRET, "status": 503}],
        "authorization": "Bearer " + SECRET,
    }
    error = RagnorError("provider failed", details=details)
    assert SECRET not in str(error.details)
    assert error.details["attempt"] == 2
    assert error.details["provider"]["region"] == "local"
    assert error.details["events"][0]["status"] == 503
    assert details["api_key"] == SECRET
    assert details["provider"]["password"] == SECRET
