"""Production errors, emitted logs, and SDK repr cannot disclose host secrets."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.retrieval.capabilities import BackendFeature, FilterOperator, StoreCapabilities

from tests.production_doubles import StoragePort, VersionPort

SECRET = "ragnor_private_provider_api_key_do_not_log_9911"


class PrivateStore(StoragePort):
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
        version_store=VersionPort(),
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
        version_store=VersionPort(),
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


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("boundary", ["retrieval", "shared_count"])
def test_error_boundaries_do_not_log_raw_provider_errors(caplog, strict, boundary) -> None:
    from ragnor.exceptions import BackendUnavailableError
    from ragnor.retrieval.strategies import _handle_retrieval_exception
    from ragnor.stores._scoring import handle_count_error

    error = ConnectionError(SECRET + " unlabelled customer content")
    with caplog.at_level(logging.DEBUG):
        try:
            if boundary == "retrieval":
                result = _handle_retrieval_exception(error, "dense", strict)
                assert result == []
            else:
                assert handle_count_error("chunks", error, strict, logging.getLogger(__name__)) == 0
        except BackendUnavailableError as raised:
            assert strict
            assert SECRET not in str(raised)
            assert raised.__cause__ is error
        else:
            assert not strict
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count", "hybrid_search", "upsert"])
async def test_pgvector_errors_keep_credentials_out_of_logs_and_domain_messages(caplog, operation):
    from ragnor.exceptions import BackendUnavailableError
    from ragnor.stores.pgvector import PgVectorStore

    error = ConnectionError(SECRET)

    class FailedSQL:
        async def fetch_scalar(self, *args, **kwargs):
            raise error

        async def fetch_all(self, *args, **kwargs):
            raise error

        async def execute(self, *args, **kwargs):
            raise error

    store = PgVectorStore(table_name="chunks", embedding_dim=2, sql_factory=FailedSQL())
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises((BackendUnavailableError, ConnectionError)) as caught,
    ):
        if operation == "count":
            await store.count(session=object(), strict=True)
        elif operation == "hybrid_search":
            await store.hybrid_search(
                "private query",
                [1.0, 0.0],
                n_results=5,
                threshold=1.3,
                session=object(),
                strict=True,
            )
        else:
            await store.upsert(
                [{"id": "a", "text": "private", "embedding": [1.0, 0.0]}], session=object()
            )
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    if isinstance(caught.value, BackendUnavailableError):
        assert SECRET not in str(caught.value)
        assert caught.value.__cause__ is error


@pytest.mark.parametrize("status", [None, 400, 401, 403, 429, 503])
def test_provider_classification_does_not_echo_unlabelled_secret_or_customer_content(status):
    from ragnor.exceptions import classify_exception

    class ProviderError(Exception):
        status_code = status

    source = ProviderError(SECRET + " confidential customer document")
    error = classify_exception(source, default_stage="embedding")
    assert SECRET not in str(error)
    assert "confidential customer" not in str(error)
    assert error.__cause__ is source
    if status is not None:
        assert error.details["status_code"] == status
