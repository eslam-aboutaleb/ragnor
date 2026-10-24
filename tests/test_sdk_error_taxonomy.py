"""SDK-03 (#22): Consistent SDK error taxonomy, retryability, and fail-closed semantics."""

from __future__ import annotations

import asyncio

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.embeddings.executor import (
    EmbeddingError,
    EmbeddingPermanentError,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingTransientError,
)
from ragnor.evidence.types import RevisionUnavailableError, ScopeAuthorizationError
from ragnor.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    IndexConsistencyError,
    OperationTimeoutError,
    RagnorError,
    ScopeDeniedError,
    UnsupportedCapabilityError,
    classify_exception,
    is_retryable_exception,
    sanitize_message,
)
from ragnor.jobs.outbox import (
    DefaultRetryPolicy,
    InMemoryEmbeddingJobStore,
    JobPayload,
    LostJobOwnershipError,
    TaxonomyRetryPolicy,
    process_pending_jobs,
)
from ragnor.retrieval.capabilities import CapabilityUnsupportedError
from ragnor.retrieval.types import ScopeViolationError
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.sync.coordinator import FencingError, LeaseAcquisitionError, StaleWorkerError, SyncError
from ragnor.types import Document, RetrievalError


class TestErrorTaxonomyHierarchy:
    """Validate machine-readable error codes, defaults, and typing hierarchy."""

    def test_ragnor_error_base_properties(self) -> None:
        err = RagnorError(
            "Generic error", code="ERR_CUSTOM", retryable=True, details={"host": "node-1"}
        )
        assert str(err) == "Generic error"
        assert err.code == "ERR_CUSTOM"
        assert err.retryable is True
        assert err.details == {"host": "node-1"}
        assert "ERR_CUSTOM" in repr(err)

    def test_default_codes_and_retryability(self) -> None:
        cases = [
            (ConfigurationError("bad config"), "ERR_CONFIG_INVALID", False),
            (BackendUnavailableError("store down"), "ERR_BACKEND_UNAVAILABLE", True),
            (EmbeddingProviderError("embed failed"), "ERR_EMBEDDING_FAILED", False),
            (IndexConsistencyError("version mismatch"), "ERR_INDEX_INCONSISTENT", False),
            (UnsupportedCapabilityError("no rrf"), "ERR_UNSUPPORTED_CAPABILITY", False),
            (ScopeDeniedError("tenant forbidden"), "ERR_SCOPE_DENIED", False),
            (OperationTimeoutError("query timed out"), "ERR_OPERATION_TIMEOUT", True),
        ]
        for err, expected_code, expected_retryable in cases:
            assert isinstance(err, RagnorError)
            assert err.code == expected_code
            assert err.retryable is expected_retryable


class TestErrorSanitization:
    """Ensure secrets, tokens, credentials, and DSN passwords are redacted."""

    def test_sanitize_message_dsn(self) -> None:
        raw = (
            "Failed to connect to postgresql://postgres:my_secret_password@db.example.com:5432/prod"
        )
        sanitized = sanitize_message(raw)
        assert "my_secret_password" not in sanitized
        assert "postgresql://postgres:***@db.example.com:5432/prod" in sanitized

    def test_sanitize_message_api_keys_and_tokens(self) -> None:
        raw = "Auth failed for api-key: sk-proj-1234567890abcdef and Bearer eyJhbGciOiJIUzI1Ni"
        sanitized = sanitize_message(raw)
        assert "sk-proj-1234567890abcdef" not in sanitized
        assert "api-key=***" in sanitized or "api-key:***" in sanitized

    def test_exception_redacts_credentials_in_message_and_details(self) -> None:
        err = ConfigurationError(
            "Invalid DSN postgresql://admin:p@ssw0rd@127.0.0.1:5432/db",
            details={"key": "api_key=secret_val_123"},
            trace_id="trace-xyz-999",
        )
        assert "p@ssw0rd" not in str(err)
        assert "p@ssw0rd" not in repr(err)
        assert "secret_val_123" not in str(err.details)
        assert err.trace_id == "trace-xyz-999"
        assert err.details.get("trace_id") == "trace-xyz-999"

    def test_sanitize_message_redacts_document_snippets_and_bearer_headers(self) -> None:
        raw = (
            "Embedding failed: Authorization: Bearer eyJhbGciOiJIUzI1Ni.secret "
            "document_text='Confidential SSN 999-00-1111'"
        )
        sanitized = sanitize_message(raw)
        assert "eyJhbGciOiJIUzI1Ni.secret" not in sanitized
        assert "Confidential SSN 999-00-1111" not in sanitized
        assert "document_text=***" in sanitized


class TestExceptionClassificationAndRetryability:
    """Validate classify_exception and is_retryable_exception across failure modes."""

    def test_classify_exception_maps_transient_and_permanent_failures(self) -> None:
        # 1. Existing RagnorError is idempotent (and can attach trace_id)
        existing = ConfigurationError("already typed")
        classified_existing = classify_exception(existing, trace_id="tr-1")
        assert classified_existing is existing
        assert classified_existing.trace_id == "tr-1"

        # 2. TimeoutError -> OperationTimeoutError (retryable=True)
        timeout_err = classify_exception(TimeoutError("deadline exceeded"), trace_id="tr-2")
        assert isinstance(timeout_err, OperationTimeoutError)
        assert timeout_err.retryable is True
        assert timeout_err.__cause__ is not None

        # 3. PermissionError -> ScopeDeniedError (retryable=False)
        perm_err = classify_exception(PermissionError("cross-tenant access denied"))
        assert isinstance(perm_err, ScopeDeniedError)
        assert perm_err.retryable is False

        # 4. ConnectionError / connection refused -> BackendUnavailableError (retryable=True)
        conn_err = classify_exception(ConnectionRefusedError("Connection refused on port 6333"))
        assert isinstance(conn_err, BackendUnavailableError)
        assert conn_err.retryable is True

        # 5. Embedding 429 / 503 -> EmbeddingProviderError (retryable=True)
        rate_err = classify_exception(
            RuntimeError("HTTP 429 Too Many Requests from embedding API"),
            default_stage="embedding",
        )
        assert isinstance(rate_err, EmbeddingProviderError)
        assert rate_err.retryable is True

        # 6. Embedding 401 / 403 / 400 -> EmbeddingProviderError (retryable=False)
        auth_err = classify_exception(
            RuntimeError("HTTP 401 Unauthorized: api_key=sk-live-secret123456"),
            default_stage="embedding",
        )
        assert isinstance(auth_err, EmbeddingProviderError)
        assert auth_err.retryable is False
        assert "sk-live-secret123456" not in str(auth_err)

        # 7. ValueError -> ConfigurationError (retryable=False)
        val_err = classify_exception(ValueError("embedding_dim must be > 0"))
        assert isinstance(val_err, ConfigurationError)
        assert val_err.retryable is False

    def test_is_retryable_exception_helper(self) -> None:
        assert is_retryable_exception(BackendUnavailableError("down")) is True
        assert is_retryable_exception(OperationTimeoutError("slow")) is True
        assert is_retryable_exception(ConfigurationError("bad")) is False
        assert is_retryable_exception(ScopeDeniedError("forbidden")) is False
        assert is_retryable_exception(TimeoutError("os timeout")) is True
        assert is_retryable_exception(ConnectionError("reset")) is True
        assert is_retryable_exception(ValueError("invalid")) is False


class TestBackwardCompatibilityAndCatching:
    """Verify that existing catch blocks and subsystem errors remain 100% compatible."""

    def test_retrieval_error_subclasses(self) -> None:
        retrieval_err = RetrievalError("Search failed")
        assert isinstance(retrieval_err, RagnorError)
        assert isinstance(retrieval_err, Exception)

        scope_err = ScopeViolationError("Cross-tenant leakage detected")
        assert isinstance(scope_err, ScopeViolationError)
        assert isinstance(scope_err, ScopeDeniedError)
        assert isinstance(scope_err, RetrievalError)
        assert isinstance(scope_err, RagnorError)

        cap_err = CapabilityUnsupportedError("Backend lacks hybrid scoring")
        assert isinstance(cap_err, CapabilityUnsupportedError)
        assert isinstance(cap_err, UnsupportedCapabilityError)
        assert isinstance(cap_err, RetrievalError)
        assert isinstance(cap_err, RagnorError)

        rev_err = RevisionUnavailableError("Historical revision retired")
        assert isinstance(rev_err, RevisionUnavailableError)
        assert isinstance(rev_err, IndexConsistencyError)
        assert isinstance(rev_err, RetrievalError)

        scope_auth_err = ScopeAuthorizationError("Scope unauthorized")
        assert isinstance(scope_auth_err, ScopeAuthorizationError)
        assert isinstance(scope_auth_err, ScopeDeniedError)
        assert isinstance(scope_auth_err, RetrievalError)

    def test_embedding_error_subclasses(self) -> None:
        transient = EmbeddingTransientError("Rate limit exceeded", retry_after=2.0)
        assert isinstance(transient, EmbeddingTransientError)
        assert isinstance(transient, EmbeddingError)
        assert isinstance(transient, EmbeddingProviderError)
        assert isinstance(transient, RagnorError)
        assert transient.retryable is True
        assert transient.retry_after == 2.0

        permanent = EmbeddingPermanentError("Invalid API key")
        assert isinstance(permanent, EmbeddingPermanentError)
        assert isinstance(permanent, EmbeddingError)
        assert isinstance(permanent, EmbeddingProviderError)
        assert permanent.retryable is False

        response_err = EmbeddingResponseError("Dimension mismatch")
        assert isinstance(response_err, EmbeddingResponseError)
        assert isinstance(response_err, EmbeddingPermanentError)
        assert isinstance(response_err, EmbeddingError)

    def test_sync_and_jobs_subclasses(self) -> None:
        sync_err = SyncError("Sync aborted")
        assert isinstance(sync_err, RagnorError)

        lease_err = LeaseAcquisitionError("Lease unavailable")
        assert isinstance(lease_err, SyncError)
        assert isinstance(lease_err, RagnorError)

        stale_err = StaleWorkerError("Worker token expired")
        assert isinstance(stale_err, StaleWorkerError)
        assert isinstance(stale_err, IndexConsistencyError)
        assert isinstance(stale_err, SyncError)

        fence_err = FencingError("Fencing check failed")
        assert isinstance(fence_err, FencingError)
        assert isinstance(fence_err, IndexConsistencyError)
        assert isinstance(fence_err, SyncError)

        lost_job = LostJobOwnershipError("Stale generation lease lost")
        assert isinstance(lost_job, LostJobOwnershipError)
        assert isinstance(lost_job, IndexConsistencyError)
        assert isinstance(lost_job, RuntimeError)
        assert isinstance(lost_job, RagnorError)


class TestFailClosedSemantics:
    """Verify cancellation preservation and cause chaining."""

    def test_cancellation_is_not_intercepted_or_swallowed(self) -> None:
        async def cancel_me() -> None:
            raise asyncio.CancelledError("task cancelled")

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cancel_me())

    def test_cause_chaining_preserved(self) -> None:
        original = ConnectionError("socket connection reset")
        try:
            try:
                raise original
            except ConnectionError as exc:
                raise BackendUnavailableError("Vector database cluster unreachable") from exc
        except BackendUnavailableError as ragnor_exc:
            assert ragnor_exc.__cause__ is original
            assert isinstance(ragnor_exc, RagnorError)
            assert ragnor_exc.retryable is True

    def test_cancellation_propagates_through_search_and_ingest(self) -> None:
        class CancellingEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                raise asyncio.CancelledError("caller cancelled during embedding")

        async def _run() -> None:
            client = await RagnorBuilder.local(
                embedder=CancellingEmbedder(),
                embedding_dim=4,
            ).build()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await client.ingest([Document(id="d1", text="hello")])
                with pytest.raises(asyncio.CancelledError):
                    await client.search("hello")
            finally:
                await client.aclose()

        asyncio.run(_run())

    def test_production_search_distinguishes_empty_matches_from_backend_outage(self) -> None:
        class HealthyEmbedder:
            async def __call__(self, texts: list[str]) -> list[list[float]]:
                return [[0.25, 0.25, 0.25, 0.25] for _ in texts]

        class OutageVectorStore(InMemoryVectorStore):
            def __init__(self) -> None:
                super().__init__(embedding_dim=4)
                self.simulate_outage = False

            async def vector_search(
                self, *args: object, **kwargs: object
            ) -> list[dict[str, object]]:  # type: ignore[override]
                if self.simulate_outage:
                    raise ConnectionError("Qdrant cluster unreachable on 10.0.0.5:6333")
                return await super().vector_search(*args, **kwargs)  # type: ignore[arg-type]

        async def _run() -> None:
            store = OutageVectorStore()
            client = await RagnorBuilder.local(
                HealthyEmbedder(),
                embedding_dim=4,
                store=store,
            ).build()
            # Enable production fail-closed semantics on the built instance
            client._production_mode = True
            try:
                # 0 matching documents -> returns empty list []
                empty = await client.search("query with zero matches")
                assert empty == []

                # Backend outage -> raises BackendUnavailableError (never silent [])
                store.simulate_outage = True
                with pytest.raises(BackendUnavailableError) as exc_info:
                    await client.search("query during outage")
                assert exc_info.value.retryable is True
                assert isinstance(exc_info.value.__cause__, ConnectionError)
            finally:
                await client.aclose()

        asyncio.run(_run())

    def test_outbox_worker_dead_letters_permanent_error_on_first_attempt_and_sanitizes(
        self,
    ) -> None:
        async def _run() -> None:
            job_store = InMemoryEmbeddingJobStore()
            await job_store.enqueue(
                {
                    "idempotency_key": "job-perm-1",
                    "document_id": "doc-1",
                    "revision_id": "rev-1",
                    "max_attempts": 5,
                }
            )

            attempts = 0

            class FailingProcessor:
                async def process(self, claimed_job: JobPayload) -> None:
                    nonlocal attempts
                    attempts += 1
                    raise EmbeddingPermanentError(
                        "HTTP 401 Invalid API key api_key=sk-secret-do-not-leak-99999"
                    )

            processed = await process_pending_jobs(
                store=job_store,
                processor=FailingProcessor(),
                worker_id="worker-1",
                retry_policy=DefaultRetryPolicy(),
            )
            assert processed == 0
            assert attempts == 1

            stored_job = job_store.jobs[0]
            assert stored_job.status == "dead_letter"
            assert stored_job.status_detail is not None
            assert "sk-secret-do-not-leak-99999" not in stored_job.status_detail

            # Also verify TaxonomyRetryPolicy behavior directly
            policy = TaxonomyRetryPolicy()
            assert policy.delay(EmbeddingPermanentError("bad key"), 1, 5) is None
            assert policy.delay(EmbeddingTransientError("rate limit"), 1, 5) is not None

        asyncio.run(_run())
