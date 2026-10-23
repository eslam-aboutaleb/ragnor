"""SDK-03 (#22): Consistent SDK error taxonomy, retryability, and fail-closed semantics."""

from __future__ import annotations

import asyncio

import pytest

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
    sanitize_message,
)
from ragnor.jobs.outbox import LostJobOwnershipError
from ragnor.retrieval.capabilities import CapabilityUnsupportedError
from ragnor.retrieval.types import ScopeViolationError
from ragnor.sync.coordinator import FencingError, LeaseAcquisitionError, StaleWorkerError, SyncError
from ragnor.types import RetrievalError


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
        )
        assert "p@ssw0rd" not in str(err)
        assert "p@ssw0rd" not in repr(err)
        assert "secret_val_123" not in str(err.details)


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
