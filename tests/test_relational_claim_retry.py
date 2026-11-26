"""Claim retry classification: transient SQLSTATEs retry, permanent ones raise."""

from __future__ import annotations

from sqlalchemy.exc import DBAPIError, OperationalError

from ragnor.jobs.adapters.relational import _is_retryable_claim_error


class _ServerError(Exception):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


def _dbapi(sqlstate: str) -> DBAPIError:
    return DBAPIError("UPDATE ragnor_embedding_jobs", {}, _ServerError(sqlstate))


def test_driver_operational_error_is_retryable() -> None:
    error = OperationalError("UPDATE ragnor_embedding_jobs", {}, Exception("database is locked"))
    assert _is_retryable_claim_error(error)


def test_transient_sqlstates_are_retryable() -> None:
    for sqlstate in ("40001", "40P01", "55P03", "57014", "08006"):
        assert _is_retryable_claim_error(_dbapi(sqlstate))


def test_permanent_sqlstates_are_not_retried() -> None:
    for sqlstate in ("23505", "42601", "42P01"):
        assert not _is_retryable_claim_error(_dbapi(sqlstate))


def test_error_without_sqlstate_is_not_retried() -> None:
    error = DBAPIError("UPDATE ragnor_embedding_jobs", {}, Exception("boom"))
    assert not _is_retryable_claim_error(error)


def test_non_database_errors_are_not_retried() -> None:
    assert not _is_retryable_claim_error(ValueError("not a database error"))
