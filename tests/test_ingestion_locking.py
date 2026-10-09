"""Tests for ragit.ingestion.locking.

Pins the advisory-lock contract extracted from
``ingest_policy``: lock/unlock SQL on the caller's
session, rollback-before-unlock on failure, and
lock-cleanup failures that must not mask the
outcome.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from ragit.ingestion import advisory_lock


class _ScriptedSession:
    """Session that records executed SQL and scripted failures."""

    def __init__(self, unlock_error: Exception | None = None) -> None:
        self.executed: list[str] = []
        self.events: list[str] = []
        self._unlock_error = unlock_error

    async def execute(self, statement: Any, params: Any = None) -> MagicMock:
        sql = str(statement)
        self.executed.append(sql)
        if "pg_advisory_lock" in sql:
            self.events.append("lock")
        if "pg_advisory_unlock" in sql:
            self.events.append("unlock")
            if self._unlock_error is not None:
                raise self._unlock_error
        return MagicMock()

    async def rollback(self) -> None:
        self.events.append("rollback")

    async def commit(self) -> None:
        self.events.append("commit")


@pytest.mark.asyncio
async def test_advisory_lock_locks_and_unlocks() -> None:
    session = _ScriptedSession()

    async with advisory_lock(session, "source-path"):
        assert session.events == ["lock"]

    assert session.events == ["lock", "unlock"]
    assert any("pg_advisory_lock" in sql for sql in session.executed)
    assert any("pg_advisory_unlock" in sql for sql in session.executed)


@pytest.mark.asyncio
async def test_advisory_lock_rolls_back_before_unlock_on_error() -> None:
    session = _ScriptedSession()

    with pytest.raises(RuntimeError, match="boom"):
        async with advisory_lock(session, "source-path"):
            raise RuntimeError("boom")

    # The rollback must run before the unlock: a failed statement
    # leaves the session in an aborted transaction, and the unlock
    # would otherwise raise InFailedSQLTransactionError.
    assert session.events == ["lock", "rollback", "unlock"]


@pytest.mark.asyncio
async def test_advisory_lock_cleanup_failure_does_not_mask_outcome() -> None:
    session = _ScriptedSession(unlock_error=SQLAlchemyError("cleanup failed"))

    # The block's outcome (normal exit here) must surface, not the
    # lock-cleanup failure: ending the transaction closes the
    # connection, which releases a session-level advisory lock anyway.
    async with advisory_lock(session, "source-path"):
        pass

    assert session.events == ["lock", "unlock"]


@pytest.mark.asyncio
async def test_advisory_lock_cleanup_failure_does_not_mask_error() -> None:
    session = _ScriptedSession(unlock_error=SQLAlchemyError("cleanup failed"))

    with pytest.raises(RuntimeError, match="boom"):
        async with advisory_lock(session, "source-path"):
            raise RuntimeError("boom")

    assert session.events == ["lock", "rollback", "unlock"]
