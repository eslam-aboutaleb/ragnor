"""PostgreSQL advisory lock helper for ingestion.

Compatibility wrapper: the lock now lives on the abstract
SQL factory (:class:`ragit.db.sql_factory.SQLFactory`),
and :class:`ragit.ingestion.pipeline.IngestionPipeline`
takes a ``sql_factory`` argument. This module-level helper
delegates to the default
:class:`SQLAlchemyFactory` for existing callers.

Ordering contract (preserved verbatim from the
original code):

* the lock is taken on the session the pipeline
  transacts on;
* on failure the transaction is rolled back *before*
  the lock is released -- a failed statement leaves the
  session in an aborted transaction, and any further
  command on it raises ``InFailedSQLTransactionError``,
  which would replace the real ingestion failure with a
  confusing error about the advisory lock;
* a lock-cleanup failure must never mask the outcome:
  ending the transaction closes the connection, which
  releases a session-level advisory lock anyway, so the
  cleanup failure is only logged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ragit.db.sql_factory import SQLAlchemyFactory


@asynccontextmanager
async def advisory_lock(session: Any, key: str) -> AsyncIterator[None]:
    """Hold a PostgreSQL advisory lock keyed by ``key``.

    Delegates to :meth:`SQLAlchemyFactory.advisory_lock`.
    """
    async with SQLAlchemyFactory().advisory_lock(session, key):
        yield
