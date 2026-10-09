"""Abstract SQL factory -- the seam between ragit and any SQL database.

Every SQL operation ragit performs -- session creation, queries,
mutations, and advisory locking -- goes through the
:class:`SQLFactory` interface. The built-in
:class:`SQLAlchemyFactory` implements it with SQLAlchemy, which
works with any SQLAlchemy dialect (PostgreSQL, SQLite, MySQL, ...).
Hosts targeting another stack subclass :class:`SQLFactory` and
inject their factory into the ingestion pipeline and stores.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from ragit.db.session import SessionProvider

logger = logging.getLogger(__name__)


class SQLFactory(ABC):
    """Abstract factory for every SQL operation ragit performs.

    Subclass this to route ragit's SQL through a different
    stack, and inject the instance via ``sql_factory=...`` on
    :class:`ragit.ingestion.pipeline.IngestionPipeline` and
    :class:`ragit.stores.pgvector.PgVectorStore`.
    """

    @abstractmethod
    def create_session_factory(self, database_url: str) -> SessionProvider:
        """Build a session factory bound to ``database_url``.

        Args:
            database_url: Database URL understood by the
                underlying driver stack.

        Returns:
            A callable yielding async session context managers.
        """

    @abstractmethod
    async def fetch_all(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a query and return every row as a dictionary.

        Args:
            session: Database session to run the statement on.
            statement: SQL statement with ``:named`` bind params.
            params: Bind parameter values.

        Returns:
            All result rows, keyed by column name.
        """

    @abstractmethod
    async def fetch_scalar(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Run a query and return its single scalar result.

        Args:
            session: Database session to run the statement on.
            statement: SQL statement with ``:named`` bind params.
            params: Bind parameter values.

        Returns:
            The first column of the first row.
        """

    @abstractmethod
    async def execute(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Run a mutation statement (INSERT/UPDATE/DELETE).

        Args:
            session: Database session to run the statement on.
            statement: SQL statement with ``:named`` bind params.
            params: Bind parameter values.
        """

    @abstractmethod
    def advisory_lock(self, session: Any, key: str) -> AsyncIterator[None]:
        """Hold a database-level lock that serializes ``key``.

        Args:
            session: Database session the lock is taken on.
            key: Lock key (the source path).

        Yields:
            Nothing. The lock is held for the duration of the
            ``async with`` block.
        """


class SQLAlchemyFactory(SQLFactory):
    """SQL factory backed by SQLAlchemy -- works with any SQLAlchemy dialect."""

    def create_session_factory(self, database_url: str) -> SessionProvider:
        """Build a session factory bound to a new async engine."""
        engine = create_async_engine(database_url)
        return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def fetch_all(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a query and return every row as a dictionary."""
        result = await session.execute(text(statement), params)
        return result.mappings().all()

    async def fetch_scalar(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Run a query and return its single scalar result."""
        result = await session.execute(text(statement), params)
        return result.scalar_one()

    async def execute(
        self,
        session: Any,
        statement: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Run a mutation statement (INSERT/UPDATE/DELETE)."""
        await session.execute(text(statement), params)

    @asynccontextmanager
    async def advisory_lock(self, session: Any, key: str) -> AsyncIterator[None]:
        """Hold a PostgreSQL advisory lock keyed by ``key``.

        The lock is session-level and taken with
        ``pg_advisory_lock(hashtext(:key))``, so concurrent
        ingestion of the same source -- across instances or
        tasks -- is serialized on the database. Hosts on another
        database subclass :class:`SQLFactory` and override this
        method with their locking primitive.

        Ordering contract:

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
        await session.execute(
            text("SELECT pg_advisory_lock(hashtext(:source))"),
            {"source": key},
        )
        try:
            yield
        except Exception:
            await session.rollback()
            raise
        finally:
            try:
                await session.execute(
                    text("SELECT pg_advisory_unlock(hashtext(:source))"),
                    {"source": key},
                )
            except SQLAlchemyError:
                logger.warning(
                    "Could not explicitly release the advisory lock for '%s'; "
                    "releasing it by closing the session instead.",
                    key,
                )
