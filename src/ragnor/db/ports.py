"""Vendor-neutral database operations port for injected SQL adapters.

This module intentionally does not import SQLAlchemy or any other provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionProvider(Protocol):
    """Host-owned callable returning an async session context manager."""

    def __call__(self) -> Any:
        """Open a caller-owned database session."""
        raise NotImplementedError


class SQLFactory(ABC):
    """Abstract factory for every SQL operation ragnor performs.

    Subclass this to route ragnor's SQL through a different
    stack, and inject the instance via ``sql_factory=...`` on
    :class:`ragnor.ingestion.pipeline.IngestionPipeline` and
    :class:`ragnor.stores.pgvector.PgVectorStore`.
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
    def advisory_lock(self, session: Any, key: str) -> AbstractAsyncContextManager[None]:
        """Hold a database-level lock that serializes ``key``.

        Args:
            session: Database session the lock is taken on.
            key: Lock key (the source path).

        Yields:
            Nothing. The lock is held for the duration of the
            ``async with`` block.
        """
