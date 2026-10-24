"""Abstract SQL factory -- the seam between ragnor and any SQL database.

Every SQL operation ragnor performs -- session creation, queries,
mutations, and advisory locking -- goes through the
:class:`SQLFactory` interface. The built-in
:class:`SQLAlchemyFactory` implements it with SQLAlchemy, which
works with any SQLAlchemy dialect (PostgreSQL, SQLite, MySQL, ...).
Hosts targeting another stack subclass :class:`SQLFactory` and
inject their factory into the ingestion pipeline and stores.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import insert, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ragnor.db.ports import SQLFactory, SessionProvider
from ragnor.db.schema import ingestion_locks


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
        """Serialize a source using a row lock in the caller's transaction.

        The host applies the ``ragnor_ingestion_locks`` schema. Updating its
        row acquires the database's write lock until the transaction ends.
        This works through the same SQLAlchemy statement on supported SQL
        databases and never creates a connection or schema implicitly.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("lock key must be non-empty")
        try:
            statement = (
                update(ingestion_locks)
                .where(ingestion_locks.c.lock_key == key)
                .values(generation=ingestion_locks.c.generation + 1)
            )
            changed = await session.execute(statement)
            if changed.rowcount == 0:
                try:
                    async with session.begin_nested():
                        await session.execute(insert(ingestion_locks).values(lock_key=key))
                except IntegrityError:
                    changed = await session.execute(statement)
                    if changed.rowcount != 1:
                        raise RuntimeError("source lock could not be acquired") from None
            elif changed.rowcount != 1:
                raise RuntimeError("database must report source-lock update counts")
            yield
        except Exception:
            await session.rollback()
            raise
