"""Session provider protocol and standalone session factory.

ragnor talks to databases through the :class:`SessionProvider`
protocol: any callable that returns an async session context
manager works structurally. The host application keeps its own
engine and session factory (including its NullPool test
behavior); :func:`create_session_factory` is the standalone
helper for projects that use ragnor without the host
application. It delegates to the default
:class:`ragnor.db.sql_factory.SQLAlchemyFactory`.
"""

from __future__ import annotations

from ragnor.db.ports import SessionProvider


def create_session_factory(database_url: str) -> SessionProvider:
    """Build a session factory bound to a new async engine.

    Args:
        database_url: SQLAlchemy async database URL (for example
            ``postgresql+asyncpg://user:pass@host/db``).

    Returns:
        A session factory. Each call yields an independent
        ``AsyncSession`` with ``expire_on_commit=False``. The caller
        owns the engine's lifecycle; ragnor creates no module-level
        singletons.
    """
    from ragnor.db.sql_factory import SQLAlchemyFactory  # noqa: PLC0415

    return SQLAlchemyFactory().create_session_factory(database_url)
