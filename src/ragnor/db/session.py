"""Host-owned session provider protocol and deprecated URL helper."""

from __future__ import annotations

from ragnor.db.ports import SessionProvider


def create_session_factory(database_url: str) -> SessionProvider:
    """Reject legacy URL binding; the host must inject its session factory.

    The import path remains for a migration window, but never opens an
    engine. Construct the engine and sessions in the host, then pass the
    sessions to the builder and relational store factory.
    """
    raise RuntimeError(
        "create_session_factory(database_url) is removed: the host must create "
        "and own its engine and async session factory, then inject the sessions"
    )
