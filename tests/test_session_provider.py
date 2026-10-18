"""Session provider protocol and standalone session factory.

Pins that :func:`ragnor.db.session.create_session_factory`
builds a working, protocol-conforming session factory against
an in-memory SQLite database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from ragnor.db.session import SessionProvider, create_session_factory


def test_create_session_factory_returns_session_provider():
    factory = create_session_factory("sqlite+aiosqlite:///:memory:")
    assert callable(factory)
    assert isinstance(factory, SessionProvider)


@pytest.mark.asyncio
async def test_create_session_factory_yields_working_session():
    factory = create_session_factory("sqlite+aiosqlite:///:memory:")
    async with factory() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
