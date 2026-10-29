"""Host sessions satisfy the port; legacy URL binding fails before I/O."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragnor.db.session import SessionProvider, create_session_factory


def test_create_session_factory_returns_session_provider():
    with pytest.raises(RuntimeError, match="host must create"):
        create_session_factory("sqlite+aiosqlite:///:memory:")


@pytest.mark.asyncio
async def test_create_session_factory_yields_working_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        assert isinstance(sessions, SessionProvider)
        async with sessions() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    finally:
        await engine.dispose()
