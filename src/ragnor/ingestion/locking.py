"""PostgreSQL advisory lock helper for ingestion.

Compatibility wrapper: the lock lives on the abstract
SQL factory (:class:`ragnor.db.sql_factory.SQLFactory`),
and :class:`ragnor.ingestion.pipeline.IngestionPipeline`
takes a ``sql_factory`` argument. This module-level helper
delegates to the default :class:`SQLAlchemyFactory` for existing callers.
See :meth:`SQLFactory.advisory_lock` for the complete transactional ordering contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ragnor.db.sql_factory import SQLAlchemyFactory


@asynccontextmanager
async def advisory_lock(session: Any, key: str) -> AsyncIterator[None]:
    """Hold a PostgreSQL advisory lock keyed by ``key``.

    Delegates to :meth:`SQLAlchemyFactory.advisory_lock`.
    """
    async with SQLAlchemyFactory().advisory_lock(session, key):
        yield
