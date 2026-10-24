"""SQLAlchemy-backed transaction lock helper for ingestion.

Compatibility wrapper: the lock lives on the abstract
SQL factory (:class:`ragnor.db.ports.SQLFactory`),
and :class:`ragnor.ingestion.pipeline.IngestionPipeline`
takes a ``sql_factory`` argument. This module-level helper
delegates to the default :class:`SQLAlchemyFactory` for existing callers.
See :meth:`SQLFactory.advisory_lock` for the complete transactional ordering contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def advisory_lock(session: Any, key: str) -> AsyncIterator[None]:
    """Serialize ingestion using a row in the host's database transaction.

    Delegates to :meth:`SQLAlchemyFactory.advisory_lock`.
    """
    # Backwards-compatible entrypoint: don't import SQLAlchemy merely because
    # the host imports the neutral ingestion API. Import only when this helper
    # is actually called.
    try:
        from ragnor.db.sql_factory import SQLAlchemyFactory  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Database locking requires the optional SQLAlchemy adapter; "
            "install ragnor[database] or inject an independent SQLFactory"
        ) from exc

    async with SQLAlchemyFactory().advisory_lock(session, key):
        yield
