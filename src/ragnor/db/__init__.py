"""Vendor-neutral database ports and lazy optional SQLAlchemy integration.

Importing a database port must not import the optional SQLAlchemy driver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragnor.db.ports import SQLFactory
from ragnor.db.session import SessionProvider, create_session_factory

if TYPE_CHECKING:
    from ragnor.db.sql_factory import SQLAlchemyFactory

__all__ = ["SQLAlchemyFactory", "SQLFactory", "SessionProvider", "create_session_factory"]


def __getattr__(name: str) -> Any:
    if name == "SQLAlchemyFactory":
        from ragnor.db.sql_factory import SQLAlchemyFactory  # noqa: PLC0415

        return SQLAlchemyFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
