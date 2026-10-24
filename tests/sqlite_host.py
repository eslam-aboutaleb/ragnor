"""Configure a host-owned SQLite engine with transactional savepoints."""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_sqlite_engine(url: str, *, timeout: float = 5) -> AsyncEngine:
    engine = create_async_engine(url, connect_args={"timeout": timeout})

    @event.listens_for(engine.sync_engine, "connect")
    def disable_driver_transaction_control(dbapi_connection, _connection_record) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine.sync_engine, "begin")
    def begin_transaction(connection) -> None:
        connection.exec_driver_sql("BEGIN")

    return engine
