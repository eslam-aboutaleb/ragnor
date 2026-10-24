"""Optional SQLAlchemy schema definitions; the host applies migrations."""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table

metadata = MetaData()

ingestion_locks = Table(
    "ragnor_ingestion_locks",
    metadata,
    Column("lock_key", String, primary_key=True),
    Column("generation", Integer, nullable=False, server_default="0"),
)
