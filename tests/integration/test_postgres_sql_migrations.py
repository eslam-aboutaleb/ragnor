"""Run the published SQL migrations twice against an actual PostgreSQL service."""

from __future__ import annotations

import os
from importlib.resources import files

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("RAGNOR_TEST_DATABASE_URL"), reason="real PostgreSQL required"
    ),
]


async def test_real_postgres_migrations_are_idempotent_and_have_required_columns() -> None:
    engine = create_async_engine(os.environ["RAGNOR_TEST_DATABASE_URL"])
    try:
        for _ in range(2):
            async with engine.begin() as connection:
                for filename in (
                    "001_embedding_outbox.sql",
                    "002_active_generation.sql",
                    "003_generation_reservations.sql",
                ):
                    resource = files("ragnor.jobs.adapters").joinpath("migrations", filename)
                    for statement in resource.read_text(encoding="utf-8").split(";"):
                        if statement.strip():
                            await connection.execute(text(statement.strip()))
        async with engine.connect() as connection:
            cols = await connection.execute(
                text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_name IN ('ragnor_embedding_jobs','ragnor_active_generations')"
                )
            )
            observed = {(row.table_name, row.column_name) for row in cols}
            for required in (
                ("ragnor_embedding_jobs", "claim_generation"),
                ("ragnor_embedding_jobs", "lease_expires_at"),
                ("ragnor_embedding_jobs", "idempotency_key"),
                ("ragnor_active_generations", "scope"),
                ("ragnor_active_generations", "revision"),
                ("ragnor_active_generations", "generation_key"),
            ):
                assert required in observed
    finally:
        await engine.dispose()
