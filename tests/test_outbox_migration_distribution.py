"""SDK-06: migration SQL is versioned, wheels include it and DDL cannot drift."""

from __future__ import annotations

import re
from importlib.resources import files

from ragnor.jobs.adapters.postgres import DDL


def _canonical(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).casefold()


def test_migrations_match_the_actual_runtime_postgres_contract() -> None:
    migrations = files("ragnor.jobs.adapters").joinpath("migrations")
    statements = []
    for filename in (
        "001_embedding_outbox.sql",
        "002_active_generation.sql",
        "003_generation_reservations.sql",
    ):
        sql = migrations.joinpath(filename).read_text(encoding="utf-8")
        assert sql.strip()
        statements.extend(part.strip() for part in sql.split(";") if part.strip())
    assert len(statements) == len(DDL)
    assert [_canonical(sql) for sql in statements] == [_canonical(sql) for sql in DDL]


def test_schema_assets_are_accessible_as_distribution_resources() -> None:
    adapter = files("ragnor.jobs.adapters").joinpath("migrations")
    for name in (
        "001_embedding_outbox.sql",
        "002_active_generation.sql",
        "003_generation_reservations.sql",
    ):
        assert adapter.joinpath(name).is_file(), name
