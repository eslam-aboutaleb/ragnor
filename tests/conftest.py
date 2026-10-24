"""Explicit configuration for optional tests against a disposable database."""

from __future__ import annotations

import os


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--ragnor-test-database-url",
        action="store",
        default=None,
        help="URL of a disposable PostgreSQL/pgvector test database; never auto-detected",
    )


def pytest_configure(config) -> None:
    database_url = config.getoption("--ragnor-test-database-url")
    if database_url is not None:
        os.environ["RAGNOR_TEST_DATABASE_URL"] = database_url
