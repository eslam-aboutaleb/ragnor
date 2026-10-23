"""Regression coverage for CodeQL findings carried by stacked SDK PRs.

These checks verify vendor-neutral import boundaries, compatibility, and
that declared extension protocols have explicit executable placeholder
bodies rather than bare expressions flagged as no-effect statements.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "ragnor"


def _imports(path: str) -> set[str]:
    source = (ROOT / path).read_text(encoding="utf-8")
    return {
        name
        for node in ast.walk(ast.parse(source))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }


def test_sql_ports_do_not_depend_on_higher_level_session_or_vendor_modules() -> None:
    assert "ragnor.db.session" not in _imports("db/ports.py")
    assert "ragnor.db.sql_factory" not in _imports("db/ports.py")
    assert "ragnor.db.session" not in _imports("db/sql_factory.py")


def test_session_provider_identity_remains_compatible() -> None:
    from ragnor.db.ports import SessionProvider
    from ragnor.db.session import SessionProvider as LegacySessionProvider

    assert LegacySessionProvider is SessionProvider


def test_security_reviewed_protocols_have_no_no_effect_ellipsis_statements() -> None:
    paths = (
        "jobs/outbox.py",
        "jobs/publication.py",
        "jobs/adapters/generation_vectors.py",
        "evidence/prompt_budget.py",
    )
    for path in paths:
        source = (ROOT / path).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                assert node.value.value is not Ellipsis, f"{path}:{node.lineno}"


def test_db_factory_optional_dependencies_remain_lazy() -> None:
    import subprocess
    import sys

    snippet = (
        "import builtins\n"
        "base = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.startswith('sqlalchemy'):\n"
        "        raise ImportError('SQLAlchemy unavailable')\n"
        "    return base(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from ragnor.db.ports import SQLFactory, SessionProvider\n"
        "from ragnor.db.session import SessionProvider as LegacySessionProvider\n"
        "assert LegacySessionProvider is SessionProvider\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and constant script
        [sys.executable, "-c", snippet],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
