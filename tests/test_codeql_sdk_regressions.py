"""CodeQL regression: dependency graph and executable protocol declarations."""

from __future__ import annotations

import ast
from pathlib import Path

from ragnor.db.ports import SessionProvider
from ragnor.db.session import SessionProvider as LegacyProvider

ROOT = Path(__file__).resolve().parents[1] / "src/ragnor"


def test_session_port_identity_is_preserved() -> None:
    assert SessionProvider is LegacyProvider


def test_pure_sql_port_imports_no_session_or_vendor() -> None:
    for path in ("db/ports.py", "db/sql_factory.py"):
        module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)}
        assert "ragnor.db.session" not in imported


def test_outbox_ports_have_executable_contract_bodies() -> None:
    tree = ast.parse((ROOT / "jobs/outbox.py").read_text(encoding="utf-8"))
    assert not [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    ]
