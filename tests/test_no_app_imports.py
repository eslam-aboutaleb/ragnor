"""Invariant: ragnor source must never import the host application package.

The master plan forbids ragnor from depending on the host
application's ``app`` package. This test greps every file under
``libs/ragnor/src`` for ``from app`` and ``import app`` and fails
on any match, mirroring the grep check in the plan's validation
section.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
# Whole-word match: "import apply_routes" must not count as "import app".
FORBIDDEN = re.compile(r"^\s*(?:from|import)\s+app(?:[.\s,]|$)")


def test_ragnor_source_does_not_import_app() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_DIR.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    message = "ragnor source must not import the app package:\n" + "\n".join(offenders)
    assert not offenders, message


def test_guard_matches_real_app_imports_but_not_lookalikes() -> None:
    for bad in ("from app import x", "import app", "from app.core import y", "import app, os"):
        assert FORBIDDEN.search(bad), bad
    for ok in (
        "from ragnor.extraction.routes import apply_routes",
        "import apply_routes",
        "from application import z",
    ):
        assert not FORBIDDEN.search(ok), ok
