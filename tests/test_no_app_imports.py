"""Invariant: ragit source must never import the host application package.

The master plan forbids ragit from depending on the host
application's ``app`` package. This test greps every file under
``libs/ragit/src`` for ``from app`` and ``import app`` and fails
on any match, mirroring the grep check in the plan's validation
section.
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = ("from app", "import app")


def test_ragit_source_does_not_import_app() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_DIR.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for needle in FORBIDDEN:
                if needle in line:
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
    message = "ragit source must not import the app package:\n" + "\n".join(offenders)
    assert not offenders, message
