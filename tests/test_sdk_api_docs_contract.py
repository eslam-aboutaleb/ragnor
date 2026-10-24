"""SDK-01: public signatures and real documented quickstart regression."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from ragnor import Ragnor, RagnorBuilder

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "docs" / "sdk" / "api-stability.md"


def test_async_facade_method_signatures_remain_backward_compatible() -> None:
    signatures = {
        "ingest": ("self", "documents", "purge_on_empty", "lock_key"),
        "search": ("self", "query", "n_results", "filters"),
        "count": ("self", "filters"),
        "aclose": ("self",),
        "build": ("self",),
    }
    for name, expected in signatures.items():
        owner = RagnorBuilder if name == "build" else Ragnor
        method = getattr(owner, name)
        assert inspect.iscoroutinefunction(method), name
        assert tuple(inspect.signature(method).parameters) == expected


def test_documented_python_blocks_are_syntactically_valid() -> None:
    markdown = QUICKSTART.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    assert blocks
    for block in blocks:
        compile(block, str(QUICKSTART), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.asyncio
async def test_documented_minimal_quickstart_actually_executes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    markdown = QUICKSTART.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    candidate = next(block for block in blocks if "class DemoEmbedder:" in block)
    # Code block calls asyncio.run(): execute outside a running pytest event
    # loop with the same user-defined demo methods and public APIs.
    source = candidate.replace("asyncio.run(main())", "")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, str(QUICKSTART), "exec"), namespace)  # noqa: S102
    await namespace["main"]()  # type: ignore[operator]
    output = capsys.readouterr().out
    assert "Refund policy" in output
    assert "1" in output
