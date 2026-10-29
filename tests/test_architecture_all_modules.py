"""Enforce *all* inner-layer eager-import boundaries, not a handpicked file list.

Provider-specific adapter packages and the SDK composition root may import
external libraries. Reusable domain/application code must not do so eagerly.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "ragnor"
VENDOR_ROOTS = frozenset(
    {
        "sqlalchemy",
        "asyncpg",
        "litellm",
        "qdrant_client",
        "chromadb",
        "faiss",
        "tiktoken",
        "fastapi",
        "django",
        "boto3",
        "httpx",
        "watchfiles",
        "mcp",
        "pypdf",
        "pdfplumber",
        "pypdfium2",
        "PIL",
        "opentelemetry",
    }
)
EXTERNAL_ADAPTERS = {
    "embeddings/litellm.py",
    "evidence/tiktoken_adapter.py",
    "db/sql_factory.py",
    "telemetry/otel.py",
}
EXTERNAL_PREFIXES = (
    "jobs/adapters/",
    "mcp/",
    "stores/",
)
EXTERNAL_NAMES = {
    "ingestion/connectors.py",
    "ingestion/directory_connector.py",
    "ingestion/http_connector.py",
    "ingestion/object_connector.py",
    "connectors.py",
    "builder.py",
    "client/builder.py",
    "registry.py",
}
INNER_DIRS = (
    "chunking/",
    "client/",
    "core/",
    "embeddings/",
    "evaluation/",
    "evidence/",
    "ingestion/",
    "jobs/",
    "retrieval/",
    "sync/",
    "telemetry/",
    "tools/",
)


def _inner_sources() -> list[Path]:
    return [
        file
        for file in ROOT.rglob("*.py")
        if (rel := file.relative_to(ROOT).as_posix()) not in EXTERNAL_ADAPTERS
        and rel not in EXTERNAL_NAMES
        and not any(rel.startswith(prefix) for prefix in EXTERNAL_PREFIXES)
        and (
            rel.startswith(INNER_DIRS)
            or rel
            in {
                "config.py",
                "exceptions.py",
                "lifecycle.py",
                "sync_client.py",
                "types.py",
                "validation.py",
            }
        )
    ]


def _eager_vendor_imports(code: str) -> list[str]:
    tree = ast.parse(code)
    results: list[str] = []
    # Only imports at module scope are eager; local imports inside an optional
    # function or TYPE_CHECKING are permitted for compatibility.
    for node in tree.body:
        if isinstance(node, ast.Import):
            results.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] in VENDOR_ROOTS
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in VENDOR_ROOTS:
            results.append(node.module or "")
    return results


def test_all_inner_modules_have_no_eager_optional_vendor_imports() -> None:
    files = _inner_sources()
    assert len(files) >= 40, "architecture guard must inspect the entire inner SDK"
    failures = {
        file.relative_to(ROOT).as_posix(): _eager_vendor_imports(file.read_text(encoding="utf-8"))
        for file in files
        if _eager_vendor_imports(file.read_text(encoding="utf-8"))
    }
    assert not failures, f"Core import dependency violation: {failures}"


def test_advisory_lock_compatibility_imports_without_optional_sqlalchemy() -> None:
    script = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name.startswith('sqlalchemy'):\n"
        "        raise ImportError('SQLAlchemy intentionally unavailable')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "from ragnor.ingestion.locking import advisory_lock\n"
        "assert callable(advisory_lock)\n"
    )
    proc = subprocess.run(  # noqa: S603 -- trusted interpreter and constant test code
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, proc.stderr


def test_checker_distinguishes_local_lazy_from_eager_imports() -> None:
    assert _eager_vendor_imports("import sqlalchemy\n") == ["sqlalchemy"]
    assert _eager_vendor_imports("def optional():\n    import sqlalchemy\n") == []


def test_pdf_sdk_adapters_import_and_configure_without_optional_dependencies():
    script = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'pypdf', 'pdfplumber', 'pypdfium2', 'PIL'}:
        raise ImportError('optional package blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from ragnor.ingestion import AutoPDFExtractor, PdfPlumberExtractor, TesseractPDFExtractor
for adapter in (AutoPDFExtractor(), PdfPlumberExtractor(), TesseractPDFExtractor()):
    assert adapter.options['policy'] == 'strict'
assert not any(name in sys.modules for name in ('pypdf', 'pdfplumber', 'pypdfium2', 'PIL'))
"""
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and literal test script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
