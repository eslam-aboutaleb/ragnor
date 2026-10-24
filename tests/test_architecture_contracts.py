"""Architecture regression contracts for the embeddable, zero-dependency SDK.

A lightweight AST check is deliberate: it tests the *import boundary* without
importing optional provider modules and without depending on extra lint tools.
Other tests cover adapter behavior; this file guards dependency direction.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "ragnor"

# Composition roots (builder, registries and lazy __init__ modules) are
# intentionally excluded; application logic and ports must remain neutral.
INNER_MODULES = (
    "chunking/base.py",
    "embeddings/base.py",
    "evidence/tokenizer.py",
    "ingestion/pipeline.py",
    "jobs/outbox.py",
    "sync/state.py",
    "core/types.py",
    "core/config.py",
    "core/exceptions.py",
    "client/sync_client.py",
    "core/registry.py",
    "db/ports.py",
    "retrieval/capabilities.py",
    "retrieval/types.py",
    "ingestion/versioning.py",
)

FORBIDDEN_NAMES = (
    "app",
    "fastapi",
    "django",
    "litellm",
    "sqlalchemy",
    "qdrant_client",
    "chromadb",
    "faiss",
    "tiktoken",
    "ragnor.stores.memory",
    "ragnor.stores.pgvector",
    "ragnor.stores.qdrant",
    "ragnor.stores.chroma",
    "ragnor.stores.faiss",
    "ragnor.embeddings.litellm",
)


def _is_forbidden_import(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_NAMES)


def _prohibited_imports(
    source: str, *, module_path: str = "ragnor.ingestion.pipeline"
) -> list[str]:
    """Find forbidden absolute and relative imports in a package module."""
    names: list[str] = []
    package_parts = module_path.split(".")[:-1]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names if _is_forbidden_import(alias.name))
        elif isinstance(node, ast.ImportFrom):
            prefix = package_parts[: len(package_parts) - node.level + 1] if node.level else []
            resolved = ".".join([*prefix, node.module] if node.module else prefix)
            if not node.level:
                resolved = node.module or ""
            if _is_forbidden_import(resolved):
                names.append(resolved)
            else:
                names.extend(
                    f"{resolved}.{alias.name}"
                    for alias in node.names
                    if _is_forbidden_import(f"{resolved}.{alias.name}")
                )
    return names


@pytest.mark.parametrize("relative_path", INNER_MODULES)
def test_core_ports_and_use_cases_do_not_depend_on_vendor_adapters(
    relative_path: str,
) -> None:
    path = PACKAGE_ROOT / relative_path
    forbidden = _prohibited_imports(
        path.read_text(encoding="utf-8"),
        module_path="ragnor." + relative_path.removesuffix(".py").replace("/", "."),
    )
    assert not forbidden, f"{relative_path} has forbidden inward dependencies: {forbidden}"


def test_import_checker_fails_for_real_forbidden_imports() -> None:
    example = "import litellm\nfrom ragnor.stores.pgvector import PgVectorStore\n"
    assert _prohibited_imports(example) == ["litellm", "ragnor.stores.pgvector"]


def test_import_checker_ignores_comments_strings_and_safe_imports() -> None:
    example = (
        '"""import fastapi"""\n'
        "# from ragnor.stores.memory import InMemoryVectorStore\n"
        "from ragnor.stores.base import VectorStore\n"
        "from ragnor.embeddings.base import EmbeddingFunction\n"
    )
    assert _prohibited_imports(example) == []


def test_custom_chunker_and_tokenizer_satisfy_existing_ports() -> None:
    from ragnor.chunking.base import Chunker
    from ragnor.evidence.tokenizer import Tokenizer
    from ragnor.types import Chunk

    class ExternalChunker:
        version = "external-v1"
        config = {"chunk_size": 4, "overlap": 0}

        def chunk(self, text: str) -> list[Chunk]:
            return [Chunk(id="external", text=text)]

    class ExternalTokenizer:
        name = "external-test-tokenizer"
        is_exact = True

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    assert isinstance(ExternalChunker(), Chunker)
    assert isinstance(ExternalTokenizer(), Tokenizer)
    assert ExternalChunker().chunk("custom source")[0].text == "custom source"
    assert ExternalTokenizer().count_tokens("custom source") == 2


def test_sql_port_is_importable_without_sqlalchemy() -> None:
    """Protocol must remain usable when optional SQLAlchemy is not installed."""
    import os
    import subprocess
    import sys

    script = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name == 'sqlalchemy' or name.startswith('sqlalchemy.'):\n"
        "        raise ImportError('simulated absent SQLAlchemy')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from ragnor.db.ports import SQLFactory\n"
        "assert SQLFactory.__name__ == 'SQLFactory'\n"
    )
    env = dict(os.environ)
    completed = subprocess.run(  # noqa: S603 -- constant script; no shell or user input
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_relative_imports_of_concrete_vendor_adapters_are_detected() -> None:
    source = "from ..stores.pgvector import PgVectorStore\n"
    assert _prohibited_imports(source, module_path="ragnor.ingestion.pipeline") == [
        "ragnor.stores.pgvector"
    ]
