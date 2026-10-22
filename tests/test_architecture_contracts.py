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
    "types.py",
    "config.py",
    "registry.py",
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


def _prohibited_imports(source: str) -> list[str]:
    """Find forbidden imports via syntax tree, not comments or docstrings."""
    nodes = ast.walk(ast.parse(source))
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names if _is_forbidden_import(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if _is_forbidden_import(module):
                names.append(module)
            else:
                names.extend(
                    f"{module}.{alias.name}" for alias in node.names
                    if _is_forbidden_import(f"{module}.{alias.name}")
                )
    return names


@pytest.mark.parametrize("relative_path", INNER_MODULES)
def test_core_ports_and_use_cases_do_not_depend_on_vendor_adapters(
    relative_path: str,
) -> None:
    path = PACKAGE_ROOT / relative_path
    forbidden = _prohibited_imports(path.read_text(encoding="utf-8"))
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
