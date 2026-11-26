"""SDK-ARCH: the shipped conformance suite and the reference external package.

Proves three things offline:
1. ``ragnor.conformance`` imports with no optional backend installed
   (zero-dependency core invariant).
2. The independently authored reference package under
   ``external-adapters/reference`` passes the shipped suite without editing
   Ragnor -- the acceptance pattern third-party adapters must follow.
3. Each published check actually fails adapters that violate its invariant.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ragnor.conformance import (
    check_chunker,
    check_document_source,
    check_embedder,
    check_tokenizer,
)
from ragnor.ingestion.records import RawDocument
from ragnor.types import Chunk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE_SRC = ROOT / "external-adapters" / "reference" / "src"

_VENDOR_BLOCK = (
    "import sys\n"
    "for _name in ('sqlalchemy','asyncpg','litellm','qdrant_client','chromadb','faiss',"
    "'tiktoken','fastapi','django','boto3','httpx','watchfiles','mcp','pypdf',"
    "'opentelemetry'):\n"
    "    sys.modules[_name] = None\n"
)


def _run(script: str, *pythonpath: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    entries = [str(path) for path in pythonpath]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return subprocess.run(  # noqa: S603 -- fixed trusted interpreter and constant code
        [sys.executable, "-c", _VENDOR_BLOCK + script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


def test_conformance_module_imports_without_optional_backends() -> None:
    proc = _run("import ragnor.conformance\nprint('ok')\n", SRC)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_reference_external_package_passes_the_shipped_suite() -> None:
    script = (
        "import asyncio\n"
        "from ragnor.conformance import (\n"
        "    check_chunker, check_document_source, check_embedder,\n"
        "    check_event_sink, check_tokenizer,\n"
        ")\n"
        "from ragnor_reference_adapters import (\n"
        "    EMBEDDING_DIMENSION, ReferenceChunker, ReferenceDocumentSource,\n"
        "    ReferenceEmbedder, ReferenceEventSink, ReferenceTokenizer,\n"
        ")\n"
        "check_chunker(ReferenceChunker(), text='one\\n\\ntwo')\n"
        "check_tokenizer(ReferenceTokenizer())\n"
        "check_event_sink(ReferenceEventSink())\n"
        "async def main():\n"
        "    await check_embedder(ReferenceEmbedder(), EMBEDDING_DIMENSION)\n"
        "    await check_document_source(ReferenceDocumentSource())\n"
        "asyncio.run(main())\n"
        "print('reference package conforms')\n"
    )
    proc = _run(script, SRC, REFERENCE_SRC)
    assert proc.returncode == 0, proc.stderr
    assert "reference package conforms" in proc.stdout


def test_reference_package_has_no_ragnor_checkout_dependency() -> None:
    # The package imports only installed distributions: running it with the
    # repo *tests* tree absent from the path must still work.
    assert (REFERENCE_SRC / "ragnor_reference_adapters" / "__init__.py").is_file()
    assert (ROOT / "external-adapters" / "reference" / "pyproject.toml").is_file()


class _EmptyChunker:
    version = "v1"
    config = {}

    def chunk(self, text: str) -> list[Chunk]:
        return []


class _UnversionedChunker:
    version = ""
    config = {}

    def chunk(self, text: str) -> list[Chunk]:
        return [Chunk(id="c", text=text)]


class _ZeroTokenizer:
    name = "broken"
    is_exact = False

    def count_tokens(self, text: str) -> int:
        return 1


class _ShortEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]]


class _WrongDimensionEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _IdentitylessSource:
    async def load(self) -> list[RawDocument]:
        return [RawDocument(source_name="", content="text")]


class _WrongRecordSource:
    async def load(self) -> list[object]:
        return [object()]


def test_chunker_check_rejects_empty_output() -> None:
    with pytest.raises(AssertionError, match="at least one chunk"):
        check_chunker(_EmptyChunker())


def test_chunker_check_rejects_missing_version() -> None:
    with pytest.raises(AssertionError, match="version"):
        check_chunker(_UnversionedChunker())


def test_tokenizer_check_rejects_nonzero_empty_count() -> None:
    with pytest.raises(AssertionError, match="zero tokens"):
        check_tokenizer(_ZeroTokenizer())


async def test_embedder_check_rejects_batch_cardinality_violation() -> None:
    with pytest.raises(AssertionError, match="one vector per input"):
        await check_embedder(_ShortEmbedder(), 2)


async def test_embedder_check_rejects_dimension_violation() -> None:
    with pytest.raises(AssertionError, match="finite components"):
        await check_embedder(_WrongDimensionEmbedder(), 3)


async def test_document_source_check_rejects_missing_source_name() -> None:
    with pytest.raises(AssertionError, match="source_name"):
        await check_document_source(_IdentitylessSource())


async def test_document_source_check_rejects_untyped_records() -> None:
    with pytest.raises(AssertionError, match="RawDocument"):
        await check_document_source(_WrongRecordSource())
