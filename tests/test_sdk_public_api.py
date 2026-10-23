"""SDK-01: public Python imports, compatibility and offline consumer."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest


def test_public_symbols_are_discoverable_and_stable() -> None:
    import ragnor
    from ragnor.builder import Ragnor as OldRagnor
    from ragnor.builder import RagnorBuilder as OldBuilder
    from ragnor.types import Chunk as OldChunk
    from ragnor.types import Document as OldDocument
    from ragnor.types import RetrievalError as OldError
    from ragnor.types import SearchResult as OldResult

    assert ragnor.Ragnor is OldRagnor
    assert ragnor.RagnorBuilder is OldBuilder
    assert ragnor.Document is OldDocument
    assert ragnor.Chunk is OldChunk
    assert ragnor.SearchResult is OldResult
    assert ragnor.RetrievalError is OldError
    assert sorted(ragnor.__all__) == sorted(
        ["Ragnor", "RagnorBuilder", "Document", "Chunk", "SearchResult", "RetrievalError"]
    )
    with pytest.raises(AttributeError):
        _ = ragnor.nonexistent_symbol


def test_public_imports_do_not_import_optional_backends() -> None:
    script = (
        "import builtins\n"
        "original = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {"
        "'sqlalchemy','litellm','qdrant_client','chromadb','faiss','tiktoken','fastapi'}:\n"
        "        raise ImportError('optional provider imported: '+name)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "from ragnor import Ragnor, RagnorBuilder, Document, SearchResult, Chunk\n"
        "assert all([Ragnor, RagnorBuilder, Document, SearchResult, Chunk])\n"
    )
    proc = subprocess.run(  # noqa: S603 -- fixed trusted interpreter and constant code
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, proc.stderr


def test_root_import_is_idempotent() -> None:
    import ragnor

    assert importlib.reload(ragnor).RagnorBuilder is ragnor.RagnorBuilder
