"""SDK-01: public Python imports, compatibility and offline consumer."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

import ragnor


def test_public_symbols_are_discoverable_and_stable() -> None:
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
    assert importlib.reload(ragnor).RagnorBuilder is ragnor.RagnorBuilder


@pytest.mark.asyncio
async def test_public_facade_ordering_metadata_empty_and_capability_errors() -> None:
    Document = ragnor.Document
    RagnorBuilder = ragnor.RagnorBuilder
    RetrievalError = ragnor.RetrievalError
    from ragnor.exceptions import UnsupportedCapabilityError
    from ragnor.retrieval.capabilities import FilterOperator, StoreCapabilities
    from ragnor.stores.memory import InMemoryVectorStore

    class _Embedder:
        async def __call__(self, texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                low = t.lower()
                if "refund" in low:
                    out.append([1.0, 0.0])
                elif "shipping" in low:
                    out.append([0.0, 1.0])
                else:
                    out.append([-1.0, -1.0])
            return out

    class _RestrictedStore(InMemoryVectorStore):
        @property
        def capabilities(self) -> StoreCapabilities:
            return StoreCapabilities(filter_operators=frozenset({FilterOperator.EQ}))

    embedder = _Embedder()
    store = _RestrictedStore(embedding_dim=2, embedding_fn=embedder)

    rag = await RagnorBuilder.local(embedder, embedding_dim=2, store=store).build()
    count = await rag.ingest(
        [
            Document(
                id="doc-refund",
                text="Refund policy allows returns within 30 days.",
                metadata={"category": "billing"},
            ),
            Document(
                id="doc-shipping",
                text="Shipping takes 3 to 5 business days.",
                metadata={"category": "logistics"},
            ),
        ]
    )
    assert count == 2

    hits = await rag.search("refund policy", n_results=5, category="billing")
    assert len(hits) == 1
    assert hits[0].metadata["source"] == "doc-refund"
    assert hits[0].metadata["version_id"] == "v1"
    assert [h.rrf_score for h in hits] == sorted([h.rrf_score for h in hits], reverse=True)

    # Unmatched query with strict distance threshold returns []
    empty_hits = await rag.search("completely unrelated topic", distance_threshold=0.01)
    assert empty_hits == []

    # Unsupported IN filter on an EQ-only store fails closed with typed error
    with pytest.raises((UnsupportedCapabilityError, RetrievalError)):
        await rag.search("refund", source=["doc-refund", "doc-shipping"])
