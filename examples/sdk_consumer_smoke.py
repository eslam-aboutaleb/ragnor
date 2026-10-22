"""Installed-wheel smoke for the zero-dependency, host-pluggable SDK.

Run from outside the repository with an isolated virtual environment:
    python /path/to/repo/examples/sdk_consumer_smoke.py
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from pathlib import Path

from ragnor.builder import RagnorBuilder
from ragnor.types import Chunk, Document


class HostEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "refund" in text.lower() else 0.0, 1.0] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


class HostChunker:
    version = "consumer-v1"
    config = {"separator": "|"}

    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(id=f"consumer-{index}", text=piece)
            for index, piece in enumerate(text.split("|"))
            if piece
        ]


async def main() -> None:
    installed_path = Path(inspect.getfile(RagnorBuilder)).resolve()
    assert "site-packages" in str(installed_path), (
        f"Expected installed wheel, got import path: {installed_path}"
    )
    assert not (importlib.metadata.requires("ragnor") or []), (
        "Core SDK unexpectedly requires third-party runtime packages"
    )

    rag = await (
        RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
        .with_chunker(HostChunker())
        .build()
    )

    count = await rag.ingest(
        [Document(id="customer-policy", text="refund conditions|office hours")]
    )
    assert count == 2
    assert await rag.count() == 2
    results = await rag.search("refund", n_results=2)
    assert results and any("refund" in result.document for result in results)

    print("Installed-wheel SDK smoke passed:", installed_path)


if __name__ == "__main__":
    asyncio.run(main())
