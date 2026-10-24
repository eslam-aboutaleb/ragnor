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
        chunks: list[Chunk] = []
        for index, piece in enumerate(text.split("|")):
            if piece:
                chunks.append(Chunk(id=f"consumer-{index}", text=piece))
        return chunks


async def main() -> None:
    installed_path = Path(inspect.getfile(RagnorBuilder)).resolve()
    assert "site-packages" in str(installed_path), (
        f"Expected installed wheel, got import path: {installed_path}"
    )
    requirements = importlib.metadata.requires("ragnor") or []
    # Optional extras are declared in wheel metadata but are not required
    # when installing only the base distribution.
    assert all("extra ==" in requirement for requirement in requirements), requirements

    builder = RagnorBuilder.local(HostEmbedder(), embedding_dim=2)
    rag = await builder.with_chunker(HostChunker()).build()

    document = Document(id="customer-policy", text="refund conditions|office hours")
    count = await rag.ingest([document])
    assert count == 2
    assert await rag.count() == 2
    results = await rag.search("refund", n_results=2)
    assert results and any("refund" in result.document for result in results)

    print("Installed-wheel SDK smoke passed:", installed_path)


if __name__ == "__main__":
    asyncio.run(main())
