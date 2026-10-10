"""Offline quickstart for ragit's high-level convenience API.

Three lines of application code -- build, ingest, search -- with
every component overridable. The embedder here is a deterministic
fake defined in this file (tokens hash into vector dimensions, so
texts that share vocabulary get close L2 distance); ragit ships no
local embedder in core, so the builder requires an injected one.

Run it with::

    python examples/local_quickstart.py

Everything runs in process: no network, no database, no API keys,
no external services -- only the core install is required
(``pip install git+https://github.com/eslam-aboutaleb/ragit.git``).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re

from ragit.builder import RagitBuilder
from ragit.types import Document


class HashingEmbedder:
    """Deterministic fake embedder: tokens hash into vector dimensions.

    Shared tokens land in shared dimensions, so texts that share
    vocabulary get close L2 distance -- bag-of-words similarity
    without any network calls.
    """

    def __init__(self, dim: int = 8) -> None:
        """Initialize the embedder with a fixed dimension.

        Args:
            dim: Embedding dimensionality.
        """
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Embed ``text`` into a deterministic vector.

        Args:
            text: Text to embed.

        Returns:
            The normalized bag-of-words embedding.
        """
        vector = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed.

        Returns:
            One embedding vector per text.
        """
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: Query text.

        Returns:
            The query embedding vector.
        """
        return self.embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: Document texts to embed.

        Returns:
            One embedding vector per document.
        """
        return [self.embed(text) for text in texts]


async def main() -> None:
    """Build a Ragit instance, ingest two documents, and search."""
    embedder = HashingEmbedder(dim=8)

    # The three-line workflow: build, ingest, search.
    rag = await RagitBuilder.local(embedder, embedding_dim=8).build()
    count = await rag.ingest(
        [
            Document(
                id="refund-policy",
                text="The refund policy covers water damage within thirty days of purchase.",
            ),
            Document(
                id="office-hours",
                text="Office hours are Monday to Friday, nine to five.",
            ),
        ]
    )
    print(f"ingested {count} chunks")

    results = await rag.search("refund policy", n_results=3)
    print("top results for 'refund policy':")
    for result in results:
        print(f"  [{result.id}] {result.document[:70]}")

    print("quickstart complete: offline build, ingest, and search all work")


if __name__ == "__main__":
    asyncio.run(main())
