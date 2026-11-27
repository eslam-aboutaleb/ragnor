"""A LlamaIndex ``TextSplitter`` as a ragnor ``Chunker``."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llama_index.core.node_parser import TextSplitter

from ragnor.core.types import Chunk


class LlamaIndexChunker:
    """Adapt a LlamaIndex text splitter (``SentenceSplitter`` etc.).

    Chunk ids are deterministic (hash of position and text) so re-ingesting
    unchanged text is idempotent. ``chunk_metadata`` is stamped on every chunk
    (for example ``{"tenant_id": "alice"}`` so a pipeline writes rows that a
    scoped :class:`RagnorVectorStore` or :class:`RagnorRetriever` can see). ``version`` and
    ``config`` feed ragnor's ingestion snapshot, so changing the splitter or its settings re-chunks.
    """

    def __init__(
        self, splitter: TextSplitter, *, chunk_metadata: dict[str, Any] | None = None
    ) -> None:
        if not hasattr(splitter, "split_text"):
            raise TypeError("splitter must provide split_text(text) -> list[str]")
        self._splitter = splitter
        self._chunk_metadata = dict(chunk_metadata or {})
        # Same text under different chunk_metadata (e.g. two tenants) gets distinct ids.
        self._salt = json.dumps(self._chunk_metadata, sort_keys=True, default=str)

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into ragnor chunks."""
        pieces = [p for p in self._splitter.split_text(text) if p.strip()]
        return [
            Chunk(
                id=hashlib.sha256(f"{self._salt}\x00{i}\x00{piece}".encode()).hexdigest()[:32],
                text=piece,
                metadata=dict(self._chunk_metadata),
            )
            for i, piece in enumerate(pieces)
        ]

    @property
    def version(self) -> str:
        """Splitter class plus llama-index-core settings tag."""
        return f"llamaindex:{type(self._splitter).__name__}"

    @property
    def config(self) -> dict[str, Any]:
        """Scalar splitter settings (chunk size, overlap, separator...)."""
        data = self._splitter.model_dump(mode="json", exclude_none=True)
        config = {
            key: value
            for key, value in sorted(data.items())
            if isinstance(value, (str, int, float, bool))
        }
        if "chunk_overlap" in config:
            config["overlap"] = config["chunk_overlap"]  # key ragnor snapshots read
        if self._chunk_metadata:
            config["chunk_metadata"] = self._salt
        return config


__all__ = ["LlamaIndexChunker"]
