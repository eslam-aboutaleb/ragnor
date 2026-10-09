"""Ingestion snapshot comparison.

The snapshot captures everything that determines
whether a stored index is still current: the source
hash, the embedding model and dimension, and the
chunker version and tunables. Comparing the current
snapshot against the stored one makes ingestion
idempotent -- an unchanged source is skipped instead
of re-embedded and re-upserted.

Version *storage* is application-side; this module
defines the value type and the comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragit.chunking.base import Chunker
from ragit.config import RagSettings
from ragit.embeddings import get_embedding_dimension

RETRIEVAL_SCHEMA_VERSION = "v1"
"""Version of the retrieval schema the snapshots carry."""


@dataclass
class IngestionSnapshot:
    """Everything that determines whether a stored index is current.

    Attributes:
        source_hash: SHA-256 hex digest of the source content.
        embedding_model: Model the embeddings were produced with.
        embedding_dim: Dimensionality of the embeddings.
        chunker_version: Version tag of the chunking strategy.
        chunk_size: Words per chunk.
        overlap: Overlapping words between consecutive chunks.
        retrieval_schema_version: Version of the retrieval schema.
    """

    source_hash: str
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    chunk_size: int
    overlap: int
    retrieval_schema_version: str


def snapshot_matches(
    current: IngestionSnapshot,
    stored: IngestionSnapshot | None,
) -> bool:
    """Return whether ``current`` equals the ``stored`` snapshot.

    Args:
        current: Snapshot computed from the source and the
            current configuration.
        stored: Snapshot recorded with the active version,
            or ``None`` when no version exists yet.

    Returns:
        ``True`` when the stored index is still current
        (``False`` when there is nothing stored).
    """
    if stored is None:
        return False
    return current == stored


def build_snapshot(
    source_hash: str,
    settings: RagSettings,
    chunker: Chunker,
) -> IngestionSnapshot:
    """Build the current snapshot for ``source_hash``.

    The embedding model and dimension come from the
    settings (the dimension via the ragit dimension
    registry); the chunker version and tunables come
    from the chunker, so a chunker upgrade changes the
    snapshot and forces a re-ingest.

    Args:
        source_hash: SHA-256 hex digest of the source content.
        settings: Settings carrying the embedding model.
        chunker: Chunker that splits the document.

    Returns:
        The snapshot describing this ingestion configuration.
    """
    config: dict[str, Any] = chunker.config
    return IngestionSnapshot(
        source_hash=source_hash,
        embedding_model=settings.embedding_model,
        embedding_dim=get_embedding_dimension(settings.embedding_model),
        chunker_version=chunker.version,
        chunk_size=int(config.get("chunk_size", 0)),
        overlap=int(config.get("overlap", 0)),
        retrieval_schema_version=RETRIEVAL_SCHEMA_VERSION,
    )
