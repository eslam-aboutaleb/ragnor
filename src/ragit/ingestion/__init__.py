"""Ingestion seam: sources, snapshot versioning, locking, pipeline."""

from __future__ import annotations

from ragit.ingestion.pipeline import (
    IngestionPipeline,
    VersionRecord,
    VersionStore,
)
from ragit.ingestion.sources import (
    DocumentSource,
    FileDocumentSource,
    RawDocument,
)
from ragit.ingestion.versioning import (
    RETRIEVAL_SCHEMA_VERSION,
    IngestionSnapshot,
    build_snapshot,
    snapshot_matches,
)


def __getattr__(name: str):
    """Lazily import the advisory lock so its driver stays optional."""
    if name == "advisory_lock":
        from ragit.ingestion.locking import advisory_lock  # noqa: PLC0415

        return advisory_lock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DocumentSource",
    "FileDocumentSource",
    "IngestionPipeline",
    "IngestionSnapshot",
    "RETRIEVAL_SCHEMA_VERSION",
    "RawDocument",
    "VersionRecord",
    "VersionStore",
    "advisory_lock",
    "build_snapshot",
    "snapshot_matches",
]
