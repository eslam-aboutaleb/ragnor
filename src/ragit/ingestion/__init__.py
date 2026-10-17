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
from ragit.ingestion import connectors
from ragit.ingestion.connectors import (
    DirectoryConnector,
    LoadResult,
    SourceConnector,
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
    "DirectoryConnector",
    "DocumentSource",
    "FileDocumentSource",
    "IngestionPipeline",
    "IngestionSnapshot",
    "RETRIEVAL_SCHEMA_VERSION",
    "RawDocument",
    "LoadResult",
    "SourceConnector",
    "VersionRecord",
    "VersionStore",
    "advisory_lock",
    "build_snapshot",
    "snapshot_matches",
]


# Re-export the explicitly declared connector API without wildcard imports.
for _name in connectors.__all__:
    globals()[_name] = getattr(connectors, _name)
__all__ = list(dict.fromkeys([*__all__, *connectors.__all__]))
