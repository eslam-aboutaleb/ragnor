"""Ingestion snapshot comparison.

The snapshot captures everything that determines
whether a stored index is still current: the source
hash, the embedding model and dimension, and the
chunker version and tunables. Comparing the current
snapshot against the stored one makes ingestion
idempotent -- an unchanged source is skipped instead
of re-embedded and re-upserted.

The source hash is a *manifest* hash: every
document's source name, external id, and content
hash are recorded in a canonical JSON manifest, so
document boundaries, source names, external ids, and
content all feed the hash. Concatenating document
content (``"ab" + "c"`` vs ``"a" + "bc"``) is not
enough to change it -- the manifest is.

Version *storage* is application-side; this module
defines the value type and the comparison only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ragnor.chunking.base import Chunker
from ragnor.core.config import RagSettings
from ragnor.embeddings import get_embedding_dimension
from ragnor.embeddings._safe_config import safe_config
from ragnor.ingestion.records import RawDocument

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
        extractor_id: Identifier of the extractor used.
        extractor_version: Version of the extractor.
        extractor_options: Non-secret options passed to the extractor.
        embedding_profile_hash: Canonical hash of the
            :class:`~ragnor.embeddings.profile.EmbeddingProfile` used for
            this ingestion.  Empty string when no profile is available
            (backward-compatible default).
    """

    source_hash: str
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    chunk_size: int
    overlap: int
    retrieval_schema_version: str
    extractor_id: str = "none"
    extractor_version: str = "0"
    extractor_options: dict[str, Any] = field(default_factory=dict)
    embedding_profile_hash: str = ""
    chunker_config_hash: str = ""


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


def compute_source_hash(documents: list[RawDocument]) -> str:
    """Compute the manifest hash of a loaded document set.

    The hash covers the whole manifest -- every
    document's source name, external id, and content
    hash -- serialized as canonical JSON (sorted
    keys, no whitespace) before hashing. Document
    boundaries, source names, external ids, and
    content all feed the result, so a changed
    document identity or metadata produces a
    different hash even when the concatenated
    content is unchanged.

    Args:
        documents: Loaded documents in source order.

    Returns:
        SHA-256 hex digest of the canonical manifest.
    """
    entries = []
    for doc in documents:
        entry = {
            "source_name": doc.source_name,
            "external_id": doc.external_id,
            "content_hash": hashlib.sha256(doc.content.encode("utf-8")).hexdigest(),
        }
        if doc.extractor is not None:
            entry["extractor"] = doc.extractor
        if "extraction" in doc.metadata:
            entry["extraction"] = safe_config(doc.metadata["extraction"])
        if doc.metadata:
            entry["metadata"] = safe_config(doc.metadata)
        if doc.location is not None:
            entry["location"] = safe_config(doc.location)
        if doc.provenance:
            entry["provenance"] = safe_config(doc.provenance)
        entries.append(entry)
    manifest = {"documents": entries}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot(  # noqa: PLR0913, PLR0917
    source_hash: str,
    settings: RagSettings,
    chunker: Chunker,
    extractor: Any = None,
    extractor_id: str | None = None,
    extractor_version: str | None = None,
    extractor_options: dict[str, Any] | None = None,
    embedding_profile: Any | None = None,
) -> IngestionSnapshot:
    """Build the current snapshot for ``source_hash``.

    The embedding model and dimension come from the
    settings (the dimension via the ragnor dimension
    registry); the chunker version and tunables come
    from the chunker, so a chunker upgrade changes the
    snapshot and forces a re-ingest.

    Args:
        source_hash: SHA-256 hex digest of the source content.
        settings: Settings carrying the embedding model.
        chunker: Chunker that splits the document.
        extractor: Optional extractor instance or string id.
        extractor_id: Optional extractor identifier (overridden by
            ``extractor`` when provided).
        extractor_version: Optional extractor version.
        extractor_options: Optional non-secret extractor options.
        embedding_profile: Optional
            :class:`~ragnor.embeddings.profile.EmbeddingProfile`.
            When provided its :meth:`canonical_hash` is stored in the
            snapshot so profile changes force a re-ingest.

    Returns:
        The snapshot describing this ingestion configuration.
    """
    config: dict[str, Any] = chunker.config
    if extractor is not None:
        extractor_id = (
            extractor
            if isinstance(extractor, str)
            else getattr(extractor, "id", type(extractor).__name__)
        )
        extractor_version = getattr(extractor, "version", extractor_version)
        extractor_options = getattr(extractor, "options", extractor_options)
    profile_hash = ""
    if embedding_profile is not None and hasattr(embedding_profile, "canonical_hash"):
        profile_hash = embedding_profile.canonical_hash()
    return IngestionSnapshot(
        source_hash=source_hash,
        embedding_model=(
            embedding_profile.model_id
            if embedding_profile is not None
            else settings.embedding_model
        ),
        embedding_dim=(
            embedding_profile.dimension
            if embedding_profile is not None
            else get_embedding_dimension(settings.embedding_model)
        ),
        chunker_version=chunker.version,
        chunk_size=int(config.get("chunk_size", 0)),
        overlap=int(config.get("overlap", 0)),
        retrieval_schema_version=RETRIEVAL_SCHEMA_VERSION,
        extractor_id=extractor_id or "none",
        extractor_version=extractor_version or "0",
        extractor_options=safe_config(extractor_options or {}),
        embedding_profile_hash=profile_hash,
        chunker_config_hash=hashlib.sha256(
            json.dumps(
                {"version": chunker.version, "config": safe_config(config)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )
