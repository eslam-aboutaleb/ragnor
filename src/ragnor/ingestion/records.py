"""Dependency-free document records shared by sources and connectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    """A loaded document, before chunking.

    Attributes:
        source_name: Identifier recorded in chunk metadata
            (typically the filename).
        content: The full document text.
        external_id: Optional identifier from the source
            system (e.g., an object key or row id).
        metadata: Source-specific metadata preserved through
            normalization.
        provenance: Original source provenance preserved
            through normalization.
        extractor: Identifier of the extractor that produced
            the content.
        location: Optional location reference (page, byte
            range, etc.).
    """

    source_name: str
    content: str
    external_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    extractor: str | None = None
    location: Any = None
