"""Dependency-free document OCR port shared by local and hosted backends."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

from ragnor.ingestion.extraction_types import PDFExtractionLimits, PageExtractionOutcome


def shared_limits(*limits: PDFExtractionLimits) -> PDFExtractionLimits:
    """An override can tighten, never relax, backend quotas."""
    configs = [asdict(value) for value in limits]
    return PDFExtractionLimits(
        **{
            key: min(values)
            if (values := [c[key] for c in configs if c[key] is not None])
            else None
            for key in configs[0]
        }
    )


@runtime_checkable
class OCRProvider(Protocol):
    """Extract PDF page evidence; unselected pages retain native outcomes.

    Implementations must honor quotas and keep unknown OCR completeness partial.
    Custom implementations are trusted host code, including their network policy.
    """

    id: str
    version: str
    limits: PDFExtractionLimits

    @property
    def options(self) -> dict[str, Any]: ...

    def extract_outcomes(
        self,
        data: bytes,
        *,
        page_numbers: frozenset[int] | None = None,
        limits: PDFExtractionLimits | None = None,
    ) -> list[PageExtractionOutcome]: ...

    async def aextract_outcomes(
        self,
        data: bytes,
        *,
        page_numbers: frozenset[int] | None = None,
        limits: PDFExtractionLimits | None = None,
    ) -> list[PageExtractionOutcome]: ...
