"""Pluggable extraction providers with pinned versions and config.

A provider turns PDF bytes into per-page results. Providers never leak page
contents into errors: failures are recorded as the exception class name plus
a short sanitized message, because provider errors can embed file paths or
stream fragments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmarks.pdf_extraction.manifest import Table


@dataclass
class ProviderPage:
    """One page of provider output."""

    page: int
    text: str
    status: str = "ok"  # "ok" | "empty" | "error"
    error: str | None = None
    duration_ms: float = 0.0
    blocks: list[dict] | None = None
    tables: list[Table] | None = None
    cost_usd: float = 0.0


@runtime_checkable
class ExtractionProvider(Protocol):
    """Provider contract: stable identity, pinned config, per-page results."""

    id: str
    version: str
    config: dict[str, Any]

    def extract(self, data: bytes, *, name: str = "") -> list[ProviderPage]:
        """Extract every real page, or raise; the runner records failures."""
        ...

    def library_versions(self) -> dict[str, str]:
        """Pinned versions of the underlying extraction libraries."""
        ...


def sanitize_error(exc: BaseException) -> str:
    """Record only the exception class; messages can embed paths or content."""
    return type(exc).__name__


@dataclass
class PyPDFProvider:
    """The in-repo pypdf extractor (``ragnor[pdf]``) as a benchmark provider.

    Image-only (scanned) pages return empty text: pypdf performs no OCR, and
    this provider honestly reports that as ``empty`` rather than fabricating
    content.
    """

    max_pages: int = 1000
    id: str = "pypdf"
    version: str = "2"
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config = {"max_pages": self.max_pages}

    def library_versions(self) -> dict[str, str]:
        try:
            import pypdf  # noqa: PLC0415
        except ImportError:
            return {"pypdf": "not-installed"}
        return {"pypdf": pypdf.__version__}

    def extract(self, data: bytes, *, name: str = "") -> list[ProviderPage]:
        import logging  # noqa: PLC0415

        from ragnor.ingestion.extractors import PDFExtractor  # noqa: PLC0415

        # pypdf logs parse warnings (e.g. for the intentionally corrupted
        # fixture); the benchmark records failures itself, so keep output clean.
        logging.getLogger("pypdf").setLevel(logging.CRITICAL)

        extractor = PDFExtractor(options={"max_pages": self.max_pages})
        pages: list[ProviderPage] = []
        for number, text in extractor.extract_pages(data):
            pages.append(
                ProviderPage(
                    page=number,
                    text=text,
                    status="ok" if text.strip() else "empty",
                )
            )
        return pages


def measure(
    provider: ExtractionProvider, data: bytes, name: str
) -> tuple[list[ProviderPage], str | None]:
    """Run a provider, timing it and converting failures into error pages.

    Returns ``(pages, error)``; when the provider raises, ``pages`` is empty
    and ``error`` is the sanitized exception class name.
    """
    start = time.perf_counter()
    try:
        pages = provider.extract(data, name=name)
    except Exception as exc:
        return [], sanitize_error(exc)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if pages:
        per_page = elapsed_ms / len(pages)
        for page in pages:
            if page.duration_ms == 0.0:
                page.duration_ms = per_page
    return pages, None


def registry() -> dict[str, type[PyPDFProvider]]:
    """Providers shipped with the benchmark. External adapters register elsewhere."""
    return {"pypdf": PyPDFProvider}
