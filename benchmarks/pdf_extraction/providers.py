"""Pluggable extraction providers with pinned versions and config.

A provider turns PDF bytes into per-page results. Providers never leak page
contents into errors: failures are recorded as the exception class name plus
a short sanitized message, because provider errors can embed file paths or
stream fragments.
"""

from __future__ import annotations

import re

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmarks.pdf_extraction.manifest import Table, TableCell


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


@dataclass
class SDKProvider:
    """Installed ragnor SDK adapters (``pdf-layout``/``pdf-ocr``/``pdf-auto``) as providers.

    ``engine`` is ``native`` (PdfPlumberExtractor), ``ocr`` (TesseractPDFExtractor)
    or ``auto`` (AutoPDFExtractor). Pages are read with ``policy="allow_partial"``
    so unverified OCR text is measured, never approved. Native/auto tables carry
    their real row/column spans, so table structure and merged-cell scores are
    populated; the runner binds them to expected tables by page order.
    """

    engine: str = "native"
    language: str = "eng"
    id: str = ""
    version: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in {"native", "ocr", "auto"}:
            raise ValueError("engine must be native, ocr or auto")
        self._adapter = self._build()
        self.id = f"sdk-{self.engine}"
        self.version = str(self._adapter.version)
        self.config = {
            key: value
            for key, value in self._adapter.options.items()
            if key not in {"verification"}
        }

    def _build(self):
        from ragnor.ingestion.pdf_extractors import (  # noqa: PLC0415
            AutoPDFExtractor,
            PdfPlumberExtractor,
            TesseractPDFExtractor,
        )

        if self.engine == "native":
            return PdfPlumberExtractor(policy="allow_partial")
        if self.engine == "ocr":
            return TesseractPDFExtractor(language=self.language, policy="allow_partial")
        return AutoPDFExtractor(
            native=PdfPlumberExtractor(),
            ocr=TesseractPDFExtractor(language=self.language),
            policy="allow_partial",
        )

    def library_versions(self) -> dict[str, str]:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        found = {}
        for name in ("pypdf", "pdfplumber", "pypdfium2", "pillow"):
            try:
                found[name] = version(name)
            except PackageNotFoundError:
                found[name] = "not-installed"
        return found

    def extract(self, data: bytes, *, name: str = "") -> list[ProviderPage]:
        pages = []
        for outcome in self._adapter.extract_outcomes(data):
            failed = outcome.status.value == "failed"
            pages.append(
                ProviderPage(
                    page=outcome.page,
                    text=_plain_text(outcome.text),
                    status="error" if failed else ("ok" if outcome.text.strip() else "empty"),
                    error="page_failed" if failed else None,
                    blocks=_sdk_blocks(outcome),
                    tables=[
                        Table(
                            region_id="",
                            rows=table.rows,
                            cols=table.cols,
                            cells=[
                                TableCell(c.row, c.col, c.text, c.rowspan, c.colspan)
                                for c in table.cells
                            ],
                        )
                        for table in outcome.tables
                    ],
                )
            )
        return pages


_TAG = re.compile(r"<[^>]+>")


def _plain_text(text: str) -> str:
    """Text view without table markup.

    Native/auto engines embed tables as HTML in ``text`` so chunkers keep the
    structure. Text metrics (CER/WER) compare reading text with the annotated
    text, so markup would be counted as insertions. Table structure is scored
    separately from the cell data, so nothing is hidden from the benchmark.
    """
    return " ".join(_TAG.sub(" ", text).split())


def _sdk_blocks(outcome) -> list[dict] | None:
    """Text of the words inside each detected region; None when no regions exist."""
    if not outcome.regions:
        return None
    blocks = []
    for region in outcome.regions:
        x0, y0, x1, y1 = region.bbox
        text = " ".join(
            word.text
            for word in outcome.words
            if x0 <= (word.bbox[0] + word.bbox[2]) / 2 <= x1
            and y0 <= (word.bbox[1] + word.bbox[3]) / 2 <= y1
        )
        blocks.append({"kind": region.kind, "text": text})
    return blocks


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


def _sdk(engine: str):
    def make() -> SDKProvider:
        return SDKProvider(engine=engine)

    return make


def registry() -> dict[str, Any]:
    """Providers shipped with the benchmark. External adapters register elsewhere."""
    return {
        "pypdf": PyPDFProvider,
        "sdk-native": _sdk("native"),
        "sdk-ocr": _sdk("ocr"),
        "sdk-auto": _sdk("auto"),
    }
