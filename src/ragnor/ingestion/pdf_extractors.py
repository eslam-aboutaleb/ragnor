"""Optional SDK adapters for native PDF layout, local OCR and page routing.

Local PDF vendors run in bounded subprocesses; hosted OCR uses host transports.
Importing these adapters has no vendor
dependencies. Unverified visual/OCR pages stay partial until a host policy accepts
them; the legacy extract/extract_pages views return text without that guarantee.
"""

from __future__ import annotations

import json
import asyncio
import hashlib
import math
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ragnor.embeddings._safe_config import safe_config
from ragnor.ingestion._pdf_process import execute
from ragnor.ingestion.ocr import OCRProvider, shared_limits
from ragnor.ingestion.extraction_types import PDFExtractionError, PDFExtractionLimits, PageVerifier
from ragnor.ingestion.extractors import Extractor, PageExtractionOutcome, PageExtractionStatus


def _json_options(value: dict) -> dict:
    try:
        serialized = json.dumps(value, allow_nan=False)
        if len(serialized.encode()) > 64_000:
            raise ValueError("Extractor configuration exceeds the size limit")
        return json.loads(serialized)
    except (TypeError, ValueError):
        raise ValueError("Extractor options must be bounded finite JSON values") from None


class _PDFAdapter(Extractor):
    version = "2"
    _engine = ""
    limits: PDFExtractionLimits

    def _configure(
        self,
        config: dict,
        limits: PDFExtractionLimits | None,
        verifier: PageVerifier | None,
        policy: str,
    ) -> None:
        if policy not in {"strict", "review", "allow_partial"}:
            raise ValueError("Invalid extraction policy")
        self._config = _json_options(config)
        if limits is not None and not isinstance(limits, PDFExtractionLimits):
            raise ValueError("limits must be PDFExtractionLimits")
        self.limits = limits or PDFExtractionLimits()
        self._verifier = verifier
        self._policy = policy
        self._verification = None
        if verifier is not None:
            if any(
                not isinstance(getattr(verifier, name, None), str)
                or not getattr(verifier, name).strip()
                for name in ("id", "version")
            ) or not callable(getattr(verifier, "verify", None)):
                raise ValueError("A verifier needs id, version and verify(page)")
            self._verification = _json_options(
                safe_config(
                    {"id": verifier.id, "version": verifier.version, "options": verifier.options}
                )
            )

    @property
    def options(self) -> dict[str, Any]:
        return _json_options(
            {
                **self._config,
                "limits": asdict(self.limits),
                "policy": self._policy,
                "verification": safe_config(self._verification),
            }
        )

    def extract_outcomes(self, data: bytes) -> list[PageExtractionOutcome]:
        pages, runtime = execute(data, self._engine, self._config, self.limits)
        source_hash = hashlib.sha256(data).hexdigest()
        return [
            self._review(
                replace(
                    page,
                    diagnostics={
                        **page.diagnostics,
                        "runtime": runtime,
                        "source_sha256": source_hash,
                    },
                )
            )
            for page in pages
        ]

    async def aextract_outcomes(self, data: bytes) -> list[PageExtractionOutcome]:
        return await asyncio.to_thread(self.extract_outcomes, data)

    async def aextract_pages(self, data: bytes) -> list[tuple[int, str]]:
        return [(page.page, page.text) for page in await self.aextract_outcomes(data)]

    async def aextract(self, data: bytes, *, name: str = "") -> str:
        return "\n\n".join(text for _, text in await self.aextract_pages(data) if text)

    def _review(self, page: PageExtractionOutcome) -> PageExtractionOutcome:
        if (
            self._verifier is None
            or page.status not in {PageExtractionStatus.PARTIAL, PageExtractionStatus.EXTRACTED}
            or not page.text
        ):
            return page
        try:
            accepted = self._verifier.verify(page)
            if type(accepted) is not bool:
                raise TypeError
            identity = _json_options(
                safe_config(
                    {
                        "id": self._verifier.id,
                        "version": self._verifier.version,
                        "options": self._verifier.options,
                    }
                )
            )
            if identity != self._verification:
                raise ValueError("Verification policy changed during extraction")
        except Exception:
            raise PDFExtractionError("verification_failed") from None
        return replace(
            page,
            status=PageExtractionStatus.EXTRACTED if accepted else PageExtractionStatus.PARTIAL,
            reason="host_policy_accepted" if accepted else "host_policy_rejected",
            diagnostics={
                **page.diagnostics,
                "verification": {**self._verification, "accepted": accepted},
            },
        )

    def extract_pages(self, data: bytes) -> list[tuple[int, str]]:
        return [(page.page, page.text) for page in self.extract_outcomes(data)]

    def extract(self, data: bytes, *, name: str = "") -> str:
        return "\n\n".join(text for _, text in self.extract_pages(data) if text)


class PdfPlumberExtractor(_PDFAdapter):
    """Machine-generated PDF text, normalized words/blocks and table cells/spans."""

    id = "pdfplumber"
    _engine = "native"

    def __init__(  # noqa: PLR0913 -- explicit optional provider configuration
        self,
        *,
        table_settings: dict | None = None,
        word_settings: dict | None = None,
        include_tables: bool = True,
        limits: PDFExtractionLimits | None = None,
        verifier: PageVerifier | None = None,
        policy: str = "strict",
    ) -> None:
        if any(
            value is not None and not isinstance(value, dict)
            for value in (table_settings, word_settings)
        ):
            raise ValueError("Table and word settings must be dictionaries")
        if type(include_tables) is not bool:
            raise ValueError("include_tables must be boolean")
        self._configure(
            {
                "table_settings": table_settings or {},
                "word_settings": {"use_text_flow": True, **(word_settings or {})},
                "include_tables": include_tables,
            },
            limits,
            verifier,
            policy,
        )


class TesseractPDFExtractor(_PDFAdapter):
    """Render PDF pages and run host-installed Tesseract, including ara+eng OCR.

    Successful OCR is PARTIAL until accepted by an explicit host verifier. A
    tessdata directory is optional; the worker discovers and hashes system models.
    The process-tree deadline currently requires a POSIX host (Linux/macOS).
    """

    id = "tesseract"
    _engine = "ocr"

    def __init__(  # noqa: PLR0913 -- explicit optional provider configuration
        self,
        *,
        language: str = "eng",
        tessdata_dir: str | Path | None = None,
        executable: str = "tesseract",
        dpi: int = 200,
        psm: int = 3,
        page_timeout: float = 30,
        limits: PDFExtractionLimits | None = None,
        verifier: PageVerifier | None = None,
        policy: str = "strict",
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language):
            raise ValueError("Invalid OCR language selection")
        if type(dpi) is not int or not 1 <= dpi <= 1200:
            raise ValueError("OCR DPI must be between 1 and 1200")
        if type(psm) is not int or not 3 <= psm <= 13:
            raise ValueError("OCR PSM must be between 3 and 13")
        if isinstance(page_timeout, bool) or not math.isfinite(page_timeout) or page_timeout <= 0:
            raise ValueError("OCR timeout must be positive and finite")
        if not isinstance(executable, str) or not executable.strip():
            raise ValueError("OCR executable must be a host executable name or path")
        self._configure(
            {
                "language": language,
                "tessdata_dir": str(Path(tessdata_dir).resolve()) if tessdata_dir else None,
                "executable": executable,
                "dpi": dpi,
                "psm": psm,
                "page_timeout": page_timeout,
            },
            limits,
            verifier,
            policy,
        )

    def extract_outcomes(
        self,
        data: bytes,
        *,
        page_numbers: frozenset[int] | None = None,
        limits: PDFExtractionLimits | None = None,
    ) -> list[PageExtractionOutcome]:
        if os.name != "posix":
            raise PDFExtractionError("ocr_requires_posix_process_cleanup")
        effective = shared_limits(self.limits, limits or self.limits)
        config = dict(self._config)
        if page_numbers is not None:
            if not page_numbers or any(type(p) is not int or p < 1 for p in page_numbers):
                raise ValueError("OCR page selection must contain positive integers")
            config["page_numbers"] = sorted(page_numbers)
        pages, runtime = execute(data, self._engine, config, effective)
        digest = hashlib.sha256(data).hexdigest()
        return [
            self._review(
                replace(
                    page,
                    diagnostics={**page.diagnostics, "runtime": runtime, "source_sha256": digest},
                )
            )
            for page in pages
        ]

    async def aextract_outcomes(self, data: bytes, **kwargs) -> list[PageExtractionOutcome]:
        return await asyncio.to_thread(self.extract_outcomes, data, **kwargs)


class TesseractOCRProvider(TesseractPDFExtractor):
    """Local implementation of the OCRProvider port; existing PDF API preserved."""


class AutoPDFExtractor(_PDFAdapter):
    """Keep complete native pages; OCR uncertain pages and retain native tables.

    Configure verification here, after routing, rather than on child adapters.
    Shared quotas are the strictest of the router and both child configurations.
    """

    id = "pdf-auto"
    _engine = "auto"

    def __init__(  # noqa: PLR0913 -- composable provider and acceptance configuration
        self,
        *,
        native: PdfPlumberExtractor | None = None,
        ocr: OCRProvider | None = None,
        limits: PDFExtractionLimits | None = None,
        verifier: PageVerifier | None = None,
        policy: str = "strict",
    ) -> None:
        native = native or PdfPlumberExtractor()
        ocr = ocr or TesseractPDFExtractor()
        if native._verifier is not None or getattr(ocr, "_verifier", None) is not None:
            raise ValueError("Configure verification on AutoPDFExtractor, not its children")
        if not isinstance(ocr, OCRProvider):
            raise ValueError("ocr must implement OCRProvider")
        self._native = native
        self._ocr_provider = ocr
        shared = shared_limits(limits or PDFExtractionLimits(), native.limits, ocr.limits)
        ocr_config = (
            ocr._config
            if isinstance(ocr, TesseractPDFExtractor)
            else safe_config(
                {
                    "id": ocr.id,
                    "version": ocr.version,
                    "options": ocr.options,
                }
            )
        )
        self._configure({"native": native._config, "ocr": ocr_config}, shared, verifier, policy)

    def extract_outcomes(self, data: bytes) -> list[PageExtractionOutcome]:
        if isinstance(self._ocr_provider, TesseractPDFExtractor):
            return super().extract_outcomes(data)
        from ragnor.ingestion._provider_pdf import extract_auto  # noqa: PLC0415

        return extract_auto(self, data)

    async def aextract_outcomes(self, data: bytes) -> list[PageExtractionOutcome]:
        if isinstance(self._ocr_provider, TesseractPDFExtractor):
            return await super().aextract_outcomes(data)
        from ragnor.ingestion._provider_pdf import aextract_auto  # noqa: PLC0415

        return await aextract_auto(self, data)
