"""Routing shared by PDF OCR backends that execute in the host application."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import replace

from ragnor.embeddings._safe_config import safe_config
from ragnor.ingestion._pdf_process import execute
from ragnor.ingestion.extraction_types import (
    PDFExtractionError,
    PageExtractionOutcome,
    PageExtractionStatus,
)


def _identity(provider):
    return safe_config(
        {"id": provider.id, "version": provider.version, "options": provider.options}
    )


def _native(adapter, data):
    if _identity(adapter._ocr_provider) != adapter._config["ocr"]:
        raise PDFExtractionError("runtime_changed")
    pages, runtime = execute(data, "native", adapter._config["native"], adapter.limits)
    digest = hashlib.sha256(data).hexdigest()
    pages = [
        replace(page, diagnostics={**page.diagnostics, "runtime": runtime, "source_sha256": digest})
        for page in pages
    ]
    selected = frozenset(
        page.page
        for page in pages
        if not page.complete and (not page.text.strip() or page.has_images)
    )
    return pages, selected


def _merge(adapter, pages, selected, outcomes):
    if _identity(adapter._ocr_provider) != adapter._config["ocr"]:
        raise PDFExtractionError("runtime_changed")
    if any(not isinstance(p, PageExtractionOutcome) for p in outcomes) or [
        p.page for p in outcomes
    ] != [p.page for p in pages]:
        raise PDFExtractionError("invalid_worker_response")
    by_page = {p.page: p for p in outcomes}
    result = []
    for native in pages:
        page = by_page[native.page] if native.page in selected else native
        if native.page in selected:
            if native.text.strip() and (
                not page.text.strip() or page.status == PageExtractionStatus.FAILED
            ):
                page = replace(
                    native,
                    status=PageExtractionStatus.FAILED,
                    reason="ocr_failed_native_text_retained",
                    diagnostics={
                        **native.diagnostics,
                        "ocr_fallback_status": str(page.status),
                        "ocr_provider": page.provider,
                        "ocr_diagnostics": page.diagnostics,
                    },
                )
            elif native.tables:
                # Keep native table evidence separately; hosted text has no boxes
                # with which to remove duplicate values reliably.
                page = replace(
                    page,
                    tables=native.tables,
                    diagnostics={**page.diagnostics, "native_table_provider": "pdfplumber"},
                )
        result.append(adapter._review(page))
    if sum(len(p.text) for p in result) > adapter.limits.max_text_chars:
        raise PDFExtractionError("resource_limit")
    if sum(len(p.text.encode()) for p in result) > adapter.limits.max_output_bytes:
        raise PDFExtractionError("output_limit")
    return result


def extract_auto(adapter, data):
    deadline = time.monotonic() + adapter.limits.timeout_seconds
    pages, selected = _native(adapter, data)
    if not selected:
        return [adapter._review(page) for page in pages]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PDFExtractionError("timeout")
    outcomes = adapter._ocr_provider.extract_outcomes(
        data, page_numbers=selected, limits=replace(adapter.limits, timeout_seconds=remaining)
    )
    if time.monotonic() >= deadline:
        raise PDFExtractionError("timeout")
    return _merge(adapter, pages, selected, outcomes)


async def aextract_auto(adapter, data):
    deadline = time.monotonic() + adapter.limits.timeout_seconds
    try:
        async with asyncio.timeout(adapter.limits.timeout_seconds):
            pages, selected = await asyncio.to_thread(_native, adapter, data)
            if not selected:
                return [adapter._review(page) for page in pages]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PDFExtractionError("timeout")
            outcomes = await adapter._ocr_provider.aextract_outcomes(
                data,
                page_numbers=selected,
                limits=replace(adapter.limits, timeout_seconds=remaining),
            )
            return _merge(adapter, pages, selected, outcomes)
    except TimeoutError:
        raise PDFExtractionError("timeout") from None
