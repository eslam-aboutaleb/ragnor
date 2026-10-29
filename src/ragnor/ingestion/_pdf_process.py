"""Bounded, content-private subprocess transport for optional PDF adapters."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from typing import Any

from ragnor.ingestion.extraction_types import (
    ExtractionCell,
    ExtractionRegion,
    ExtractionTable,
    ExtractionWord,
    PDFExtractionError,
    PDFExtractionLimits,
)
from ragnor.ingestion.extractors import PageExtractionOutcome, PageExtractionStatus


def decode_page(value: dict) -> PageExtractionOutcome:
    value = dict(value)
    value["status"] = PageExtractionStatus(value["status"])
    value["words"] = tuple(
        ExtractionWord(**{**word, "bbox": tuple(word["bbox"])}) for word in value.pop("words", [])
    )
    value["regions"] = tuple(
        ExtractionRegion(**{**region, "bbox": tuple(region["bbox"])})
        for region in value.pop("regions", [])
    )
    value["tables"] = tuple(
        ExtractionTable(
            **{
                **table,
                "bbox": tuple(table["bbox"]),
                "cells": tuple(
                    ExtractionCell(**{**cell, "bbox": tuple(cell["bbox"])})
                    for cell in table["cells"]
                ),
            }
        )
        for table in value.pop("tables", [])
    )
    if type(value["page"]) is not int or not isinstance(value["text"], str):
        raise ValueError("Invalid worker page")
    return PageExtractionOutcome(**value)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() == 0:
        return
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.wait()


def execute(
    data: bytes, engine: str, config: dict, limits: PDFExtractionLimits
) -> tuple[list[PageExtractionOutcome], dict[str, Any]]:
    if len(data) > limits.max_bytes:
        raise PDFExtractionError("input_limit")
    request = {"engine": engine, "config": config, "limits": asdict(limits)}
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(  # noqa: S603 -- fixed SDK worker; config is data, never code
            [sys.executable, "-m", "ragnor.ingestion._pdf_worker", json.dumps(request)],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        try:
            process.communicate(input=data, timeout=limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            raise PDFExtractionError("timeout") from None
        finally:
            _stop(process)
        if process.returncode != 0:
            raise PDFExtractionError("worker_failed")
        output.seek(0)
        raw = output.read(limits.max_output_bytes + 1)
    if len(raw) > limits.max_output_bytes:
        raise PDFExtractionError("output_limit")
    try:
        response = json.loads(raw)
        if "error" in response:
            # Never surface arbitrary worker/provider strings as error messages.
            code = response["error"]
            allowed = {
                "dependency_missing",
                "invalid_document",
                "output_limit",
                "resource_limit",
                "runtime_changed",
                "timeout",
            }
            raise PDFExtractionError(code if code in allowed else "provider_failed")
        if response.get("schema_version") != 1 or not isinstance(response.get("runtime"), dict):
            raise ValueError("Invalid worker protocol")
        pages = [decode_page(value) for value in response["pages"]]
        count = response["page_count"]
        if type(count) is not int or not 1 <= count <= limits.max_pages:
            raise ValueError("Invalid worker page count")
        if [page.page for page in pages] != list(range(1, count + 1)):
            raise ValueError("Missing, duplicated or reordered worker pages")
        return pages, response["runtime"]
    except (KeyError, TypeError, ValueError):
        raise PDFExtractionError("invalid_worker_response") from None
