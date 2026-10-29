"""Private, installed SDK worker. Vendor stdout/stderr never reach host logs."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version

from ragnor.ingestion.extraction_types import PDFExtractionError, PDFExtractionLimits
from ragnor.ingestion.extractors import PDFExtractor


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.used = {}

    def add(self, key, count):
        self.used[key] = self.used.get(key, 0) + count
        if self.used[key] > getattr(self.limits, "max_" + key):
            raise PDFExtractionError("resource_limit")


def _versions():
    versions = {}
    for name in ("pypdf", "pdfplumber", "pdfminer.six", "pypdfium2", "pillow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def process(data, request):
    limits = PDFExtractionLimits(**request["limits"])
    try:
        baseline = PDFExtractor(
            options={"max_bytes": limits.max_bytes, "max_pages": limits.max_pages}
        ).extract_outcomes(data)
    except ImportError:
        raise
    except Exception:
        raise PDFExtractionError("invalid_document") from None
    if sum(len(page.text) for page in baseline) > limits.max_text_chars:
        raise PDFExtractionError("resource_limit")
    budget, config = Budget(limits), request["config"]
    runtime = {"packages": _versions()}
    engine = request["engine"]
    if engine not in {"native", "ocr", "auto"}:
        raise PDFExtractionError("invalid_document")
    if engine in {"native", "auto"}:
        from ragnor.ingestion._pdf_native import extract_native  # noqa: PLC0415

        pages = extract_native(
            data, baseline, config["native"] if engine == "auto" else config, budget
        )
    else:
        pages = baseline
    if engine in {"ocr", "auto"}:
        selected = {page.page for page in pages if engine == "ocr" or not page.complete}
        if selected:
            if os.name != "posix":
                raise PDFExtractionError("provider_failed")
            from ragnor.ingestion._pdf_ocr import extract_ocr  # noqa: PLC0415

            ocr, runtime["tesseract"] = extract_ocr(
                data, baseline, config["ocr"] if engine == "auto" else config, budget, selected
            )
            merged = []
            for native in pages:
                page = ocr.get(native.page, native)
                if engine == "auto" and native.tables:
                    page = replace(
                        page,
                        tables=native.tables,
                        diagnostics={**page.diagnostics, "native_table_provider": "pdfplumber"},
                    )
                    if page.text and config["native"]["include_tables"]:
                        page = replace(
                            page,
                            text=page.text + "\n\n" + "\n".join(t.to_html() for t in native.tables),
                        )
                merged.append(page)
            pages = merged
    if sum(len(page.text) for page in pages) > limits.max_text_chars:
        raise PDFExtractionError("resource_limit")
    return {
        "schema_version": 1,
        "pages": [asdict(page) for page in pages],
        "page_count": len(baseline),
        "runtime": runtime,
    }


def main():
    # Keep an original result channel while silencing Python and C-library output.
    channel = os.dup(1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                request = json.loads(sys.argv[1])
                limits = PDFExtractionLimits(**request["limits"])
                data = sys.stdin.buffer.read(limits.max_bytes + 1)
                result = process(data, request)
                raw = json.dumps(result, ensure_ascii=True, allow_nan=False).encode()
                if len(raw) > limits.max_output_bytes:
                    raise PDFExtractionError("output_limit")
            except ImportError:
                raw = b'{"error":"dependency_missing"}'
            except PDFExtractionError as exc:
                raw = json.dumps({"error": exc.code}).encode()
            except Exception:
                raw = b'{"error":"provider_failed"}'
    with os.fdopen(channel, "wb") as output:
        output.write(raw)


if __name__ == "__main__":
    main()
