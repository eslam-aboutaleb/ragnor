"""Private, installed SDK worker. Vendor stdout/stderr never reach host logs."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
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
        selected = {
            page.page
            for page in pages
            if engine == "ocr" or (not page.complete and (not page.text.strip() or page.has_images))
        }
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
                if engine == "auto" and native.text.strip() and not page.text.strip():
                    # A failed OCR attempt must not erase recoverable native text.
                    page = replace(
                        native,
                        diagnostics={
                            **native.diagnostics,
                            "ocr_fallback_status": str(page.status),
                        },
                    )
                if engine == "auto" and native.tables and page.provider == "tesseract":
                    page = replace(
                        page,
                        tables=native.tables,
                        diagnostics={**page.diagnostics, "native_table_provider": "pdfplumber"},
                    )
                    if page.text and config["native"]["include_tables"]:
                        page = replace(
                            page,
                            text=" ".join(
                                word.text
                                for word in page.words
                                if not any(
                                    t.bbox[0] <= (word.bbox[0] + word.bbox[2]) / 2 <= t.bbox[2]
                                    and t.bbox[1] <= (word.bbox[1] + word.bbox[3]) / 2 <= t.bbox[3]
                                    for t in native.tables
                                )
                            )
                            + "\n\n"
                            + "\n".join(t.to_html() for t in native.tables),
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


def _watch_parent(parent_pid):
    """Terminate the isolated worker group if its owning host disappears."""
    while True:
        time.sleep(0.25)
        if os.getppid() != parent_pid:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(1)
            os.killpg(os.getpgrp(), signal.SIGKILL)


def _set_resource_limits(limits):
    if limits.max_memory_bytes is not None and not sys.platform.startswith("linux"):
        raise PDFExtractionError("resource_limit")
    if os.name != "posix":
        return
    import resource  # noqa: PLC0415

    def ceiling(kind, requested):
        # Preserve tighter deployment limits inherited from the host.
        previous = resource.getrlimit(kind)
        bounded = min([requested, *(value for value in previous if value >= 0)])
        resource.setrlimit(kind, (bounded, bounded))

    ceiling(resource.RLIMIT_FSIZE, limits.max_output_bytes)
    if limits.max_memory_bytes is not None:
        ceiling(resource.RLIMIT_AS, limits.max_memory_bytes)


def main():
    # Keep an original result channel while silencing Python and C-library output.
    channel = os.dup(1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):

            def terminate(_signum, _frame):
                # Unwind subprocess.run so it reaps its OCR child before exit.
                raise PDFExtractionError("timeout")

            if os.name == "posix":
                signal.signal(signal.SIGTERM, terminate)
            try:
                request = json.loads(sys.argv[1])
                limits = PDFExtractionLimits(**request["limits"])
                _set_resource_limits(limits)
                if os.name == "posix":
                    threading.Thread(
                        target=_watch_parent, args=(request["parent_pid"],), daemon=True
                    ).start()
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
