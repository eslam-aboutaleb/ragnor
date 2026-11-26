"""Private, installed SDK worker. Vendor stdout/stderr never reach host logs."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
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
    for name in ("pypdf", "pypdfium2", "pillow"):
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
    except Exception as exc:
        # A quota breach is a limit, not a malformed document: report the fixed limit code.
        if "exceeding max_pages" in str(exc):
            raise PDFExtractionError("input_limit") from None
        raise PDFExtractionError("invalid_document") from None
    if sum(len(page.text) for page in baseline) > limits.max_text_chars:
        raise PDFExtractionError("resource_limit")
    budget, config = Budget(limits), request["config"]
    runtime = {"packages": _versions()}
    engine = request["engine"]
    if engine == "render":
        from ragnor.ingestion._pdf_render import render_pages  # noqa: PLC0415

        return {
            "schema_version": 1,
            "page_count": len(baseline),
            "runtime": runtime,
            **render_pages(data, baseline, config, limits),
        }
    if engine != "ocr":
        raise PDFExtractionError("invalid_document")
    selected = {page.page for page in baseline}
    if config.get("page_numbers") is not None:
        requested = set(config["page_numbers"])
        if not requested <= {p.page for p in baseline}:
            raise PDFExtractionError("invalid_document")
        selected &= requested
    pages = baseline
    if selected:
        if os.name != "posix":
            raise PDFExtractionError("provider_failed")
        from ragnor.ingestion._pdf_ocr import extract_ocr  # noqa: PLC0415

        ocr, runtime["tesseract"] = extract_ocr(data, baseline, config, budget, selected)
        pages = [ocr.get(native.page, native) for native in pages]
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
