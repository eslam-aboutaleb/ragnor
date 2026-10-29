"""Offline PDF smoke benchmark; no provider text or exception messages in reports.

Run: python -m benchmarks.pdf_extraction --output /tmp/pdf-report.json
This synthetic gate is not the manually validated EX-02 acceptance corpus.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import logging
import multiprocessing
import os
import signal
import platform
import statistics
import tracemalloc
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from ragnor.embeddings._safe_config import safe_config
from ragnor.ingestion.extractors import PDFExtractor, PageExtractionStatus

DEFAULT_MANIFEST = Path(__file__).parent / "data/smoke/manifest.json"


class ProviderTimeoutError(Exception):
    """A benchmark provider exceeded the host's per-document deadline."""


def _provider_process(provider: Any, data: bytes, connection: Any) -> None:
    if os.name == "posix":
        os.setsid()
    logging.disable(logging.CRITICAL)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            try:
                connection.send((True, provider.extract_outcomes(data)))
            except Exception:
                connection.send((False, None))
    finally:
        connection.close()


def _extract(provider: Any, data: bytes, timeout: float | None):
    if timeout is None:
        return provider.extract_outcomes(data)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_provider_process, args=(provider, data, sender))
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise ProviderTimeoutError
        success, result = receiver.recv()
        if not success:
            raise RuntimeError("Provider failed")
        return result
    finally:
        if process.pid is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                with contextlib.suppress(ProcessLookupError):
                    if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    with contextlib.suppress(ProcessLookupError):
                        if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    process.join(timeout=1)
        receiver.close()
        sender.close()


def edit_distance(expected: Sequence, actual: Sequence) -> int:
    """Unit-cost Levenshtein distance, using linear working memory."""
    try:
        from rapidfuzz.distance.Levenshtein import distance  # noqa: PLC0415
    except ImportError:
        pass
    else:
        return distance(expected, actual)
    row = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        following = [i]
        for j, right in enumerate(actual, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = following
    return row[-1]


def text_error_rate(expected: Sequence, actual: Sequence) -> float:
    # An empty reference has rate 0 for empty output and 1 for unexpected output.
    if not expected:
        return float(bool(actual))
    return edit_distance(expected, actual) / len(expected)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("documents"):
        raise ValueError("Unsupported or empty benchmark manifest")
    ids: set[str] = set()
    for doc in manifest["documents"]:
        identity = doc["document_id"]
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Duplicate or invalid document id")
        ids.add(identity)
        if not doc.get("license") or not doc.get("family"):
            raise ValueError("Missing fixture license or family")
        source = (path.parent / doc["path"]).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("Fixture path escapes manifest directory")
        if hashlib.sha256(source.read_bytes()).hexdigest() != doc["source_hash"]:
            raise ValueError("Fixture checksum mismatch")
        pages = doc["pages"]
        if not pages or [page["page"] for page in pages] != list(range(1, len(pages) + 1)):
            raise ValueError("Fixture pages must be contiguous, ordered and unique")
        for page in pages:
            if "expected_status" in page:
                PageExtractionStatus(page["expected_status"])
            elif manifest.get("quality_gate") != "report_only":
                raise ValueError("A gated fixture requires reference page status")
            if not isinstance(page["expected_text"], str):
                raise ValueError("Invalid reference text")
    return manifest


def score_pages(
    expected: list[dict], actual: list, *, normalize: bool = False
) -> list[dict[str, Any]]:
    numbers = [outcome.page for outcome in actual]
    if len(numbers) != len(set(numbers)) or any(
        not isinstance(number, int) or number < 1 for number in numbers
    ):
        raise ValueError("Provider returned duplicate or invalid page numbers")
    by_page = {outcome.page: outcome for outcome in actual}
    rows = []
    safe_statuses = {"extracted", "verified_blank", "policy_excluded"}
    for reference in expected:
        outcome = by_page.pop(reference["page"], None)
        text = outcome.text if outcome is not None else ""
        reference_text = reference["expected_text"]
        if normalize:
            reference_text, text = " ".join(reference_text.split()), " ".join(text.split())
        status = str(outcome.status) if outcome is not None else "missing"
        expected_status = reference.get("expected_status")
        rows.append(
            {
                "page": reference["page"],
                "status": status,
                "expected_status": expected_status,
                "missing": outcome is None,
                "unexpected": False,
                "status_match": status == expected_status if expected_status is not None else None,
                "unsafe_complete": (
                    status in safe_statuses
                    and expected_status is not None
                    and expected_status not in safe_statuses
                ),
                "character_error_rate": text_error_rate(reference_text, text),
                "word_error_rate": text_error_rate(reference_text.split(), text.split()),
            }
        )
    for page in sorted(by_page):
        rows.append(
            {
                "page": page,
                "status": str(by_page[page].status),
                "expected_status": None,
                "missing": False,
                "unexpected": True,
                "status_match": False,
                "unsafe_complete": False,
                "character_error_rate": None,
                "word_error_rate": None,
            }
        )
    return rows


def _summary(rows: list[dict]) -> dict[str, Any]:
    errors = [
        row["character_error_rate"] for row in rows if row["character_error_rate"] is not None
    ]
    words = [row["word_error_rate"] for row in rows if row["word_error_rate"] is not None]
    return {
        "pages": len(rows),
        "missing_pages": sum(row["missing"] for row in rows),
        "unexpected_pages": sum(row["unexpected"] for row in rows),
        "status_mismatches": sum(row["status_match"] is False for row in rows),
        "status_unmeasured_pages": sum(row["status_match"] is None for row in rows),
        "unsafe_complete_pages": (
            sum(row["unsafe_complete"] for row in rows)
            if any(row["expected_status"] is not None for row in rows)
            else None
        ),
        "mean_character_error_rate": statistics.mean(errors) if errors else None,
        "mean_word_error_rate": statistics.mean(words) if words else None,
    }


def run_benchmark(
    path: Path = DEFAULT_MANIFEST, provider: Any = None, *, timeout_seconds: float | None = None
) -> dict[str, Any]:
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Provider timeout must be positive and finite")
    manifest = load_manifest(path)
    provider = PDFExtractor() if provider is None else provider
    rows, durations, failures = [], [], []
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        for doc in manifest["documents"]:
            data = (path.parent / doc["path"]).read_bytes()
            started = perf_counter()
            try:
                outcomes = _extract(provider, data, timeout_seconds)
                scored = score_pages(
                    doc["pages"],
                    outcomes,
                    normalize=manifest.get("text_normalization") == "whitespace",
                )
            except Exception as exc:
                # Provider exception names/messages may also contain private information.
                reason = (
                    "provider_timeout"
                    if isinstance(exc, ProviderTimeoutError)
                    else "provider_failure"
                )
                failures.append({"document_id": doc["document_id"], "reason": reason})
                scored = score_pages(doc["pages"], [])
            durations.append(perf_counter() - started)
            rows.extend(
                {"document_id": doc["document_id"], "family": doc["family"], **row}
                for row in scored
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing:
            tracemalloc.stop()
    identity = {
        "id": provider.id,
        "version": provider.version,
        "options": safe_config(provider.options),
    }
    corpus_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    run_id = hashlib.sha256(
        json.dumps([corpus_hash, identity, timeout_seconds], sort_keys=True).encode()
    ).hexdigest()
    summary = _summary(rows)
    try:
        pypdf_version = version("pypdf")
    except PackageNotFoundError:
        pypdf_version = None
    # Exact synthetic reference expectations, not a production accuracy threshold.
    passed = (
        not failures
        and not summary["status_mismatches"]
        and not summary["mean_character_error_rate"]
        and not summary["unexpected_pages"]
    )
    report_only = manifest.get("quality_gate") == "report_only"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "corpus_hash": corpus_hash,
        "corpus_provenance": safe_config(manifest.get("provenance", {})),
        "annotation_review": manifest.get("annotation_review", "unreviewed"),
        "provider": identity,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "pypdf": pypdf_version,
        },
        "summary": summary,
        "pages": rows,
        "failures": failures,
        "families": {
            family: _summary([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
        "performance": {
            "document_p50_seconds": statistics.median(durations),
            "document_p95_seconds": sorted(durations)[math.ceil(len(durations) * 0.95) - 1],
            "python_allocation_peak_bytes": peak,
            "memory_scope": "runner_only_excludes_provider_child"
            if timeout_seconds is not None
            else "in_process_python_allocations",
            "cost_per_1000_pages": None,
        },
        "unmeasured": [
            "region_precision_recall",
            "ink_coverage",
            "reading_order",
            "table_teds",
            "cell_value_accuracy",
            "merged_cell_fidelity",
            "critical_fields",
            "citation_resolution",
            "retrieval_recall_at_k",
            "retrieval_mrr",
            "provider_cost",
        ],
        "synthetic_gate_passed": None if report_only else bool(passed),
        "quality_gate": "report_only" if report_only else "synthetic_exact",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", help="Host-supplied module:factory; defaults to pypdf")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10,
        help="Per-document deadline in a spawned process (default: 10)",
    )
    args = parser.parse_args()
    if args.manifest is None:
        from benchmarks.smoke_corpus import create_smoke_corpus  # noqa: PLC0415

        args.manifest = create_smoke_corpus(DEFAULT_MANIFEST.parent)
    provider = None
    if args.provider:
        module, factory = args.provider.split(":", 1)
        provider = getattr(importlib.import_module(module), factory)()
    try:
        report = run_benchmark(args.manifest, provider, timeout_seconds=args.timeout_seconds)
    except (ValueError, KeyError, OSError):
        parser.exit(2, "Invalid or unreadable benchmark corpus; no source data printed.\n")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"run_id": report["run_id"], "synthetic_gate_passed": report["synthetic_gate_passed"]}
        )
    )
    return 0 if report["quality_gate"] == "report_only" or report["synthetic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
