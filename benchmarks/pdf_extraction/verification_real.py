"""EX-08 evaluation on real extractor output (no injected faults).

Each fixture page is extracted by a real engine (``native`` = pdfplumber,
``ocr`` = Tesseract, ``auto``). The page verifier gets only the independent
expectation built from the annotations (region inventory and content boxes).
Ground truth for "this extraction is faulty" is the EX-02 scorer on the same
run: a page is faulty when CER > 0.05 or region recall < 1. The verifier is
right when it passes a good page and fails or flags a faulty one.

    python -m benchmarks.pdf_extraction.verification_real --engine auto \
        --run /path/to/run-report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.verification_eval import MANIFEST, _norm, _regions
from ragnor.ingestion.pdf_extractors import (
    AutoPDFExtractor,
    PdfPlumberExtractor,
    TesseractPDFExtractor,
)
from ragnor.ingestion.verification import ExpectedPage, ExpectedRegion, PageVerificationPipeline

DOCS = MANIFEST.parent


def make_engine(name: str):
    if name == "native":
        return PdfPlumberExtractor(policy="allow_partial")
    if name == "ocr":
        return TesseractPDFExtractor(policy="allow_partial")
    return AutoPDFExtractor(
        native=PdfPlumberExtractor(),
        ocr=TesseractPDFExtractor(),
        policy="allow_partial",
    )


def truth(report: dict) -> dict[tuple[str, int], bool]:
    """Faulty-page ground truth from an EX-02 run report."""
    out = {}
    for page in report["pages"]:
        m = page["metrics"]
        if page["status"] == "expected-error" or m.get("cer") is None:
            continue
        out[(page["document_id"], page["page"])] = (
            m["cer"] > 0.05 or (m.get("region_recall") or 0) < 1.0
        )
    return out


def evaluate(engine_name: str, report: dict) -> dict:
    manifest = load_manifest(MANIFEST)
    labels = truth(report)
    engine, pipeline = make_engine(engine_name), PageVerificationPipeline()
    outcomes: dict[str, list] = {}
    matrix: Counter = Counter()
    statuses: Counter = Counter()
    for page in manifest.pages:
        key = (page.document_id, page.page)
        if page.expect_error or key not in labels:
            continue
        if page.document_id not in outcomes:
            data = (DOCS / page.file).read_bytes()
            outcomes[page.document_id] = engine.extract_outcomes(data)
        outcome = outcomes[page.document_id][page.page - 1]
        regions = _regions(page)
        expected = ExpectedPage(
            tuple(ExpectedRegion(r.region_id, r.kind, _norm(page, r.bbox)) for r in regions),
            tuple(_norm(page, r.bbox) for r in regions),
        )
        report_v = pipeline.verify(outcome, expected=expected)
        statuses[report_v.status] += 1
        faulty = labels[key]
        flagged = report_v.status != "verified"
        matrix[("faulty" if faulty else "good", "flagged" if flagged else "passed")] += 1
    good = matrix[("good", "flagged")] + matrix[("good", "passed")]
    bad = matrix[("faulty", "flagged")] + matrix[("faulty", "passed")]
    return {
        "engine": engine_name,
        "pages": good + bad,
        "good_pages_wrongly_flagged": matrix[("good", "flagged")],
        "good_pages": good,
        "faulty_pages_missed": matrix[("faulty", "passed")],
        "faulty_pages": bad,
        "false_positive_rate": matrix[("good", "flagged")] / max(1, good),
        "false_negative_rate": matrix[("faulty", "passed")] / max(1, bad),
        "verifier_status": dict(statuses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["native", "ocr", "auto"], required=True)
    parser.add_argument("--run", required=True, help="EX-02 run report JSON for the same engine")
    args = parser.parse_args()
    result = evaluate(args.engine, json.loads(Path(args.run).read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
