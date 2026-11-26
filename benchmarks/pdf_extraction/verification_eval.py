"""EX-08 evaluation: verifier false positives/negatives over the EX-02 corpus.

Ground-truth annotations stand in for a perfect extraction ("clean" variant).
A "faulty" variant omits one table/figure/heading region per page, the failure
the count and coverage checks exist to catch. This measures the verifiers
only; it says nothing about any real extractor. Run:

    python -m benchmarks.pdf_extraction.verification_eval
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.pdf_extraction.manifest import PageAnnotation, load_manifest
from ragnor.ingestion import PageExtractionOutcome, PageExtractionStatus
from ragnor.ingestion.extraction_types import ExtractionRegion
from ragnor.ingestion.verification import (
    ExpectedPage,
    ExpectedRegion,
    PageVerificationPipeline,
)

MANIFEST = Path(__file__).parent / "fixtures" / "manifest.jsonl"
_KINDS = {"table", "figure", "heading"}


def _norm(page: PageAnnotation, box) -> tuple[float, float, float, float]:
    return (
        max(0.0, box.x0 / page.width),
        max(0.0, box.y0 / page.height),
        min(1.0, box.x1 / page.width),
        min(1.0, box.y1 / page.height),
    )


def _regions(page: PageAnnotation):
    return [r for r in page.regions if not r.excluded and r.kind != "decoration"]


def evaluate() -> dict:
    manifest = load_manifest(MANIFEST)
    pipeline = PageVerificationPipeline()
    per_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pages": 0, "clean_false_positives": 0, "faulty_pages": 0, "faulty_caught": 0}
    )
    for page in manifest.pages:
        if page.expect_error:
            continue
        regions = _regions(page)
        expected = ExpectedPage(
            tuple(ExpectedRegion(r.region_id, r.kind, _norm(page, r.bbox)) for r in regions),
            tuple(_norm(page, r.bbox) for r in regions),
        )

        def outcome(rs, page=page):
            return PageExtractionOutcome(
                page.page,
                PageExtractionStatus.EXTRACTED,
                text="x",
                regions=tuple(ExtractionRegion(r.kind, _norm(page, r.bbox)) for r in rs),
            )

        stats = per_family[page.family]
        stats["pages"] += 1
        clean = pipeline.verify(outcome(regions), expected=expected)
        stats["clean_false_positives"] += clean.status != "verified"
        victim = next((r for r in regions if r.kind in _KINDS), None)
        if victim is not None:
            stats["faulty_pages"] += 1
            faulty = pipeline.verify(
                outcome([r for r in regions if r is not victim]), expected=expected
            )
            stats["faulty_caught"] += faulty.status == "failed"
    total = {k: sum(f[k] for f in per_family.values()) for k in next(iter(per_family.values()))}
    return {
        "note": "annotation-as-extraction with injected omissions; measures verifiers only",
        "per_family": dict(sorted(per_family.items())),
        "total": total,
        "false_positive_rate": total["clean_false_positives"] / max(1, total["pages"]),
        "false_negative_rate": 1 - total["faulty_caught"] / max(1, total["faulty_pages"]),
    }


if __name__ == "__main__":
    json.dump(evaluate(), sys.stdout, indent=2, sort_keys=True)
    print()
