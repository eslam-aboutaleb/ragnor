"""Compare fixed-corpus quality reports with explicit, reviewable tolerances."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LOWER_IS_BETTER = {
    "mean_page_cer",
    "mean_page_wer",
    "missing_pages",
    "invalid_table_predictions",
    "unsafe_complete_pages",
    "empty_text_regions",
}
HIGHER_IS_BETTER = {
    "mean_region_precision",
    "mean_region_recall",
    "mean_field_task_accuracy",
    "mean_field_answer_coverage",
    "mean_token_recall",
    "mean_numeric_token_recall",
    "mean_region_token_recall",
    "mean_reading_order_pair_accuracy",
    "mean_reading_order_pair_coverage",
    "table_recall_iou50",
    "table_precision_iou50",
    "mean_teds_normalized",
    "mean_teds_structure",
    "mean_cell_exact_accuracy",
    "mean_span_accuracy",
    "mean_numeric_cell_accuracy",
}


def compare(baseline: dict, candidate: dict, *, tolerance=0.0) -> dict:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Quality tolerance must be nonnegative and finite")
    if baseline["corpus_sha256"] != candidate["corpus_sha256"]:
        raise ValueError("Reports use different corpus/annotation manifests")
    if baseline["configuration"]["cohort"] != candidate["configuration"]["cohort"]:
        raise ValueError("Reports use different cohorts")
    if set(baseline["families"]) != set(candidate["families"]):
        raise ValueError("Reports use different family coverage")
    if baseline.get("schema_version") != candidate.get("schema_version"):
        raise ValueError("Reports use different scoring schemas")
    if baseline.get("pages") and candidate.get("pages"):

        def keys(report):
            return {(row["document_id"], row["page"]) for row in report["pages"]}

        if keys(baseline) != keys(candidate):
            raise ValueError("Reports use different page identities")
    checks = []
    sections = {"all": (baseline["summary"], candidate["summary"])}
    sections.update(
        {
            name: (baseline["families"][name], candidate["families"][name])
            for name in baseline["families"]
        }
    )
    for family, (left, right) in sections.items():
        if left["pages"] != right["pages"]:
            raise ValueError("Reports use different page counts")
        for metric in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
            before, after = left.get(metric), right.get(metric)
            if before is None:
                continue
            if not isinstance(before, (int, float)) or not math.isfinite(before):
                raise ValueError("Invalid baseline metric")
            measured = isinstance(after, (int, float)) and math.isfinite(after)
            delta = (after - before) if measured else None
            passed = measured and (
                delta <= tolerance if metric in LOWER_IS_BETTER else -delta <= tolerance
            )
            checks.append(
                {
                    "family": family,
                    "metric": metric,
                    "before": before,
                    "after": after,
                    "passed": passed,
                }
            )
    left_rag, right_rag = baseline.get("retrieval"), candidate.get("retrieval")
    if left_rag:
        if not right_rag or any(
            left_rag.get(key) != right_rag.get(key)
            for key in ["k", "probe_origin", "total_queries"]
        ):
            raise ValueError("Reports use different retrieval probe configurations")
        if {row["probe_id"] for row in left_rag.get("probes", [])} != {
            row["probe_id"] for row in right_rag.get("probes", [])
        }:
            raise ValueError("Reports use different retrieval queries")
        for metric in ["recall_at_k", "mrr", "citation_resolution_rate", "source_answer_hit_rate"]:
            before, after = left_rag.get(metric), right_rag.get(metric)
            if before is not None:
                checks.append(
                    {
                        "family": "retrieval",
                        "metric": metric,
                        "before": before,
                        "after": after,
                        "passed": isinstance(after, (int, float))
                        and math.isfinite(after)
                        and after + tolerance >= before,
                    }
                )
        if (left_rag.get("authorization_and_revision") or {}).get("passed") is True:
            checks.append(
                {
                    "family": "retrieval",
                    "metric": "authorization_and_revision",
                    "passed": (right_rag.get("authorization_and_revision") or {}).get("passed")
                    is True,
                }
            )
    failures_passed = len(candidate["failures"]) <= len(baseline["failures"])
    return {
        "passed": bool(checks) and all(row["passed"] for row in checks) and failures_passed,
        "tolerance": tolerance,
        "provider_failures_non_increasing": failures_passed,
        "checks": checks,
        "interpretation": (
            "Baseline-relative regression check; not independent production qualification"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        tolerance=args.tolerance,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"])}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
