"""Report aggregation, slicing and non-regression gates.

Aggregates ignore ``None`` metric values (no ground truth or not applicable
for the provider). Gates compare a candidate report against a checked-in
baseline whose thresholds were established from a measured baseline run,
never invented: each gate names the metric, direction, threshold and the
tolerance granted for provider/library version drift.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HIGHER_IS_BETTER = {
    "region_recall",
    "region_precision",
    "ocr_coverage",
    "reading_order",
    "table_structure",
    "cell_value_accuracy",
    "merged_cell_fidelity",
    "field_accuracy",
}
LOWER_IS_BETTER = {"cer", "wer"}
SCORED_METRICS = sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)

# Metrics whose values are byte-for-byte reproducible for a fixed provider,
# pinned library versions and corpus. Gates only ever cover these; timings,
# memory and cost are reported but never gated.
DETERMINISTIC_METRICS = SCORED_METRICS + [
    "regions_expected",
    "regions_recovered",
    "silent_omission_pages",
    "error_pages",
    "unexpected_success_pages",
]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate_pages(page_rows: list[dict]) -> dict:
    """Aggregate per-page metrics over scored (non-error, expected-content) pages."""
    scored = [p["metrics"] for p in page_rows if p["metrics"] is not None]
    totals = {
        metric: _mean([m[metric] for m in scored if m[metric] is not None])
        for metric in SCORED_METRICS
    }
    totals["pages"] = len(page_rows)
    totals["scored_pages"] = len(scored)
    totals["error_pages"] = sum(1 for p in page_rows if p["status"] == "error")
    totals["expected_error_pages"] = sum(1 for p in page_rows if p["status"] == "expected-error")
    totals["unexpected_success_pages"] = sum(1 for p in page_rows if p["unexpected"])
    totals["silent_omission_pages"] = sum(1 for m in scored if m["silent_omission"])
    totals["regions_expected"] = sum(m["regions_expected"] for m in scored)
    totals["regions_recovered"] = sum(m["regions_recovered"] for m in scored)
    return totals


def family_slices(page_rows: list[dict]) -> dict:
    """Per-family aggregates so regressions are attributable to a document family."""
    families: dict[str, list[dict]] = {}
    for row in page_rows:
        families.setdefault(row["family"], []).append(row)
    return {family: aggregate_pages(rows) for family, rows in sorted(families.items())}


def performance_summary(page_rows: list[dict], peak_memory_bytes: int) -> dict:
    """p50/p95 extraction latency, memory peak and cost per 1k pages."""
    durations = sorted(p["duration_ms"] for p in page_rows if p["duration_ms"] is not None)
    total_cost = sum(p.get("cost_usd") or 0.0 for p in page_rows)
    pages = len(page_rows) or 1
    return {
        "p50_ms": _percentile(durations, 50),
        "p95_ms": _percentile(durations, 95),
        "peak_memory_bytes": peak_memory_bytes,
        "total_cost_usd": total_cost,
        "cost_per_1k_pages_usd": round(total_cost * 1000.0 / pages, 6),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float | None:
    if not sorted_values:
        return None
    rank = max(0, min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1))))
    return round(sorted_values[rank], 3)


def derive_gates(report: dict, tolerances: dict[str, float] | None = None) -> dict:
    """Build a gate table from a measured baseline report.

    Thresholds equal the baseline's observed values; deterministic metrics
    default to zero tolerance because the runner is reproducible for a fixed
    provider and corpus. Callers may pass explicit per-metric tolerances for
    documented provider/library drift.
    """
    tolerances = tolerances or {}
    gates = []
    for metric in DETERMINISTIC_METRICS:
        value = report["totals"].get(metric)
        if value is None:
            continue
        gates.append(
            {
                "metric": metric,
                "direction": "max" if metric in LOWER_IS_BETTER else "min",
                "threshold": value,
                "tolerance": tolerances.get(metric, 0.0),
                "source": "baseline",
            }
        )
    return {
        "rationale": (
            "thresholds are the measured values of the checked-in baseline run; "
            "deterministic metrics carry zero tolerance"
        ),
        "gates": gates,
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Evaluate the candidate against the baseline's gates.

    Returns ``{"passed": bool, "failures": [...], "checks": int}``. Every
    failure names the metric, threshold and observed value; page-level ids
    are available in the candidate report's ``pages``/``failures`` lists.
    """
    gates = baseline.get("gates", {}).get("gates", [])
    failures = []
    for gate in gates:
        observed = candidate["totals"].get(gate["metric"])
        if observed is None:
            failures.append({**gate, "observed": None, "reason": "metric missing in candidate"})
            continue
        if gate["direction"] == "min":
            breached = observed < gate["threshold"] - gate["tolerance"]
        else:
            breached = observed > gate["threshold"] + gate["tolerance"]
        if breached:
            failures.append({**gate, "observed": observed, "reason": "threshold breached"})
    unexpected = candidate["totals"].get("unexpected_success_pages", 0)
    if unexpected:
        failures.append(
            {
                "metric": "unexpected_success_pages",
                "reason": f"{unexpected} page(s) succeeded on inputs expected to fail",
                "observed": unexpected,
            }
        )
    return {"passed": not failures, "failures": failures, "checks": len(gates) + 1}


def load_report(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_report(report: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
