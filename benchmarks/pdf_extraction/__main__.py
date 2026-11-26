"""``python -m benchmarks.pdf_extraction`` — benchmark CLI.

Subcommands:
  validate   schema-validate the manifest and verify fixture checksums
  run        run a provider over the corpus and write a JSON report
  baseline   run a provider and write a baseline report with derived gates
  compare    evaluate a candidate report against a baseline's gates
  probes     run the deterministic RAG probes and write a JSON report

Only metrics, ids and hashes are printed or written. PDF contents never
appear on stdout or in artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.pdf_extraction.manifest import load_manifest, validate_manifest
from benchmarks.pdf_extraction.probes import evaluate_probes, load_probes
from benchmarks.pdf_extraction.providers import registry
from benchmarks.pdf_extraction.report import (
    compare_reports,
    derive_gates,
    load_report,
    write_report,
)
from benchmarks.pdf_extraction.runner import run_benchmark

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = FIXTURES / "manifest.jsonl"
PROBES = FIXTURES / "probes.json"
BASELINE = FIXTURES / "baseline" / "pypdf.json"


def _provider(name: str):
    providers = registry()
    if name not in providers:
        known = ", ".join(sorted(providers))
        raise SystemExit(f"unknown provider {name!r}; available: {known}")
    return providers[name]()


def cmd_validate(_args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST)
    errors = validate_manifest(manifest, FIXTURES)
    for error in errors:
        print(f"INVALID: {error}")
    if errors:
        return 1
    print(
        f"manifest v{manifest.corpus_version}: {len(manifest.documents())} documents, "
        f"{len(manifest.pages)} pages, all checks passed"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST)
    report = run_benchmark(manifest, FIXTURES, _provider(args.provider))
    if args.out:
        write_report(report, args.out)
        print(f"report written to {args.out}")
    _print_summary(report)
    if args.baseline:
        return _gate(report, load_report(args.baseline))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST)
    report = run_benchmark(manifest, FIXTURES, _provider(args.provider))
    report["gates"] = derive_gates(report)
    write_report(report, args.out)
    print(f"baseline with {len(report['gates']['gates'])} gates written to {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    return _gate(load_report(args.candidate), load_report(args.baseline))


def cmd_probes(args: argparse.Namespace) -> int:
    manifest = load_manifest(MANIFEST)
    result = evaluate_probes(manifest, FIXTURES, _provider(args.provider), load_probes(PROBES))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"probe report written to {args.out}")
    print(
        f"probes: {result['probes']} queries, recall@{result['k']}="
        f"{result['recall_at_k']:.3f} mrr={result['mrr']:.3f} "
        f"citation_resolution={result['citation_resolution']:.3f}"
    )
    return 0


def _gate(candidate: dict, baseline: dict) -> int:
    outcome = compare_reports(baseline, candidate)
    for failure in outcome["failures"]:
        print(f"GATE FAILURE: {json.dumps(failure, sort_keys=True)}")
    print(f"gates: {outcome['checks']} checks, {'PASSED' if outcome['passed'] else 'FAILED'}")
    return 0 if outcome["passed"] else 1


def _print_summary(report: dict) -> None:
    totals = report["totals"]
    keys = (
        "region_recall",
        "ocr_coverage",
        "cer",
        "wer",
        "reading_order",
        "cell_value_accuracy",
        "field_accuracy",
    )
    rendered = ", ".join(f"{k}={totals[k]:.3f}" for k in keys if totals.get(k) is not None)
    print(
        f"run {report['run_id']}: {totals['scored_pages']}/{totals['pages']} pages scored, "
        f"{totals['error_pages']} errors, {totals['silent_omission_pages']} silent omissions"
    )
    print(f"metrics: {rendered}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmarks.pdf_extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").set_defaults(func=cmd_validate)

    run = commands.add_parser("run")
    run.add_argument("--provider", default="pypdf")
    run.add_argument("--out", default=None)
    run.add_argument("--baseline", default=None, help="gate against this baseline report")
    run.set_defaults(func=cmd_run)

    baseline = commands.add_parser("baseline")
    baseline.add_argument("--provider", default="pypdf")
    baseline.add_argument("--out", required=True)
    baseline.set_defaults(func=cmd_baseline)

    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.set_defaults(func=cmd_compare)

    probes = commands.add_parser("probes")
    probes.add_argument("--provider", default="pypdf")
    probes.add_argument("--out", default=None)
    probes.set_defaults(func=cmd_probes)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
