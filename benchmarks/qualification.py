"""Hash-bound review packets and fail-closed, host-approved release policies.

This module records external review evidence. It never invents human approval,
production tolerances, corpus-family labels or host billing prices.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path


def reference_fingerprint(document: dict, page: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {"source_hash": document["source_hash"], "reference": page},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def prepare_packet(
    manifest_path: Path,
    output: Path,
    *,
    count=50,
    baseline_path: Path | None = None,
    predictions_path: Path | None = None,
) -> dict:
    import pypdfium2 as pdfium  # noqa: PLC0415
    from PIL import ImageDraw  # noqa: PLC0415
    from benchmarks.run_quality import validate_manifest  # noqa: PLC0415

    if not 1 <= count <= 50:
        raise ValueError("Review packets contain at most 50 pages")
    manifest = validate_manifest(manifest_path)
    predictions = {}
    if predictions_path is not None:
        for prediction in json.loads(predictions_path.read_text(encoding="utf-8")):
            key = f"{prediction['document_id']}:{prediction['page']}"
            digest = prediction.get("evidence_sha256")
            if (
                key in predictions
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("SDK review predictions need unique pages and evidence hashes")
            predictions[key] = prediction
    groups = defaultdict(list)
    for document in manifest["documents"]:
        for page in document["pages"]:
            groups[document["family"]].append((document, page))
    selected = []
    while len(selected) < count and any(groups.values()):
        for family in sorted(groups):
            if groups[family] and len(selected) < count:
                selected.append(groups[family].pop(0))
    output.mkdir(parents=True, exist_ok=True)
    records, sections = [], []
    for index, (document, page) in enumerate(selected):
        key = f"{document['document_id']}:{page['page']}"
        prediction = predictions.get(key)
        if predictions_path is not None and prediction is None:
            raise ValueError("SDK review page is missing its prediction")
        with pdfium.PdfDocument(manifest_path.parent / document["path"]) as pdf:
            pdf.init_forms()
            source = pdf[page["page"] - 1]
            source.set_cropbox(*source.get_mediabox())
            scale = min(150 / 72, (20_000_000 / (source.get_width() * source.get_height())) ** 0.5)
            bitmap = source.render(scale=scale)
            image = bitmap.to_pil().convert("RGB").copy()
            bitmap.close()
            source.close()
        draw = ImageDraw.Draw(image)
        for region in page["regions"]:
            x0, y0, x1, y1 = region["bbox"]
            draw.rectangle(
                [x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height],
                outline="red",
                width=2,
            )
        name = f"preview-{index:03d}.png"
        image.save(output / name)
        records.append(
            {
                "key": key,
                "reference_sha256": reference_fingerprint(document, page),
                **(
                    {
                        "extraction_evidence_sha256": prediction["evidence_sha256"],
                        "extraction_status": prediction["status"],
                        "extraction_review_status": "pending",
                        "extraction_reviewer": None,
                        "extraction_reviewed_at": None,
                    }
                    if prediction
                    else {}
                ),
                "status": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "families": [],
                "notes": None,
            }
        )
        sections.append(
            "<section><h2>"
            + html.escape(key)
            + '</h2><img width="700" src="'
            + name
            + '" alt="Source page and reference boxes"><h3>Reference transcription</h3>'
            + '<pre dir="auto">'
            + html.escape(page["expected_text"])
            + "</pre><h3>Fields and reading order</h3><pre>"
            + html.escape(
                json.dumps(
                    {
                        "critical_fields": page.get("critical_fields", []),
                        "reading_order": page.get("reading_order"),
                        "caption_links": page.get("caption_links", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre>"
            + (
                "<h3>SDK extraction (unapproved)</h3><pre dir=auto>"
                + html.escape(prediction["text"])
                + "</pre><h3>SDK tables</h3><pre>"
                + html.escape(json.dumps(prediction.get("tables"), ensure_ascii=False, indent=2))
                + "</pre>"
                if prediction
                else ""
            )
            + "</section>"
        )
    packet = {
        **(
            {"predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest()}
            if predictions_path is not None
            else {}
        ),
        "schema_version": 1,
        "corpus_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "instructions": "Human reviewer checks source/transcription, omitted regions, "
        "values/spans, "
        "order, captions and family labels. Record ambiguities; do not approve guesses.",
        "reviews": records,
    }
    baseline = json.loads(baseline_path.read_text()) if baseline_path else None
    template = {
        "approved_by": None,
        "approved_at": None,
        "allowed_policy_approvers": [],
        "allowed_reviewers": [],
        "minimum_reviewed_pages": 30,
        "required_families": [
            "plain_digital",
            "multicolumn",
            "scan",
            "mixed",
            "rotated",
            "ruled_table",
            "borderless_table",
            "split_merged_table",
            "multilevel_headings",
            "arabic_rtl",
            "multilingual",
            "invoice",
            "contract",
            "chart",
            "formula",
            "dense_financial",
            "handwriting",
            "corrupted",
            "oversized",
        ],
        "required_summary_metrics": [
            "mean_page_cer",
            "mean_region_recall",
            "mean_reading_order_pair_coverage",
            "mean_cell_exact_accuracy",
            "mean_span_accuracy",
            "mean_field_task_accuracy",
        ],
        "quality_tolerance": 0,
        "baseline_sha256": hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
        if baseline
        else None,
        "notes": (
            "Unapproved template; maintainers must review scope, required metrics and tolerances. "
            "Human family labels must match inspected content, never guesses."
        ),
    }
    (output / "policy-template.json").write_text(
        json.dumps(template, indent=2) + "\n", encoding="utf-8"
    )
    (output / "reviews.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>PDF review packet</title>'
        + "<h1>Pending human review</h1>"
        + "".join(sections),
        encoding="utf-8",
    )
    return packet


def qualify(
    report: dict,
    manifest: dict,
    policy: dict,
    reviews: dict | None,
    *,
    baseline_report: dict | None = None,
) -> dict:
    from benchmarks.compare_quality import compare  # noqa: PLC0415

    checks = []

    def check(name, passed, detail):
        checks.append({"gate": name, "passed": bool(passed), "detail": detail})

    approved_by = policy.get("approved_by")
    check(
        "policy_approval",
        isinstance(approved_by, str)
        and bool(approved_by.strip())
        and approved_by in policy.get("allowed_policy_approvers", [])
        and bool(policy.get("approved_at")),
        "Host-maintainer-approved policy and identity allowlist required",
    )
    valid = []
    index = {
        f"{doc['document_id']}:{page['page']}": (doc, page)
        for doc in manifest["documents"]
        for page in doc["pages"]
    }
    if report.get("pages"):
        executed = {(row["document_id"], row["page"]) for row in report["pages"]}
        index = {
            key: pair
            for key, pair in index.items()
            if (pair[0]["document_id"], pair[1]["page"]) in executed
        }
    if reviews and reviews.get("corpus_sha256") == report["corpus_sha256"]:
        seen = set()
        for review in reviews.get("reviews", []):
            pair = index.get(review.get("key"))
            if (
                pair
                and review.get("key") not in seen
                and review.get("status") == "approved"
                and isinstance(review.get("reviewer"), str)
                and bool(review["reviewer"].strip())
                and review.get("reviewer") in policy.get("allowed_reviewers", [])
                and review.get("reviewed_at")
                and review.get("reference_sha256") == reference_fingerprint(*pair)
            ):
                valid.append(review)
                seen.add(review["key"])
    check(
        "human_annotation_review",
        len(valid) >= max(30, policy.get("minimum_reviewed_pages", 30)),
        {"valid_reviews": len(valid), "required": policy.get("minimum_reviewed_pages", 30)},
    )
    covered = {family for review in valid for family in review.get("families", [])}
    missing = sorted(set(policy.get("required_families", [])) - covered)
    check(
        "reviewed_family_coverage", not missing and bool(policy.get("required_families")), missing
    )
    baseline = policy.get("baseline_sha256")
    actual = (
        hashlib.sha256(json.dumps(baseline_report, sort_keys=True).encode()).hexdigest()
        if baseline_report
        else None
    )
    check(
        "approved_baseline",
        baseline is not None and baseline == actual,
        "Canonical baseline report hash must match approved policy",
    )
    if baseline_report and baseline == actual:
        try:
            result = compare(baseline_report, report, tolerance=policy.get("quality_tolerance", 0))
            check("quality_non_regression", result["passed"], result)
        except (ValueError, TypeError):
            check("quality_non_regression", False, "Incompatible reports or invalid tolerance")
    else:
        check("quality_non_regression", False, "No bound approved baseline")
    check(
        "metric_policy",
        bool(policy.get("required_summary_metrics")),
        "Explicit required quality metrics must be approved",
    )
    for field in policy.get("required_summary_metrics", []):
        check(
            "measured:" + field,
            report["summary"].get(field) is not None,
            "Required metric cannot be unmeasured",
        )
    rag = report.get("retrieval") or {}
    check(
        "rag_and_citation_contract",
        rag.get("total_queries", 0) > 0
        and rag.get("citation_resolution_rate") == 1.0
        and (rag.get("authorization_and_revision") or {}).get("passed") is True,
        "SDK retrieval, citations and deny/revision probes required",
    )
    check(
        "complete_cost_scope",
        report.get("cost", {}).get("total_usd_per_1000_pages") is not None,
        "A compute price with provenance is required for a total monetary estimate",
    )
    return {
        "qualified": all(row["passed"] for row in checks),
        "checks": checks,
        "interpretation": (
            "Only recorded host approvals are evaluated; no human review is synthesized"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--manifest", type=Path, required=True)
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--count", type=int, default=50)
    packet.add_argument("--baseline", type=Path)
    packet.add_argument("--predictions", type=Path)
    gate = sub.add_parser("check")
    for name in ["report", "manifest", "policy", "reviews", "baseline", "output"]:
        gate.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "packet":
        result = prepare_packet(
            args.manifest,
            args.output,
            count=args.count,
            baseline_path=args.baseline,
            predictions_path=args.predictions,
        )
        print(json.dumps({"review_pages": len(result["reviews"]), "status": "pending"}))
        return 0

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    from benchmarks.run_quality import validate_manifest  # noqa: PLC0415

    manifest = validate_manifest(args.manifest)
    report = load(args.report)
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != report["corpus_sha256"]:
        raise ValueError("Report is not bound to this corpus manifest")
    result = qualify(
        report,
        manifest,
        load(args.policy),
        load(args.reviews),
        baseline_report=load(args.baseline),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "qualified": result["qualified"],
                "failed_gates": sum(not r["passed"] for r in result["checks"]),
            }
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
