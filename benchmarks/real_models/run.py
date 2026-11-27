"""Run the real-model benchmark suites and write JSON plus a Markdown summary.

    python -m benchmarks.real_models.run --suites docling,embeddings,qdrant,vlm \
        --max-cost-usd 2 --out results/

Only workflow_dispatch runs this with real credentials. A suite that is not
configured is recorded as skipped with the setting names it needs, never with
values. Output never contains keys, request bodies or provider error text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.pdf_extraction.manifest import load_manifest
from benchmarks.pdf_extraction.probes import load_probes
from benchmarks.pdf_extraction.providers import SDKProvider
from benchmarks.pdf_extraction.runner import run_benchmark
from benchmarks.real_models.budget import BudgetExceededError, CostMeter
from benchmarks.real_models.config import SUITES, ConfigError, Settings
from benchmarks.real_models.models import LiteLLMEmbedder, LiteLLMPageReader, ModelCallError
from benchmarks.real_models.retrieval import brute_force, embed_corpus, qdrant_search

FIXTURES = Path(__file__).resolve().parents[1] / "pdf_extraction" / "fixtures"
HARD_CAP_USD = 10.0  # the workflow input cannot raise spend above this


class DoclingProvider(SDKProvider):
    """EX-02 provider around the optional Docling adapter (local models, no cost)."""

    def __init__(self, artifacts_path: str | None = None) -> None:
        self._artifacts_path = artifacts_path
        super().__init__(engine="native")

    def __post_init__(self) -> None:
        from ragnor.ingestion.docling_adapter import DoclingPDFExtractor  # noqa: PLC0415

        self._adapter = DoclingPDFExtractor(
            artifacts_path=self._artifacts_path, do_ocr=False, policy="allow_partial"
        )
        self.id = "docling"
        self.version = str(self._adapter.version)
        self.config = {k: v for k, v in self._adapter.options.items() if k != "verification"}


def _slim(report: dict) -> dict:
    keep = ("run_id", "provider", "totals", "performance", "families", "failures")
    return {k: report[k] for k in keep if k in report}


def _extraction_suite(provider, manifest) -> dict:
    return _slim(run_benchmark(manifest, FIXTURES, provider))


def _real_index(settings, meter, manifest, probes):
    embedder = LiteLLMEmbedder(settings.embedding, meter)
    provider = SDKProvider(engine="auto")
    chunks, vectors = embed_corpus(manifest, FIXTURES, provider, embedder.embed)
    return embedder, chunks, vectors


def run_suites(suites, settings, meter, *, max_pages: int, artifacts_path: str | None) -> dict:
    manifest = load_manifest(FIXTURES / "manifest.jsonl")
    probes = load_probes(FIXTURES / "probes.json")
    results: dict[str, dict] = {}
    cache: dict = {}
    for suite in suites:
        started = time.perf_counter()
        try:
            settings.require(suite)
            if suite == "docling":
                body = _extraction_suite(DoclingProvider(artifacts_path), manifest)
            elif suite == "vlm":
                body = _extraction_suite(
                    LiteLLMPageReader(settings.vlm, meter, max_pages=max_pages), manifest
                )
            else:
                if "index" not in cache:
                    cache["index"] = _real_index(settings, meter, manifest, probes)
                embedder, chunks, vectors = cache["index"]
                if suite == "embeddings":
                    body = brute_force(chunks, vectors, probes, embedder.embed)
                    body["embed_call_ms_p50"] = round(
                        sorted(embedder.latencies_ms)[len(embedder.latencies_ms) // 2], 2
                    )
                else:
                    from qdrant_client import QdrantClient  # noqa: PLC0415

                    body = qdrant_search(
                        QdrantClient(url=settings.qdrant_url),
                        chunks,
                        vectors,
                        probes,
                        embedder.embed,
                    )
            results[suite] = {
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 1),
                **body,
            }
        except ConfigError as exc:
            results[suite] = {"status": "skipped", "reason": str(exc)}
        except BudgetExceededError as exc:
            results[suite] = {"status": "stopped_by_cost_cap", "reason": str(exc)}
        except ModelCallError as exc:
            results[suite] = {"status": "failed", "reason": str(exc)}
        except Exception as exc:  # never include exception text: it can echo request data
            results[suite] = {"status": "failed", "reason": f"{type(exc).__name__}"}
    return results


def summary(results: dict, meter: CostMeter, settings: Settings) -> str:
    lines = ["# Real-model benchmark", ""]
    lines.append(
        f"Spend: {meter.spent_usd:.4f} of {meter.max_usd:.2f} USD cap, {meter.calls} calls."
    )
    pub = settings.public()
    lines.append(f"Embedding model: `{pub['embedding_model']}`. VLM: `{pub['vlm_model']}`.")
    lines += ["", "| Suite | Status | Key numbers |", "|---|---|---|"]
    for suite, r in results.items():
        if r["status"] != "ok":
            lines.append(f"| {suite} | {r['status']} | {r.get('reason', '')} |")
            continue
        if "recall_at_k" in r:
            nums = (
                f"Recall@{r['k']} {r['recall_at_k']}, MRR {r['mrr']}, {r['indexed_chunks']} chunks"
            )
            if "query_ms_p50" in r:
                nums += f", query p50 {r['query_ms_p50']} ms, p95 {r['query_ms_p95']} ms"
        else:
            t = r["totals"]
            nums = (
                f"CER {t.get('cer')}, region recall {t.get('region_recall')}, "
                f"p50 {r['performance']['p50_ms']} ms/page, "
                f"cost/1k pages {r['performance']['cost_per_1k_pages_usd']} USD"
            )
        lines.append(f"| {suite} | ok | {nums} |")
    lines += [
        "",
        "Corpus: in-repo synthetic EX-02 fixtures (30 pages, 8 probes). "
        "Single run: no confidence intervals.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--max-vlm-pages", type=int, default=40)
    parser.add_argument("--docling-artifacts")
    parser.add_argument("--out", default="bench-results")
    parser.add_argument("--dry-run", action="store_true", help="validate settings, call nothing")
    args = parser.parse_args(argv)
    chosen = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in SUITES]
    if unknown:
        print(f"unknown suites: {', '.join(unknown)}", file=sys.stderr)
        return 2
    if not 0 < args.max_cost_usd <= HARD_CAP_USD:
        print(f"--max-cost-usd must be in (0, {HARD_CAP_USD}]", file=sys.stderr)
        return 2
    settings = Settings.from_env(dict(os.environ))
    if args.dry_run:
        status = {}
        for suite in chosen:
            try:
                settings.require(suite)
                status[suite] = "configured"
            except ConfigError as exc:
                status[suite] = str(exc)
        print(json.dumps({"settings": settings.public(), "suites": status}, indent=2))
        return 0
    meter = CostMeter(args.max_cost_usd)
    results = run_suites(
        chosen, settings, meter, max_pages=args.max_vlm_pages, artifacts_path=args.docling_artifacts
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings.public(), "cost": meter.to_dict(), "results": results}
    (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    text = summary(results, meter, settings)
    (out / "summary.md").write_text(text)
    print(text)
    failed = [s for s, r in results.items() if r["status"] in {"failed"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
