# ruff: noqa: PLR0913, PLR0917
"""Compare full layout detection with the guarded recipe fast path."""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any

from ragnor.ingestion.extraction_types import (
    EvidenceAllowlistVerifier,
    PageExtractionOutcome,
    PageExtractionStatus,
    extraction_evidence_hash,
)
from ragnor.ingestion.recipes import (
    CountCoverageVerifier,
    DocumentFingerprint,
    ExtractedRegion,
    ExtractionRecipe,
    FamilyClassifier,
    FamilyCriteria,
    InMemoryRecipeStore,
    RecipeAnchor,
    RecipeFastPath,
    RecipeLearner,
    RecipeProvenance,
    RecipeRegion,
    RecipeValidity,
)

TENANT = "bench"
VALIDITY = RecipeValidity.for_options(
    {"dpi": 200},
    extractor="synthetic",
    extractor_version="1",
    model="none",
    layout_schema="3",
    rotation=0,
    template_version="1",
)


@dataclass
class Doc:
    family: str
    truth: dict[str, ExtractedRegion]
    drifted: bool


class _Clock:
    """Counts simulated provider calls and sleeps ``latency`` seconds per call."""

    def __init__(self, latency: float) -> None:
        self.latency, self.calls = latency, 0

    def call(self) -> None:
        self.calls += 1
        if self.latency:
            time.sleep(self.latency)


def _family(idx: int) -> tuple[FamilyCriteria, tuple[RecipeRegion, ...]]:
    criteria = FamilyCriteria(
        f"fam{idx}",
        (("vendor", f"vendor{idx}"),),
        (RecipeAnchor(f"VENDOR {idx} INVOICE", (0.1, 0.05, 0.5, 0.1)),),
        page_count=1,
    )
    regions = (
        RecipeRegion("header", 1, (0.1, 0.05, 0.9, 0.2), "text-ocr"),
        RecipeRegion("lines", 1, (0.1, 0.3, 0.9, 0.7), "table-vlm", kind="table"),
        RecipeRegion("total", 1, (0.6, 0.8, 0.9, 0.9), "text-ocr"),
    )
    return criteria, regions


def _make_doc(rng: random.Random, family: int, n: int, drifted: bool) -> Doc:
    rows = 3
    truth = {
        "header": ExtractedRegion("header", f"INV-{family}-{n}", 1),
        "lines": ExtractedRegion(
            "lines",
            ";".join(str(rng.randrange(100)) for _ in range(rows + drifted)),
            rows + (1 if drifted else 0),
            (rows + (1 if drifted else 0), 2),
        ),
        "total": ExtractedRegion("total", str(rng.randrange(1000)), 1),
    }
    return Doc(f"fam{family}", truth, drifted)


def _fp(doc: Doc, family: int) -> DocumentFingerprint:
    return DocumentFingerprint(
        {"vendor": f"vendor{family}"},
        ((f"VENDOR {family} INVOICE", (0.1, 0.05, 0.5, 0.1)),),
        1,
    )


def _accuracy(results: list[tuple[Doc, tuple[ExtractedRegion, ...]]]) -> float:
    right = sum(
        {r.name: r.text for r in got} == {n: r.text for n, r in doc.truth.items()}
        for doc, got in results
    )
    return right / len(results)


def _run(path_factory, docs, families, clock: _Clock) -> dict[str, Any]:
    results, latencies = [], []
    path = path_factory()
    for doc, family in docs:
        start = time.perf_counter()
        out = path.process(TENANT, doc, _fp(doc, family))
        latencies.append((time.perf_counter() - start) * 1000)
        results.append((doc, out.regions))
    total = sum(latencies) / 1000
    ordered = sorted(latencies)
    return {
        "accuracy": _accuracy(results),
        "provider_calls": clock.calls,
        "p50_ms": statistics.median(latencies),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "docs_per_second": len(docs) / total if total else float("inf"),
        "telemetry": path.telemetry.snapshot(TENANT),
        "rates": path.telemetry.rates(TENANT),
    }


def run_benchmark(
    *,
    families: int = 4,
    docs_per_family: int = 40,
    drift_every: int = 10,
    latency: float = 0.0005,
    reconcile_calls_per_region: int = 1,
    seed: int = 11,
) -> dict[str, Any]:
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture
    docs: list[tuple[Doc, int]] = []
    for n in range(docs_per_family):
        for fam in range(families):
            docs.append((_make_doc(rng, fam, n, drifted=n > 0 and n % drift_every == 0), fam))
    criteria = {f: _family(f) for f in range(families)}
    # ASSUMPTION (reported): layout detection + per-region extraction + per-region reconciliation.
    full_cost = 1 + 3 * (1 + reconcile_calls_per_region)

    def build(use_recipes: bool, clock: _Clock):
        def full(doc: Doc):
            for _ in range(full_cost):
                clock.call()
            return list(doc.truth.values())

        class Extractor:
            def extract_region(self, doc: Doc, region: RecipeRegion) -> ExtractedRegion:
                clock.call()
                return doc.truth[region.name]

        store = InMemoryRecipeStore()
        classifier = FamilyClassifier([c for c, _ in criteria.values()] if use_recipes else [])
        path = RecipeFastPath(
            classifier=classifier,
            store=store,
            region_extractor=Extractor(),
            verifier=CountCoverageVerifier(),
            full_extractor=full,
            engine_validity=VALIDITY,
            full_cost_calls=full_cost,
            quarantine_after=3,
        )
        if use_recipes:
            _learn(store, criteria)
        return path

    full_clock, recipe_clock = _Clock(latency), _Clock(latency)
    baseline = _run(lambda: build(False, full_clock), docs, families, full_clock)
    recipe = _run(lambda: build(True, recipe_clock), docs, families, recipe_clock)
    verified_ok = recipe["accuracy"] >= baseline["accuracy"]
    return {
        "benchmark": "extraction-recipes-family",
        "documents": len(docs),
        "families": families,
        "drifted_documents": sum(d.drifted for d, _ in docs),
        "simulated_latency_per_call_s": latency,
        "assumed_full_path_calls_per_document": full_cost,
        "no_recipe": baseline,
        "recipe": recipe,
        "provider_call_reduction": 1 - recipe["provider_calls"] / baseline["provider_calls"],
        "throughput_speedup": recipe["docs_per_second"] / baseline["docs_per_second"],
        "no_accuracy_regression": verified_ok,
        "notes": [
            "Synthetic families; provider latency is simulated, call counts and accuracy exact.",
            "Drifted documents (extra table row) must be caught by verification and fall back.",
        ],
    }


def _learn(
    store: InMemoryRecipeStore, criteria: dict[int, tuple[FamilyCriteria, tuple[RecipeRegion, ...]]]
) -> None:
    """Learn one verified recipe per family through the guarded learner."""
    page = PageExtractionOutcome(
        1, PageExtractionStatus.EXTRACTED, text="verified", provider="synthetic"
    )
    verifier = EvidenceAllowlistVerifier([extraction_evidence_hash(page)])
    learner = RecipeLearner(
        store, verifier, sample_confirmer=lambda r: True, approval_required=False
    )
    for crit, regions in criteria.values():
        draft = ExtractionRecipe(
            tenant=TENANT,
            criteria=crit,
            regions=regions,
            validity=VALIDITY,
            provenance=RecipeProvenance("bench", "1970-01-01T00:00:00Z", "a" * 64, "v", "1"),
            expected_counts=(("lines", 3, 3), ("header", 1, 1), ("total", 1, 1)),
        )
        learner.learn(draft, [replace(page)])
