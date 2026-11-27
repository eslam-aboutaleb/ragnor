"""Run visual retrieval against a text-only baseline on unparseable pages."""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Sequence
from typing import Any

from benchmarks.visual_retrieval.fake_model import (
    BINS,
    GRID,
    FakeVisualProvider,
    make_page,
    png,
)
from ragnor.retrieval.visual import (
    InMemoryVisualIndex,
    ScoreCalibrator,
    VisualIndexer,
    VisualRetriever,
    fuse_results,
)

VECTOR_BYTES = 4  # float32 estimate for storage cost


class _AllowAll:
    def is_authorized(self, tenant: str, principal: str, page: Any) -> bool:
        return True


def _corpus(pages: int, seed: int) -> list[dict[tuple[int, int], int]]:
    rng = random.Random(seed)  # noqa: S311 - deterministic fixture, not security
    seen: set[frozenset[tuple[int, int]]] = set()
    out = []
    while len(out) < pages:
        cells = {}
        for cell in rng.sample(range(GRID * GRID), rng.randint(3, 6)):
            cells[(cell // GRID, cell % GRID)] = rng.randrange(BINS)
        key = frozenset(cells.items())
        if key not in seen:
            seen.add(key)
            out.append(cells)
    return out


def _token(cell: tuple[int, int], level: int) -> str:
    return f"c{cell[0] * GRID + cell[1]}b{level}"


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_benchmark(
    *,
    pages: int = 40,
    seed: int = 7,
    k: int = 5,
    provider: Any | None = None,
    user_queries: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Return Recall@K, MRR, latency, memory, patch cost and false positives.

    ``user_queries`` are extra ``(query, expected_page_id)`` pairs, evaluated
    alongside the generated ones (user-defined visual page queries).
    """
    provider = provider or FakeVisualProvider()
    index = InMemoryVisualIndex(provider.model)
    drawings = _corpus(pages, seed)
    batch = []
    for i, cells in enumerate(drawings):
        data = png(cells)
        batch.append((make_page("bench", f"doc{i}", 1, data), data))
    started = time.perf_counter()
    report = VisualIndexer(provider, index).index("bench", batch)
    index_seconds = time.perf_counter() - started
    retriever = VisualRetriever(provider, index, _AllowAll())

    rng = random.Random(seed + 1)  # noqa: S311
    queries = []
    for i, cells in enumerate(drawings):
        picks = rng.sample(sorted(cells.items()), 2)
        queries.append((" ".join(_token(c, lvl) for c, lvl in picks), f"doc{i}-p1"))
    queries.extend(user_queries)

    def evaluate(search):
        recalls, rrs, latencies = [], [], []
        for query, expected in queries:
            t0 = time.perf_counter()
            ranked = search(query)
            latencies.append((time.perf_counter() - t0) * 1000)
            recalls.append(1.0 if expected in ranked[:k] else 0.0)
            rrs.append(1 / (ranked.index(expected) + 1) if expected in ranked else 0.0)
        return {
            f"recall_at_{k}": statistics.fmean(recalls),
            "mrr": statistics.fmean(rrs),
            "latency_ms_p50": _percentile(latencies, 0.5),
            "latency_ms_p95": _percentile(latencies, 0.95),
        }

    # Text baseline: these pages have no extractable text, so a lexical retriever has nothing.
    text_only = evaluate(lambda q: [])
    visual = evaluate(lambda q: [e.page_id for e in retriever.search("bench", "u", q, limit=k)])

    # Fusion with an (empty) text channel must not lose visual wins nor duplicate pages.
    cal = ScoreCalibrator(
        [
            h.score
            for h in index.search(
                "bench", provider.embed_query(queries[0][0]), provider.model, pages
            )
        ]
    )

    def fused_search(q):
        ev = retriever.search("bench", "u", q, limit=k)
        return [
            r.evidence.page_id
            for r in fuse_results(ev, [], visual_calibrator=cal, text_calibrator=cal, limit=k)
            if r.evidence
        ]

    fused = evaluate(fused_search)

    # False positives: queries for patches that exist on no page (cell/level pairs never drawn).
    drawn = {(c, lvl) for cells in drawings for c, lvl in cells.items()}
    absent = [
        (c, lvl)
        for c in ((r, col) for r in range(GRID) for col in range(GRID))
        for lvl in range(BINS)
        if (c, lvl) not in drawn
    ]
    negatives = absent[:20]
    false_pos = 0
    for cell, lvl in negatives:
        top = index.search("bench", provider.embed_query(_token(cell, lvl)), provider.model, 1)
        false_pos += bool(
            top and top[0].score >= 0.99
        )  # near-exact patch match to a page lacking it
    patches = report.patches
    return {
        "benchmark": "visual-page-retrieval",
        "provider": f"{provider.model.model_id}@{provider.model.revision}",
        "provider_is_fake": isinstance(provider, FakeVisualProvider),
        "pages": pages,
        "queries": len(queries),
        "k": k,
        "text_only": text_only,
        "visual": visual,
        "fused": fused,
        "index_seconds": index_seconds,
        "patches_total": patches,
        "patches_per_page": patches / pages,
        "index_memory_bytes_estimate": patches * provider.model.dim * VECTOR_BYTES,
        "storage_cost_bytes_per_page": patches / pages * provider.model.dim * VECTOR_BYTES,
        "negative_queries": len(negatives),
        "false_positive_rate": false_pos / len(negatives) if negatives else 0.0,
        "notes": [
            "Pages are generated text-free rasters; text-only recall is zero by construction.",
            "Fake embedder validates plumbing, not real model quality.",
        ],
    }
