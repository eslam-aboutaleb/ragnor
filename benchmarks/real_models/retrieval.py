"""Recall@K and MRR with real embeddings, brute force and through a Qdrant service."""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from benchmarks.pdf_extraction.manifest import Manifest
from benchmarks.pdf_extraction.probes import IndexedChunk, Probe, build_index
from benchmarks.pdf_extraction.providers import ExtractionProvider

TOP_K = 5
TIMING_REPEATS = 50
Embed = Callable[[list[str]], list[list[float]]]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _is_hit(chunk: IndexedChunk, probe: Probe) -> bool:
    return (chunk.document_id, chunk.page, chunk.region_id) == (
        probe.document_id,
        probe.page,
        probe.region_id,
    )


def _summarise(order_per_probe: list[list[IndexedChunk]], probes: list[Probe], k: int) -> dict:
    ranks = []
    for top, probe in zip(order_per_probe, probes, strict=True):
        ranks.append(next((n for n, c in enumerate(top[:k], 1) if _is_hit(c, probe)), None))
    total = max(len(probes), 1)
    return {
        "k": k,
        "probes": len(probes),
        "recall_at_k": round(sum(1 for r in ranks if r) / total, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / total, 4),
    }


def embed_corpus(
    manifest: Manifest, root: Path, provider: ExtractionProvider, embed: Embed
) -> tuple[list[IndexedChunk], list[list[float]]]:
    chunks = build_index(manifest, root, provider)
    vectors = [_unit(v) for v in embed([c.text for c in chunks])]
    return chunks, vectors


def brute_force(chunks, vectors, probes: list[Probe], embed: Embed, k: int = 5) -> dict:
    queries = [_unit(v) for v in embed([p.query for p in probes])]
    tops = []
    for q in queries:
        scored = sorted(
            range(len(chunks)),
            key=lambda i: (
                -sum(a * b for a, b in zip(q, vectors[i], strict=True)),
                chunks[i].chunk_id,
            ),
        )
        tops.append([chunks[i] for i in scored[:k]])
    return {**_summarise(tops, probes, k), "indexed_chunks": len(chunks)}


def qdrant_search(client, chunks, vectors, probes: list[Probe], embed: Embed) -> dict:
    """Index in a throwaway Qdrant collection, search, and time repeated queries."""
    from qdrant_client import models  # noqa: PLC0415

    k, repeats = TOP_K, TIMING_REPEATS

    name = f"ragnor-bench-{uuid.uuid4().hex[:12]}"
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    try:
        started = time.perf_counter()
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(id=i, vector=vec, payload={"chunk": i})
                for i, vec in enumerate(vectors)
            ],
            wait=True,
        )
        upsert_ms = (time.perf_counter() - started) * 1000
        queries = [_unit(v) for v in embed([p.query for p in probes])]
        tops, latencies = [], []
        for q in queries:
            hits = client.query_points(collection_name=name, query=q, limit=k).points
            tops.append([chunks[h.payload["chunk"]] for h in hits])
        for _ in range(repeats):
            for q in queries:
                began = time.perf_counter()
                client.query_points(collection_name=name, query=q, limit=k)
                latencies.append((time.perf_counter() - began) * 1000)
        latencies.sort()
        return {
            **_summarise(tops, probes, k),
            "indexed_chunks": len(chunks),
            "upsert_ms": round(upsert_ms, 2),
            "query_ms_p50": round(statistics.median(latencies), 3),
            "query_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3),
            "queries_timed": len(latencies),
        }
    finally:
        client.delete_collection(collection_name=name)
