"""End-to-end RAG probes over extracted text.

Each annotated text region that the provider actually recovered (per the
benchmark's own squashed-containment check) becomes one indexed chunk with
``(document_id, page, region_id)`` citation metadata. Chunks are embedded
with a deterministic hashing embedder (no external model, no network, stable
across runs) and labeled queries are scored with Recall@K and MRR.

This design deliberately separates parser quality from retrieval-model
nondeterminism: extraction failures surface as *unresolvable citations*
(the region never enters the index), while ranking quality is measured on
what was extracted. Chunk texts are the verified-recovered region contents,
never the raw page dump, so citation attribution is exact rather than
heuristic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from benchmarks.pdf_extraction.manifest import Manifest
from benchmarks.pdf_extraction.metrics import normalize_text, region_recovered
from benchmarks.pdf_extraction.providers import ExtractionProvider, measure

EMBEDDING_DIM = 512


@dataclass
class Probe:
    id: str
    query: str
    document_id: str
    page: int
    region_id: str


@dataclass
class IndexedChunk:
    chunk_id: str
    document_id: str
    page: int
    region_id: str
    text: str
    vector: list[float]


def load_probes(path: Path | str) -> list[Probe]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Probe(**item) for item in records["probes"]]


def embed(text: str) -> list[float]:
    """Deterministic hashing-vector embedding with L2 normalization."""
    vector = [0.0] * EMBEDDING_DIM
    for token in normalize_text(text).lower().split(" "):
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def build_index(
    manifest: Manifest, root: Path | str, provider: ExtractionProvider
) -> list[IndexedChunk]:
    """Index one chunk per region the provider verifiably recovered."""
    root = Path(root)
    index: list[IndexedChunk] = []
    for document_id, pages in manifest.documents().items():
        data = (root / pages[0].file).read_bytes()
        results, error = measure(provider, data, document_id)
        if error is not None:
            continue
        by_page = {p.page: p for p in results}
        for annotation in pages:
            index.extend(_page_chunks(annotation, by_page.get(annotation.page)))
    return index


def _page_chunks(annotation, result) -> list[IndexedChunk]:
    if annotation.expect_error or result is None or not result.text.strip():
        return []
    chunks = []
    for region in annotation.text_regions():
        if not region_recovered(region, result.text):
            continue
        text = normalize_text(region.expected_text)
        chunks.append(
            IndexedChunk(
                chunk_id=f"{annotation.document_id}:p{annotation.page}:{region.region_id}",
                document_id=annotation.document_id,
                page=annotation.page,
                region_id=region.region_id,
                text=text,
                vector=embed(text),
            )
        )
    return chunks


def rank(index: list[IndexedChunk], query: str, k: int) -> list[IndexedChunk]:
    """Cosine top-K over the deterministic embedding index."""
    query_vector = embed(query)
    scored = [
        (sum(a * b for a, b in zip(query_vector, chunk.vector, strict=True)), chunk)
        for chunk in index
    ]
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [chunk for _, chunk in scored[:k]]


def evaluate_probes(
    manifest: Manifest,
    root: Path | str,
    provider: ExtractionProvider,
    probes: list[Probe],
    k: int = 5,
) -> dict:
    """Recall@K, MRR and citation resolution for the labeled probes.

    ``citation_resolution`` is the fraction of probes whose expected region
    was recovered and therefore resolvable in the index; Recall@K can never
    exceed it, which keeps extraction failures visible in retrieval numbers.
    """
    index = build_index(manifest, root, provider)
    indexed = {(c.document_id, c.page, c.region_id) for c in index}
    hits, reciprocal_ranks, resolvable = 0, 0.0, 0
    details = []
    for probe in probes:
        expected = (probe.document_id, probe.page, probe.region_id)
        if expected in indexed:
            resolvable += 1
        rank_of_hit = _first_hit_rank(rank(index, probe.query, k), probe)
        if rank_of_hit is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank_of_hit
        details.append(
            {
                "probe_id": probe.id,
                "expected": f"{probe.document_id} p{probe.page} {probe.region_id}",
                "resolvable": expected in indexed,
                "hit_at_rank": rank_of_hit,
            }
        )
    total = max(len(probes), 1)
    return {
        "k": k,
        "probes": len(probes),
        "recall_at_k": hits / total,
        "mrr": reciprocal_ranks / total,
        "citation_resolution": resolvable / total,
        "indexed_chunks": len(index),
        "details": details,
    }


def _first_hit_rank(top: list[IndexedChunk], probe: Probe) -> int | None:
    for position, chunk in enumerate(top, start=1):
        if (
            chunk.document_id == probe.document_id
            and chunk.page == probe.page
            and chunk.region_id == probe.region_id
        ):
            return position
    return None
