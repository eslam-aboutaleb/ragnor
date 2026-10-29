"""Deterministic lexical RAG and immutable, authorized citation qualification.

Uses the real SDK chunker, retrieval strategy, evaluator, evidence assembler and
source lookup. Reference-derived probes are labelled as contract probes; curated
human QA can be supplied separately and is never claimed from generated anchors.
"""

from __future__ import annotations

import asyncio
import hashlib
import unicodedata
from dataclasses import asdict

from benchmarks.quality_metrics import normalize, tokens
from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.evaluation.retrieval import LabeledQuery, RagEvaluationHarness
from ragnor.evidence.assembler import EvidenceAssembler
from ragnor.evidence.references import parse_source_reference
from ragnor.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragnor.evidence.types import ContextBudget, EvidenceStatus
from ragnor.retrieval.capabilities import StoreCapabilities
from ragnor.retrieval.strategies import KeywordRetrieval
from ragnor.retrieval.types import RetrievalRequest


def search_key(text):
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", normalize(text)).casefold()
        if not unicodedata.combining(char)
    )


SCOPE = "benchmark-authorized"
SOURCE = "pdf-qualification"


class BM25Index:
    capabilities = StoreCapabilities(
        filter_operators=frozenset({"eq"}),
        features=frozenset({"isolation_scope", "native_lexical_search"}),
    )

    def __init__(self, rows):
        self.rows = rows
        self._cache = {}

    async def keyword_search(self, query, n_results=5, **kwargs):
        from rank_bm25 import BM25Plus  # noqa: PLC0415 -- benchmark-only lexical model

        constraints = {
            key: value
            for key, value in kwargs.items()
            if key not in {"text_field", "metadata_fields", "session"}
        }
        visible = [
            row
            for row in self.rows
            if all(row["metadata"].get(key) == value for key, value in constraints.items())
        ]
        if not visible:
            return []
        key = tuple(row["id"] for row in visible)
        if key not in self._cache:
            self._cache[key] = BM25Plus(
                [tokens(search_key(row["document"])) or [""] for row in visible]
            )
        model = self._cache[key]
        query_tokens = tokens(search_key(query))
        baseline = model.delta * sum(model.idf.get(token, 0) for token in query_tokens)
        scores = model.get_scores(query_tokens) - baseline
        ranked = sorted(
            zip(visible, scores, strict=True), key=lambda pair: (-pair[1], pair[0]["id"])
        )
        return [
            {**row, "keyword_score": float(score), "score": float(score)}
            for row, score in ranked[:n_results]
            if score > 1e-12
        ]


def _probes(manifest: dict, limit: int) -> tuple[list[dict], str]:
    if limit < 1:
        raise ValueError("Probe limit must be positive")
    if manifest.get("qa_probes"):
        queries = [search_key(probe["query"]) for probe in manifest["qa_probes"]]
        if len(queries) != len(set(queries)):
            raise ValueError("Curated QA queries must be unique")
        return manifest["qa_probes"][:limit], "curated_qa"
    # Balance probes across families; every anchor comes from reference annotations.
    families = sorted({doc["family"] for doc in manifest["documents"]})
    groups = {family: [] for family in families}
    for doc in manifest["documents"]:
        for page in doc["pages"]:
            regions = sorted(page["regions"], key=lambda r: (r["category"] != "Table", r["id"]))
            if not regions and page.get("expected_text"):
                regions = [
                    {"id": "transcription", "category": "text", "text": page["expected_text"]}
                ]
            for region in regions:
                words = normalize(region.get("text", "")).split()
                if len(words) >= 6:
                    anchor = " ".join(words[:8])
                    groups[doc["family"]].append(
                        {
                            "id": f"{doc['document_id']}:{page['page']}:{region['id']}",
                            "query": anchor,
                            "answer": anchor,
                            "document_id": doc["document_id"],
                            "page": page["page"],
                            "region_id": region["id"],
                        }
                    )
                    break
    result = []
    while len(result) < limit and any(groups.values()):
        for family in families:
            if groups[family] and len(result) < limit:
                result.append(groups[family].pop(0))
    grouped = {}
    for probe in result:
        key = search_key(probe["query"])
        if key not in grouped:
            grouped[key] = {**probe, "targets": []}
        grouped[key]["targets"].append({"document_id": probe["document_id"], "page": probe["page"]})
    return list(grouped.values()), "reference_derived_contract_probes"


def _index(manifest, predictions):
    rows, snapshots = [], MemorySnapshotStore()
    chunker = SlidingWindowChunker(chunk_size=120, overlap=30)
    documents = {doc["document_id"]: doc for doc in manifest["documents"]}
    for (document_id, page), predicted in sorted(predictions.items()):
        text = normalize(predicted.text)
        document = documents[document_id]
        doc_key = f"{document_id}/page/{page}"
        revision = document["source_hash"]
        snapshots.put(
            SnapshotRecord(
                SOURCE,
                doc_key,
                revision,
                text,
                SCOPE,
                {"page": page, "original_document_id": document_id},
            )
        )
        cursor = 0
        for ordinal, chunk in enumerate(chunker.chunk(text)):
            start = text.find(chunk.text, cursor)
            if start < 0:
                raise ValueError("Chunk does not resolve to immutable source span")
            end = start + len(chunk.text)
            cursor = start + 1
            identity = hashlib.sha256(f"{doc_key}:{revision}:{ordinal}".encode()).hexdigest()
            metadata = {
                "tenant_id": SCOPE,
                "source_id": SOURCE,
                "document_id": doc_key,
                "revision": revision,
                "page": page,
                "original_document_id": document_id,
                "start_offset": start,
                "end_offset": end,
            }
            rows.append({"id": identity, "document": chunk.text, "metadata": metadata})
    return rows, snapshots


def _security_checks(rows, snapshots):
    checks = []
    if not rows:
        return {"checks": 0, "passed": None}
    row = rows[0]
    metadata = row["metadata"]
    source, doc, revision = metadata["source_id"], metadata["document_id"], metadata["revision"]
    lookup = SourceLookup(snapshots)
    wrong = lookup.lookup_span(source, doc, revision, authorization_scope="another-tenant")
    checks.append(not wrong.text and wrong.status != EvidenceStatus.SUCCESS)
    missing = lookup.lookup_span(source, doc, "unavailable-revision", authorization_scope=SCOPE)
    checks.append(not missing.text and missing.status == EvidenceStatus.REVISION_UNAVAILABLE)
    # A new retained revision must not change the old citation's bytes.
    original = snapshots.get_scoped(source, doc, revision, SCOPE)
    local = MemorySnapshotStore()
    local.put(original)
    local.put(SnapshotRecord(source, doc, "new-revision", "new bytes", SCOPE))
    checks.append(
        SourceLookup(local).lookup_span(source, doc, revision, authorization_scope=SCOPE).text
        == original.content[:4000]
    )
    local.revoke(source, doc, SCOPE)
    revoked = SourceLookup(local).lookup_span(source, doc, revision, authorization_scope=SCOPE)
    checks.append(not revoked.text and revoked.status != EvidenceStatus.SUCCESS)
    return {"checks": len(checks), "passed": all(checks)}


async def evaluate_rag(manifest: dict, predictions: dict, *, limit=40, k=5) -> dict:
    rows, snapshots = _index(manifest, predictions)
    probes, origin = _probes(manifest, limit)
    retrieval = KeywordRetrieval(BM25Index(rows))
    lookup = SourceLookup(snapshots)
    assembler = EvidenceAssembler(merge_overlapping_spans=False)
    labelled, expected = [], {}
    for probe in probes:
        matches = set()
        for target in probe.get("targets", [probe]):
            found = {
                row["id"]
                for row in rows
                if row["metadata"]["original_document_id"] == target["document_id"]
                and row["metadata"]["page"] == target["page"]
                and search_key(probe["answer"]) in search_key(row["document"])
            }
            matches.update(
                found
                or {
                    "missing:"
                    + probe["id"]
                    + ":"
                    + target["document_id"]
                    + ":"
                    + str(target["page"])
                }
            )
        expected[probe["query"]] = matches
        labelled.append(LabeledQuery(probe["query"], matches))
    citations, resolved, query_rows = 0, 0, []

    async def retrieve(query):
        nonlocal citations, resolved
        request = RetrievalRequest(query=query, final_k=k, isolation_scope=SCOPE, strict=True)
        found = await retrieval.retrieve(request)
        bundle = assembler.assemble(found, ContextBudget(total_budget=20000))
        for item in bundle.items:
            citations += 1
            source, doc, rev, start, end = parse_source_reference(item.durable_id)
            span = lookup.lookup_span(
                source,
                doc,
                rev,
                authorization_scope=SCOPE,
                start_offset=start,
                end_offset=end,
                max_span_length=20000,
            )
            resolved += span.status == EvidenceStatus.SUCCESS and span.text == item.text
        query_rows.append(
            {
                "probe_id": hashlib.sha256(query.encode()).hexdigest(),
                "expected_source_hit": bool(expected[query] & {row["id"] for row in found}),
            }
        )
        return found

    metrics = await RagEvaluationHarness(labelled).evaluate(retrieve)
    denied = await retrieval.retrieve(
        RetrievalRequest(
            query=probes[0]["query"] if probes else "none",
            final_k=k,
            isolation_scope="another-tenant",
            strict=True,
        )
    )
    security = _security_checks(rows, snapshots)
    security["retrieval_cross_tenant_denied"] = not denied
    if security["checks"]:
        security["passed"] = security["passed"] and not denied
    return {
        "mode": "SDK lexical BM25Plus; no embedding/answer model",
        "probe_origin": origin,
        "k": k,
        **asdict(metrics),
        "citation_count": citations,
        "citation_resolution_rate": resolved / citations if citations else None,
        "source_answer_hit_rate": sum(row["expected_source_hit"] for row in query_rows)
        / len(query_rows)
        if query_rows
        else None,
        "authorization_and_revision": security,
        "probes": query_rows,
    }


def run_rag(manifest, predictions, **kwargs):
    return asyncio.run(evaluate_rag(manifest, predictions, **kwargs))
