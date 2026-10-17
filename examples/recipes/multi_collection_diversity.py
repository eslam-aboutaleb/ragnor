"""Recipe 3: Search across multiple collections with source diversity.

Demonstrates:
1. Registering multiple collections ('docs' and 'incidents') on a single KnowledgeBridge.
2. Indexing documents with distinct source identifiers.
3. Querying across collections and merging candidate hits.
4. Enabling source diversity balancing (balance_sources=True, max_per_source=1)
   to ensure the assembled EvidenceBundle includes representation from each source
   rather than allowing one collection/source to monopolize the context budget.
5. Production profile illustrating multi-tenant collection routing and diversity parameters.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

from ragit.evidence.assembler import EvidenceAssembler
from ragit.evidence.source_lookup import MemorySnapshotStore, SourceLookup
from ragit.evidence.tokenizer import ApproximateTokenizer
from ragit.evidence.types import ContextBudget, EvidenceBundle, SearchHit
from ragit.stores.memory import InMemoryVectorStore
from ragit.tools.bridge import CollectionConfig, KnowledgeBridge


class HashingEmbedder:
    """Deterministic offline bag-of-words embedder."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    async def __call__(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, list):
            return [self.embed(t) for t in texts]
        return self.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        return self.embed(query)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def production_profile() -> dict[str, Any]:
    """Return production configuration specifications for multi-collection search."""
    return {
        "profile": "production-multi-collection-federation",
        "collections": [
            {
                "name": "product_docs",
                "backend": "pgvector",
                "table": "product_documentation_chunks",
                "source_id": "docs_repo",
            },
            {
                "name": "incident_reports",
                "backend": "pgvector",
                "table": "postmortem_chunks",
                "source_id": "incident_tracker",
            },
        ],
        "source_diversity": {
            "balance_sources": True,
            "max_per_source": 2,
            "round_robin_selection": True,
        },
        "budget": {
            "total_tokens": 4096,
            "reserved_instructions": 512,
            "reserved_answer": 1024,
        },
    }


async def run_recipe() -> None:
    """Run the multi-collection diversity recipe."""
    print("--- 1. Initializing Multi-Collection Setup ---")
    embedder = HashingEmbedder(dim=16)

    # 1. Setup Collection 1: Official Product Documentation
    docs_store = InMemoryVectorStore(
        table_name="docs_chunks", embedding_dim=16, embedding_fn=embedder
    )
    docs_chunks = [
        {
            "id": "doc_conn_pool",
            "text": (
                "Database Connection Pooling: Set pool_size to 20 to prevent exhausting "
                "database connections under load."
            ),
            "embedding": embedder.embed(
                "database connection pooling pool_size 20 exhausting database connections load"
            ),
            "metadata": {
                "source_id": "official_docs",
                "document_id": "db_pooling_guide",
                "title": "Database Connection Guide",
                "section": "Connection Pools",
            },
        },
        {
            "id": "doc_conn_timeout",
            "text": (
                "Database Timeouts: Set connection timeout to 5 seconds to fail fast "
                "during network partitions."
            ),
            "embedding": embedder.embed(
                "database connection timeout 5 seconds fail fast network partitions"
            ),
            "metadata": {
                "source_id": "official_docs",
                "document_id": "db_timeout_guide",
                "title": "Timeout Guidelines",
                "section": "Network Timeouts",
            },
        },
    ]
    await docs_store.upsert(docs_chunks)

    # 2. Setup Collection 2: Production Incident Postmortems
    incidents_store = InMemoryVectorStore(
        table_name="incident_chunks", embedding_dim=16, embedding_fn=embedder
    )
    incident_chunks = [
        {
            "id": "incident_2026_04",
            "text": (
                "Postmortem INC-402: Database outage occurred due to connection pool "
                "exhaustion when pool_size was set to 100 without database scaling."
            ),
            "embedding": embedder.embed(
                "postmortem inc 402 database outage connection pool exhaustion pool_size 100"
            ),
            "metadata": {
                "source_id": "incident_postmortems",
                "document_id": "inc_402_report",
                "title": "INC-402 Postmortem",
                "section": "Root Cause Analysis",
            },
        },
    ]
    await incidents_store.upsert(incident_chunks)

    # 3. Configure KnowledgeBridge with both collections
    snapshot_store = MemorySnapshotStore()
    source_lookup = SourceLookup(snapshot_store)

    collections = {
        "docs": CollectionConfig(
            name="docs",
            retriever=docs_store,
            source_lookup=source_lookup,
        ),
        "incidents": CollectionConfig(
            name="incidents",
            retriever=incidents_store,
            source_lookup=source_lookup,
        ),
    }

    # Assembler with source diversity enabled: max 1 per source
    diverse_assembler = EvidenceAssembler(
        tokenizer=ApproximateTokenizer(),
        balance_sources=True,
        max_per_source=1,
    )

    bridge = KnowledgeBridge(
        collections=collections,
        assembler=diverse_assembler,
        source_lookup=source_lookup,
    )

    print("--- 2. Querying Across Collections with Source Diversity ---")
    query = "database connection pool exhaustion"

    # Search each collection using the bridge
    bundle_docs: EvidenceBundle = await bridge.search_knowledge(
        query=query,
        collection="docs",
        candidate_k=5,
    )
    bundle_incidents: EvidenceBundle = await bridge.search_knowledge(
        query=query,
        collection="incidents",
        candidate_k=5,
    )

    # Combine hits from both collections and assemble with diversity balancing
    all_hits: list[SearchHit] = []
    for item in bundle_docs.items:
        all_hits.append(
            SearchHit(
                id=item.document_id,
                document=item.text,
                source_id=item.source_id,
                document_id=item.document_id,
                revision=item.revision,
                title=item.title,
                section=item.section,
                score=item.score or 0.8,
            )
        )
    for item in bundle_incidents.items:
        all_hits.append(
            SearchHit(
                id=item.document_id,
                document=item.text,
                source_id=item.source_id,
                document_id=item.document_id,
                revision=item.revision,
                title=item.title,
                section=item.section,
                score=item.score or 0.9,
            )
        )

    # Assemble combined hits with budget & diversity
    budget = ContextBudget(total_budget=1000, max_evidence_items=2)
    diverse_bundle = diverse_assembler.assemble(all_hits, budget=budget)

    print(f"Diverse Bundle status: {diverse_bundle.status}")
    print(f"Items in bundle: {len(diverse_bundle.items)}")
    sources_in_bundle = [item.source_id for item in diverse_bundle.items]
    print(f"Sources represented: {sources_in_bundle}")

    for item in diverse_bundle.items:
        txt = item.text[:70]
        print(f"  [{item.citation_label}] Source: {item.source_id} | {item.title}: {txt}...")

    # Verify both sources are represented despite score differences
    assert "official_docs" in sources_in_bundle
    assert "incident_postmortems" in sources_in_bundle
    print("Multi-collection source diversity verified: hits balanced across distinct sources!")


if __name__ == "__main__":
    asyncio.run(run_recipe())
