"""Recipe 1: Policy knowledge with page citations and revision updates.

Demonstrates:
1. Setting up an offline KnowledgeBridge with in-memory storage and deterministic fake embeddings.
2. Indexing policy documents with page numbers and section citations.
3. Querying knowledge and receiving an EvidenceBundle with page-level citations.
4. Updating document from rev-1 to rev-2.
5. Verifying that get_source for rev-1 returns the immutable rev-1 snapshot
   (never substituting rev-2).
6. Production profile illustrating how to configure pgvector and production models.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any

from ragit.evidence.assembler import EvidenceAssembler
from ragit.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragit.evidence.tokenizer import ApproximateTokenizer
from ragit.evidence.types import EvidenceBundle
from ragit.stores.memory import InMemoryVectorStore
from ragit.tools.bridge import KnowledgeBridge


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
    """Return production configuration specifications for policy knowledge deployment."""
    return {
        "profile": "production-policy-compliance",
        "vector_store_provider": "pgvector",
        "table_name": "compliance_policy_chunks",
        "embedding_provider": "litellm",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "tsquery_config": "english",
        "distance_threshold": 0.45,
        "token_budget": 4096,
        "max_evidence_items": 8,
        "immutable_snapshot_retention": "365_days",
    }


async def run_recipe() -> None:
    """Run the policy knowledge recipe."""
    print("--- 1. Initializing Offline Store & Bridge ---")
    embedder = HashingEmbedder(dim=16)
    vector_store = InMemoryVectorStore(embedding_dim=16, embedding_fn=embedder)
    snapshot_store = MemorySnapshotStore()
    source_lookup = SourceLookup(snapshot_store)
    assembler = EvidenceAssembler(tokenizer=ApproximateTokenizer())

    bridge = KnowledgeBridge(
        retriever=vector_store,
        source_lookup=source_lookup,
        assembler=assembler,
        allow_management=True,
    )

    # Ingest Policy v1
    policy_doc_v1 = (
        "Page 1: Travel and Expense Policy (rev-1)\n"
        "All business travel must be booked at least 14 days in advance.\n"
        "Meals are reimbursed up to $75 per diem with receipts.\n\n"
        "Page 2: Equipment and Hardware Policy (rev-1)\n"
        "Laptops are refreshed every 3 years. Remote monitors are eligible up to $300."
    )

    # Register immutable snapshot for rev-1
    snapshot_store.put(
        SnapshotRecord(
            source_id="hr_portal",
            document_id="travel_policy",
            revision="rev-1",
            content=policy_doc_v1,
            isolation_scope="default",
            metadata={"title": "Company Expense and Hardware Guidelines", "department": "HR"},
        )
    )

    # Ingest chunks into vector store
    chunk_1 = {
        "id": "chunk_rev1_p1",
        "text": (
            "All business travel must be booked at least 14 days in advance. "
            "Meals are reimbursed up to $75 per diem."
        ),
        "embedding": embedder.embed(
            "travel expense reimbursement booked 14 days in advance meals $75"
        ),
        "metadata": {
            "source_id": "hr_portal",
            "document_id": "travel_policy",
            "revision": "rev-1",
            "page": 1,
            "section": "Travel and Expense",
            "title": "Expense Guidelines",
            "start_offset": 0,
            "end_offset": 160,
        },
    }
    chunk_2 = {
        "id": "chunk_rev1_p2",
        "text": "Laptops are refreshed every 3 years. Remote monitors are eligible up to $300.",
        "embedding": embedder.embed("hardware equipment laptop monitor refreshed 3 years $300"),
        "metadata": {
            "source_id": "hr_portal",
            "document_id": "travel_policy",
            "revision": "rev-1",
            "page": 2,
            "section": "Equipment and Hardware",
            "title": "Expense Guidelines",
            "start_offset": 161,
            "end_offset": 280,
        },
    }
    await vector_store.upsert([chunk_1, chunk_2])

    print("--- 2. Querying Knowledge: Policy with Page Citations ---")
    bundle: EvidenceBundle = await bridge.search_knowledge(
        query="what is the meal reimbursement per diem",
        budget_limit=2000,
        final_k=3,
    )
    print(f"Bundle status: {bundle.status}")
    print(f"Evidence items found: {len(bundle.items)}")
    for item in bundle.items:
        loc = f"Page {item.page}, Section: {item.section}"
        print(f"Citation: {item.citation_label} [{loc}] (rev: {item.revision})")
        print(f"  Excerpt: {item.text}")

    assert len(bundle.items) > 0
    assert bundle.items[0].page == 1
    assert bundle.items[0].revision == "rev-1"

    print("\n--- 3. Updating Policy to rev-2 (New Revision) ---")
    policy_doc_v2 = (
        "Page 1: Travel and Expense Policy (rev-2)\n"
        "All business travel must be booked at least 21 days in advance.\n"
        "Meals are reimbursed up to $100 per diem with receipts.\n\n"
        "Page 2: Equipment and Hardware Policy (rev-2)\n"
        "Laptops are refreshed every 2 years. Remote monitors are eligible up to $400."
    )
    snapshot_store.put(
        SnapshotRecord(
            source_id="hr_portal",
            document_id="travel_policy",
            revision="rev-2",
            content=policy_doc_v2,
            isolation_scope="default",
            metadata={"title": "Company Expense Guidelines", "department": "HR"},
        )
    )

    # Delete old chunks and add rev-2 chunks
    await vector_store.delete(ids=["chunk_rev1_p1", "chunk_rev1_p2"])
    chunk_1_v2 = {
        "id": "chunk_rev2_p1",
        "text": (
            "All business travel must be booked at least 21 days in advance. "
            "Meals are reimbursed up to $100 per diem."
        ),
        "embedding": embedder.embed(
            "travel expense reimbursement booked 21 days in advance meals $100"
        ),
        "metadata": {
            "source_id": "hr_portal",
            "document_id": "travel_policy",
            "revision": "rev-2",
            "page": 1,
            "section": "Travel and Expense",
            "title": "Expense Guidelines",
            "start_offset": 0,
            "end_offset": 165,
        },
    }
    await vector_store.upsert([chunk_1_v2])

    print("--- 4. Verifying Citation Resolution to Stated Revision ---")
    # Verify lookup for old citation rev-1 returns rev-1 ($75), NOT rev-2 ($100)
    old_citation_span = await bridge.get_source(
        source_ref="hr_portal",
        document_id="travel_policy",
        revision="rev-1",
        start_offset=0,
        end_offset=200,
    )

    print(f"Old citation lookup status: {old_citation_span.status}")
    print(f"Resolved text: {old_citation_span.text[:80]}...")
    assert "$75 per diem" in old_citation_span.text
    assert old_citation_span.revision == "rev-1"

    # Search for latest policy retrieves rev-2
    new_bundle = await bridge.search_knowledge(query="what is the meal reimbursement per diem")
    assert "$100 per diem" in new_bundle.items[0].text
    assert new_bundle.items[0].revision == "rev-2"
    print(
        "Citation resolution verified: never silently substitutes newer content for old citations!"
    )


if __name__ == "__main__":
    asyncio.run(run_recipe())
