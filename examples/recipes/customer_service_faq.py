"""Recipe 2: Customer service knowledge with product/locale filters and FAQ deletion.

Demonstrates:
1. Setting up an offline KnowledgeBridge for customer service FAQs.
2. Ingesting FAQs with multi-dimensional metadata (product, locale, category).
3. Querying knowledge using narrowing filters (product='cloud_storage', locale='en-US').
4. Proving that results strictly respect filters and do not leak across products or locales.
5. Deleting / revoking an outdated FAQ using bridge.revoke_document().
6. Verifying that the revoked FAQ is immediately invisible to search.
7. Production profile illustrating scalable vector stores and filter index configurations.
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
    """Return production configuration specifications for customer service deployment."""
    return {
        "profile": "production-customer-service-faqs",
        "vector_store_provider": "qdrant",
        "collection_name": "support_faqs",
        "payload_schema": {
            "product": "keyword",
            "locale": "keyword",
            "tier": "keyword",
        },
        "embedding_provider": "litellm",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "filter_pushdown_enabled": True,
        "max_evidence_items": 5,
        "token_budget": 2048,
    }


async def run_recipe() -> None:
    """Run the customer service FAQ recipe."""
    print("--- 1. Initializing Customer Service Bridge ---")
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

    # Ingest multiple FAQs across products and locales
    faqs = [
        {
            "id": "faq_storage_en",
            "text": (
                "How do I increase cloud storage bucket quota? Submit a quota increase "
                "request via the Cloud Console under Storage Quotas."
            ),
            "embedding": embedder.embed("increase cloud storage bucket quota console request"),
            "metadata": {
                "source_id": "kb_portal",
                "document_id": "faq_storage_quota",
                "revision": "v1",
                "product": "cloud_storage",
                "locale": "en-US",
                "title": "Cloud Storage Quota Increase",
            },
        },
        {
            "id": "faq_storage_de",
            "text": (
                "Wie kann ich mein Speicherplatzkontingent erhöhen? Senden Sie eine Anfrage "
                "über die Verwaltungskonsole."
            ),
            "embedding": embedder.embed("speicherplatzkontingent erhoehen anfrage konsole"),
            "metadata": {
                "source_id": "kb_portal",
                "document_id": "faq_storage_quota_de",
                "revision": "v1",
                "product": "cloud_storage",
                "locale": "de-DE",
                "title": "Speicherplatzkontingent",
            },
        },
        {
            "id": "faq_compute_en",
            "text": (
                "How do I resize a virtual machine instance? Stop the VM instance, "
                "select edit configuration, choose new instance type, and restart."
            ),
            "embedding": embedder.embed(
                "resize virtual machine instance edit configuration new type"
            ),
            "metadata": {
                "source_id": "kb_portal",
                "document_id": "faq_vm_resize",
                "revision": "v1",
                "product": "cloud_compute",
                "locale": "en-US",
                "title": "Virtual Machine Resizing",
            },
        },
    ]
    await vector_store.upsert(faqs)

    print("--- 2. Querying Knowledge with Product & Locale Filters ---")
    # Search for quota increase specifically in English for cloud_storage
    bundle_en: EvidenceBundle = await bridge.search_knowledge(
        query="increase quota",
        filters={"product": "cloud_storage", "locale": "en-US"},
        final_k=3,
    )
    print(f"Bundle status (en-US): {bundle_en.status}")
    print(f"Items found: {len(bundle_en.items)}")
    for item in bundle_en.items:
        print(f"  [{item.citation_label}] {item.title}: {item.text}")

    assert len(bundle_en.items) == 1
    assert "Cloud Storage Quota Increase" in bundle_en.items[0].title
    assert "Console under Storage Quotas" in bundle_en.items[0].text

    # Search for quota in German
    bundle_de = await bridge.search_knowledge(
        query="kontingent erhoehen",
        filters={"product": "cloud_storage", "locale": "de-DE"},
        final_k=3,
    )
    assert len(bundle_de.items) == 1
    assert "Speicherplatzkontingent" in bundle_de.items[0].title

    # Query with compute filter should NOT return storage FAQ
    bundle_compute = await bridge.search_knowledge(
        query="how to increase quota",
        filters={"product": "cloud_compute", "locale": "en-US"},
        final_k=3,
    )
    # The compute VM FAQ doesn't match quota query and storage FAQ was filtered out
    for item in bundle_compute.items:
        assert item.title != "Cloud Storage Quota Increase"
    print("Filter enforcement verified: zero leakage across products or locales!")

    print("\n--- 3. Authoritative FAQ Deletion / Revocation ---")
    revoke_res = await bridge.revoke_document(
        collection="default",
        document_id="faq_storage_quota",
        source_id="kb_portal",
    )
    print(f"Revoke response: {revoke_res}")
    assert revoke_res["status"] == "revoked"

    # Subsequent search for cloud_storage quota should yield zero hits
    bundle_after_delete = await bridge.search_knowledge(
        query="increase quota",
        filters={"product": "cloud_storage", "locale": "en-US"},
        final_k=3,
    )
    print(f"Bundle items after revocation: {len(bundle_after_delete.items)}")
    assert len(bundle_after_delete.items) == 0
    print("FAQ deletion verified: revoked content immediately excluded from search!")


if __name__ == "__main__":
    asyncio.run(run_recipe())
