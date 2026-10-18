"""Pins the two properties that make ``examples/host_integration.py`` a safe
copy-paste starting point for any domain host application.

(a) The template's settings class satisfies the ``RagSettings`` protocol, so
any host settings class works with ragnor's seams.
(b) The template's pipeline runs end to end against the in-memory store and
the deterministic fake embedder with no external services — no network, no
database, no API keys — so a copied template is immediately runnable offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import host_integration  # noqa: E402

from ragnor.config import RagSettings  # noqa: E402
from ragnor.embeddings import embedding_registry, get_embedding_function  # noqa: E402
from ragnor.stores import vector_store_registry  # noqa: E402


def test_example_settings_satisfy_the_rag_settings_protocol() -> None:
    """The template's plain settings class satisfies the ``RagSettings`` protocol."""
    assert isinstance(host_integration.DomainSettings(), RagSettings)


def test_example_registers_its_domain_providers() -> None:
    """Importing the module registers the domain providers and the embedder dimension."""
    assert embedding_registry.get("domain-provider") is host_integration.DomainEmbeddingProvider
    assert vector_store_registry.get("domain-backend") is host_integration.DomainVectorBackend
    embedder = get_embedding_function(host_integration.DomainSettings())
    assert isinstance(embedder, host_integration.FakeDomainEmbedder)


async def test_example_pipeline_runs_offline_with_the_memory_store() -> None:
    """The template's pipeline ingests, retrieves, and evaluates fully offline."""
    # The demo backend shares one in-memory index per table name, so a unique
    # table name isolates this test from the demo's "domain_chunks" index.
    app = host_integration.build_host(table_name="test_host_integration")
    document = Path(host_integration.__file__).with_name("sample_document.md")
    source = host_integration.FileDocumentSource(str(document))

    count = await app.ingest(source)
    assert count > 0

    results = await app.retrieve("What is the refund policy?", n_results=5)
    assert results
    assert any(result["id"] for result in results)

    metrics = await app.evaluate()
    assert metrics.total_queries == len(host_integration.TEST_SET)
    assert metrics.errors == 0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
