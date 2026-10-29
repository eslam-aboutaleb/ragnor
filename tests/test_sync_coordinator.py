"""Tests for SyncCoordinator, ManifestStore, SyncStateStore, and SyncRetriever.

Verifies:
1. Editing one document embeds only that document.
2. Identical scans perform zero embedding calls.
3. Metadata-only updates perform zero embedding calls.
4. Failed embedding leaves prior permitted content intact.
5. Authoritative deletion removes visibility immediately.
6. Explicit access revocation removes visibility immediately.
7. Partial listings and non-authoritative scans never delete unseen documents.
8. Stale workers cannot activate obsolete revisions.
9. Crashes around activation are recoverable without duplicating active chunks.
10. Collection-level activation enforces all-or-nothing visibility.
11. Opaque revision hints (ETags) avoid unnecessary content hashing and re-embedding.
"""

from __future__ import annotations

from typing import Any
import pytest

from ragnor.chunking.base import Chunker
from ragnor.embeddings.dimensions import register_dimension
from ragnor.ingestion.connector_core import LoadResult, SourceStatus
from ragnor.ingestion.sources import RawDocument
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.sync.coordinator import (
    ActivationMode,
    DefaultSyncCoordinator,
    LeaseAcquisitionError,
)
from ragnor.sync.manifest import InMemoryManifestStore, ManifestStatus
from ragnor.sync.retrieval import SyncRetriever
from ragnor.sync.state import InMemorySyncStateStore
from ragnor.types import Chunk

TEST_MODEL = "test-sync-model"
register_dimension(TEST_MODEL, 4)


class _FakeSettings:
    def __init__(self, embedding_model: str = TEST_MODEL) -> None:
        self.embedding_model = embedding_model


class _RecordingEmbedder:
    def __init__(self, dim: int = 4, fail_on: str | None = None) -> None:
        self._dim = dim
        self.fail_on = fail_on
        self.call_count = 0
        self.embedded_texts: list[str] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        for text in texts:
            if self.fail_on and self.fail_on in text:
                raise RuntimeError(f"Embedding failed on text matching {self.fail_on}")
            self.embedded_texts.append(text)
        return [[float(len(t)) + i for i in range(self._dim)] for t in texts]


class _SimpleChunker(Chunker):
    def __init__(self, chunk_size: int = 100, overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        parts = text.split("\n\n")
        return [
            Chunk(
                id=f"part_{i}",
                text=part.strip(),
                metadata={"part_index": i},
            )
            for i, part in enumerate(parts)
            if part.strip()
        ]


@pytest.fixture
def manifest_store() -> InMemoryManifestStore:
    return InMemoryManifestStore()


@pytest.fixture
def state_store() -> InMemorySyncStateStore:
    return InMemorySyncStateStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(embedding_dim=4)


@pytest.fixture
def settings() -> _FakeSettings:
    return _FakeSettings()


@pytest.fixture
def chunker() -> _SimpleChunker:
    return _SimpleChunker()


@pytest.mark.asyncio
async def test_initial_sync_and_single_doc_edit_embeds_only_that_document(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    doc1 = RawDocument(source_name="doc1.txt", content="Hello world\n\nFirst doc body")
    doc2 = RawDocument(source_name="doc2.txt", content="Welcome\n\nSecond doc body")

    # Initial sync embeds both
    report1 = await coordinator.run_sync(scope="tenant1", source=[doc1, doc2])
    assert report1.success is True
    assert report1.added_count == 2
    assert report1.embedded_count == 4
    initial_embed_count = len(embedder.embedded_texts)
    assert initial_embed_count == 4

    # Second sync: edit doc1 only, doc2 unchanged
    doc1_v2 = RawDocument(
        source_name="doc1.txt",
        content="Hello world\n\nFirst doc body MODIFIED",
    )
    report2 = await coordinator.run_sync(scope="tenant1", source=[doc1_v2, doc2])
    assert report2.success is True
    assert report2.changed_count == 1
    assert report2.unchanged_count == 1
    # Only doc1 chunks (2 chunks) were embedded in the second pass
    assert len(embedder.embedded_texts) == initial_embed_count + 2
    assert embedder.embedded_texts[-1] == "First doc body MODIFIED"


@pytest.mark.asyncio
async def test_identical_scans_perform_zero_embedding_calls(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    docs = [
        RawDocument(source_name="doc1.txt", content="Content one"),
        RawDocument(source_name="doc2.txt", content="Content two"),
    ]

    await coordinator.run_sync(scope="tenant1", source=docs)
    embedder.call_count = 0

    # Identical scan
    report = await coordinator.run_sync(scope="tenant1", source=docs)
    assert report.success is True
    assert report.unchanged_count == 2
    assert report.embedded_count == 0
    assert embedder.call_count == 0


@pytest.mark.asyncio
async def test_metadata_only_update_performs_zero_embedding_calls(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    doc1 = RawDocument(
        source_name="doc1.txt",
        content="Static content",
        metadata={"category": "finance"},
    )
    await coordinator.run_sync(scope="tenant1", source=[doc1])
    embedder.call_count = 0

    # Update metadata only
    doc1_updated_meta = RawDocument(
        source_name="doc1.txt",
        content="Static content",
        metadata={"category": "legal", "confidential": True},
    )
    report = await coordinator.run_sync(scope="tenant1", source=[doc1_updated_meta])
    assert report.success is True
    assert report.metadata_only_count == 1
    assert report.embedded_count == 0
    assert embedder.call_count == 0

    rec = await manifest_store.get("tenant1", "doc1.txt")
    assert rec is not None
    assert rec.metadata["category"] == "legal"


@pytest.mark.asyncio
async def test_failed_embedding_leaves_prior_permitted_content_intact(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )
    retriever = SyncRetriever(vector_store, manifest_store)

    doc1 = RawDocument(source_name="doc1.txt", content="Good content A")
    doc2 = RawDocument(source_name="doc2.txt", content="Good content B")
    await coordinator.run_sync(scope="tenant1", source=[doc1, doc2])

    rec_before = await manifest_store.get("tenant1", "doc1.txt")
    assert rec_before is not None
    assert rec_before.active_generation == 1

    # Configure embedder to fail on BAD content for doc1
    embedder.fail_on = "EXPLODE"
    doc1_bad = RawDocument(source_name="doc1.txt", content="EXPLODE update")
    doc2_good = RawDocument(source_name="doc2.txt", content="Good content B")

    report = await coordinator.run_sync(scope="tenant1", source=[doc1_bad, doc2_good])
    assert len(report.errors) == 1
    assert report.errors[0]["doc_id"] == "doc1.txt"

    # Prior permitted content remains intact in manifest and queryable via retriever
    rec_after = await manifest_store.get("tenant1", "doc1.txt")
    assert rec_after is not None
    assert rec_after.status == ManifestStatus.ACTIVE
    assert rec_after.active_generation == 1
    assert rec_after.content_hash == rec_before.content_hash

    # Query retriever
    query_emb = [1.0, 2.0, 3.0, 4.0]
    hits = await retriever.hybrid_search("Good", query_emb, scope="tenant1")
    assert any(h["document"] == "Good content A" for h in hits)


@pytest.mark.asyncio
async def test_authoritative_deletion_removes_visibility_immediately(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )
    retriever = SyncRetriever(vector_store, manifest_store)

    doc1 = RawDocument(source_name="keep.txt", content="Keep me")
    doc2 = RawDocument(source_name="delete_me.txt", content="Secret to remove")
    await coordinator.run_sync(scope="tenant1", source=[doc1, doc2])

    query_emb = [1.0, 2.0, 3.0, 4.0]
    hits_before = await retriever.hybrid_search("Secret", query_emb, scope="tenant1")
    assert len(hits_before) > 0

    # Authoritative scan omitting doc2
    report = await coordinator.run_sync(
        scope="tenant1",
        source=[doc1],
        is_authoritative=True,
    )
    assert report.removed_count == 1

    rec2 = await manifest_store.get("tenant1", "delete_me.txt")
    assert rec2 is not None
    assert rec2.status == ManifestStatus.TOMBSTONED
    assert rec2.active_generation == 0

    # Retriever immediately filters out tombstoned document
    hits_after = await retriever.hybrid_search("Secret", query_emb, scope="tenant1")
    assert len(hits_after) == 0

    # doc1 is still queryable
    hits_doc1 = await retriever.hybrid_search("Keep", query_emb, scope="tenant1")
    assert len(hits_doc1) > 0


@pytest.mark.asyncio
async def test_revoke_access_removes_visibility_immediately(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )
    retriever = SyncRetriever(vector_store, manifest_store)

    doc1 = RawDocument(source_name="sensitive.txt", content="Classified info")
    await coordinator.run_sync(scope="tenant1", source=[doc1])

    # Revoke access immediately
    ok = await coordinator.revoke_access("tenant1", "sensitive.txt")
    assert ok is True

    rec = await manifest_store.get("tenant1", "sensitive.txt")
    assert rec is not None
    assert rec.status == ManifestStatus.TOMBSTONED

    # Immediately invisible through retriever
    hits = await retriever.hybrid_search("Classified", [1.0, 2.0, 3.0, 4.0], scope="tenant1")
    assert len(hits) == 0


@pytest.mark.asyncio
async def test_partial_listing_or_non_authoritative_scan_never_deletes(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    doc1 = RawDocument(source_name="doc1.txt", content="Doc 1")
    doc2 = RawDocument(source_name="doc2.txt", content="Doc 2")
    await coordinator.run_sync(scope="tenant1", source=[doc1, doc2])

    # Partial listing: only doc1 seen, but authoritative=False
    report = await coordinator.run_sync(
        scope="tenant1",
        source=[doc1],
        is_authoritative=False,
    )
    assert report.removed_count == 0

    # doc2 must NOT be deleted or tombstoned
    rec2 = await manifest_store.get("tenant1", "doc2.txt")
    assert rec2 is not None
    assert rec2.status == ManifestStatus.ACTIVE


@pytest.mark.asyncio
async def test_stale_worker_cannot_activate_obsolete_revisions(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    w1 = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
        worker_id="worker-1",
        lease_ttl_seconds=0.1,  # short lease for test
    )
    w2 = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
        worker_id="worker-2",
        lease_ttl_seconds=30.0,
    )

    # Worker 1 acquires lease
    t1 = await state_store.acquire_lease("tenant1", "worker-1", ttl_seconds=0.01)
    assert t1 is not None

    # Worker 2 acquires lease after W1 expires, superseding epoch
    import asyncio

    await asyncio.sleep(0.02)
    t2 = await state_store.acquire_lease("tenant1", "worker-2", ttl_seconds=30.0)
    assert t2 is not None
    assert t2.epoch > t1.epoch

    # Worker 1's token is now rejected
    assert await state_store.validate_token(t1) is False

    # Worker 1 cannot acquire lease while worker 2 holds it
    with pytest.raises(LeaseAcquisitionError):
        await w1.run_sync(
            "tenant1",
            source=[RawDocument(source_name="w1.txt", content="hi")],
        )

    # A worker must release its existing lease before starting another run.
    await state_store.release_lease(t2)
    # Worker 2 runs successfully
    report2 = await w2.run_sync(
        "tenant1",
        source=[RawDocument(source_name="w2.txt", content="hi")],
    )
    assert report2.success is True


@pytest.mark.asyncio
async def test_collection_level_activation_all_or_nothing(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
        activation_mode=ActivationMode.COLLECTION,
    )

    doc1 = RawDocument(source_name="doc1.txt", content="Doc 1")
    doc2 = RawDocument(source_name="doc2.txt", content="Doc 2")
    await coordinator.run_sync(scope="tenant1", source=[doc1, doc2])

    gen1 = await state_store.get_active_generation("tenant1")

    # Now run an update where doc2 fails
    embedder.fail_on = "FAIL"
    doc1_mod = RawDocument(source_name="doc1.txt", content="Doc 1 MOD")
    doc2_fail = RawDocument(source_name="doc2.txt", content="Doc 2 FAIL")

    with pytest.raises(RuntimeError):
        await coordinator.run_sync(scope="tenant1", source=[doc1_mod, doc2_fail])

    # Collection generation must NOT have advanced
    gen2 = await state_store.get_active_generation("tenant1")
    assert gen2 == gen1

    # Neither document is activated to new generation
    r1 = await manifest_store.get("tenant1", "doc1.txt")
    assert r1 is not None
    assert r1.active_generation == gen1


@pytest.mark.asyncio
async def test_opaque_etag_revision_hints(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    doc = RawDocument(
        source_name="doc.txt",
        content="Content with etag",
        metadata={"etag": "etag-v1"},
    )
    await coordinator.run_sync(scope="tenant1", source=[doc])
    embedder.call_count = 0

    # Scan with identical etag
    doc_same_etag = RawDocument(
        source_name="doc.txt",
        content="Content with etag",
        metadata={"etag": "etag-v1"},
    )
    plan1 = await coordinator.plan("tenant1", [doc_same_etag])
    assert len(plan1.unchanged) == 1
    assert len(plan1.changed) == 0

    # Scan with changed etag hint, but content hash is actually unchanged
    doc_new_etag_same_content = RawDocument(
        source_name="doc.txt",
        content="Content with etag",
        metadata={"etag": "etag-v2-refreshed"},
    )
    plan2 = await coordinator.plan("tenant1", [doc_new_etag_same_content])
    # ETag changed, but content hash matched and metadata only changed
    assert len(plan2.metadata_only) == 1
    assert len(plan2.changed) == 0


@pytest.mark.asyncio
async def test_outbox_integration_enqueues_idempotency_keys(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    from tests.job_doubles import FakeJobStore

    job_store = FakeJobStore()
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
        job_store=job_store,
    )

    doc = RawDocument(source_name="outbox_doc.txt", content="Some text")
    report = await coordinator.run_sync(scope="tenant1", source=[doc])
    assert report.success is True
    assert len(job_store.enqueued_payloads) == 1
    payload = job_store.enqueued_payloads[0]
    expected_key = "tenant1:outbox_doc.txt:1:e1:activate"
    assert payload["idempotency_key"] == expected_key


@pytest.mark.asyncio
async def test_crashes_around_activation_are_recoverable(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )
    retriever = SyncRetriever(vector_store, manifest_store)

    doc = RawDocument(source_name="doc_crash.txt", content="Initial state")
    await coordinator.run_sync(scope="tenant1", source=[doc])

    rec1 = await manifest_store.get("tenant1", "doc_crash.txt")
    assert rec1 is not None
    assert rec1.active_generation == 1

    # Simulate a crash right after staging by manually staging gen 2 without CAS activating it
    staged_chunk = {
        "id": "tenant1:doc_crash.txt:g2:0",
        "text": "Staged unactivated content",
        "metadata": {
            "scope": "tenant1",
            "doc_id": "doc_crash.txt",
            "generation": 2,
            "active": False,
        },
        "embedding": [2.0, 3.0, 4.0, 5.0],
    }
    await vector_store.upsert([staged_chunk])

    # Retriever still returns only generation 1
    hits = await retriever.hybrid_search(
        "state", [1.0, 2.0, 3.0, 4.0], scope="tenant1", threshold=100.0
    )
    assert any(h["document"] == "Initial state" for h in hits)
    assert not any("Staged unactivated" in h["document"] for h in hits)

    # Next run resumes and activates cleanly
    doc_v2 = RawDocument(source_name="doc_crash.txt", content="Staged unactivated content")
    report = await coordinator.run_sync(scope="tenant1", source=[doc_v2])
    assert report.success is True

    rec2 = await manifest_store.get("tenant1", "doc_crash.txt")
    assert rec2 is not None
    assert rec2.active_generation == 2
    assert rec2.status == ManifestStatus.ACTIVE


@pytest.mark.asyncio
async def test_paginated_connector_source_traversal(
    manifest_store: InMemoryManifestStore,
    state_store: InMemorySyncStateStore,
    vector_store: InMemoryVectorStore,
    settings: _FakeSettings,
    chunker: _SimpleChunker,
) -> None:
    class _MockPaginatedConnector:
        def __init__(self) -> None:
            self._pages = [
                LoadResult(
                    [RawDocument(source_name="page1_doc.txt", content="Page 1")],
                    has_more=True,
                    next_cursor="cur_2",
                    complete=False,
                ),
                LoadResult(
                    [RawDocument(source_name="page2_doc.txt", content="Page 2")],
                    has_more=False,
                    next_cursor=None,
                    complete=True,
                    status=SourceStatus.OK,
                ),
            ]

        async def load_page(self, cursor: Any = None, limit: int | None = None) -> LoadResult:
            if cursor is None:
                return self._pages[0]
            return self._pages[1]

    connector = _MockPaginatedConnector()
    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,  # type: ignore[arg-type]
    )

    report = await coordinator.run_sync(scope="tenant1", source=connector)
    assert report.success is True
    assert report.added_count == 2
    assert await manifest_store.get("tenant1", "page1_doc.txt") is not None
    assert await manifest_store.get("tenant1", "page2_doc.txt") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("activation_mode", [ActivationMode.DOCUMENT, ActivationMode.COLLECTION])
async def test_incomplete_pdf_preserves_manifest_and_active_vectors(  # noqa: PLR0913
    manifest_store, state_store, vector_store, settings, chunker, activation_mode
):
    from ragnor.ingestion.connector_core import extract_documents
    from tests.test_pdf_integrity import mixed_pdf

    embedder = _RecordingEmbedder()
    coordinator = DefaultSyncCoordinator(
        manifest_store=manifest_store,
        state_store=state_store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        settings=settings,
        activation_mode=activation_mode,
    )
    original = RawDocument("invoice.pdf", "previous complete text", "invoice")
    await coordinator.run_sync(scope="tenant", source=[original])
    before = await manifest_store.get("tenant", "invoice")
    generation = before.active_generation
    ids = list(before.chunk_ids)
    calls = embedder.call_count
    incomplete = extract_documents(mixed_pdf(), "invoice.pdf", "invoice", {}, {})
    report = await coordinator.run_sync(scope="tenant", source=incomplete)
    after = await manifest_store.get("tenant", "invoice")
    assert not report.authoritative
    assert after.active_generation == generation
    assert after.chunk_ids == ids
    assert embedder.call_count == calls
    assert await vector_store.get_embeddings(ids)
