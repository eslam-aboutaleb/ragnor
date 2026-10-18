"""Behavioral regressions for review findings R1–R28."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ragit.chunking.markdown import MarkdownSectionChunker
from ragit.chunking.sliding_window import SlidingWindowChunker
from ragit.embeddings.dimensions import register_dimension
from ragit.embeddings.executor import EmbeddingExecutor, EmbeddingPermanentError
from ragit.embeddings.profile import EmbeddingProfile, TaskMode
from ragit.evidence import ContextBudget, EvidenceAssembler, SearchHit
from ragit.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragit.evidence.tokenizer import ExactTokenizerAdapter
from ragit.evidence.types import EvidenceStatus, ScopeAuthorizationError
from ragit.ingestion.connector_core import LoadResult, extract_documents
from ragit.ingestion.pipeline import IngestionPipeline, VersionRecord
from ragit.ingestion.sources import ListDocumentSource, RawDocument
from ragit.jobs.outbox import InMemoryEmbeddingJobStore, process_pending_jobs
from ragit.mcp.server import create_mcp_server
from ragit.retrieval.capabilities import CapabilityUnsupportedError
from ragit.retrieval.retriever import HybridRetriever, RetrieverSpec
from ragit.retrieval.strategies import DenseRetrieval, HybridRetrieval, KeywordRetrieval
from ragit.retrieval.types import RetrievalRequest
from ragit.stores.memory import InMemoryVectorStore
from ragit.sync import (
    ActivationMode,
    DefaultSyncCoordinator,
    InMemoryManifestStore,
    InMemorySyncStateStore,
    ManifestStatus,
    StaleWorkerError,
    SyncRetriever,
)
from ragit.telemetry.health import HealthChecker, HealthState
from ragit.tools.bridge import CollectionConfig, KnowledgeBridge
from ragit.tools.security import ScopeAuthorizer
from tests.test_mcp_adapter import MockMCPServer


class Embedder:
    def __init__(self):
        self.calls = 0
        self.fail = None

    async def __call__(self, texts):
        self.calls += 1
        if self.fail and any(self.fail in text for text in texts):
            raise RuntimeError("embedding unavailable")
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def env():
    register_dimension("review-regression", 2)
    settings = SimpleNamespace(embedding_model="review-regression", rag_distance_threshold=5)
    embedder = Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    manifest = InMemoryManifestStore()
    state = InMemorySyncStateStore()
    coordinator = DefaultSyncCoordinator(
        manifest, state, store, embedder, SlidingWindowChunker(100, 0), settings
    )
    return SimpleNamespace(
        settings=settings,
        embedder=embedder,
        store=store,
        manifest=manifest,
        state=state,
        coordinator=coordinator,
    )


def doc(name="a", text="content", **kwargs):
    return RawDocument(name, text, **kwargs)


async def visible(env):
    return await SyncRetriever(env.store, env.manifest, env.state).hybrid_search(
        "content", [1.0, 0.0], "scope", threshold=5
    )


@pytest.mark.parametrize("operation", ["search", "source", "status", "sync", "revoke"])
async def test_unknown_collection_never_routes_to_private_data(env, operation):
    """R1: enforce collection routing on every tool, including management."""
    bridge = KnowledgeBridge(
        collections={"private": env.store},
        allow_management=True,
        authorizer=ScopeAuthorizer({"alice": {"public"}}),
    )
    options = {"collection": "public", "client_id": "alice"}
    with pytest.raises(KeyError):
        if operation == "search":
            await bridge.search_knowledge("content", **options)
        elif operation == "source":
            await bridge.get_source("repo", document_id="a", **options)
        elif operation == "status":
            await bridge.knowledge_status(**options)
        elif operation == "sync":
            await bridge.sync_collection(source_id="repo", documents=[], **options)
        else:
            await bridge.revoke_document(document_id="a", **options)


async def test_omitted_collection_uses_declared_default_but_explicit_unknown_does_not(env):
    bridge = KnowledgeBridge(env.store, default_collection="support")
    assert (await bridge.knowledge_status())["collection"] == "support"
    server = create_mcp_server(bridge, server_class=MockMCPServer)
    assert (await server.call_tool("knowledge_status", {}))["collection"] == "support"
    with pytest.raises(KeyError):
        await bridge.knowledge_status(collection="default")


@pytest.mark.parametrize("allowed", [{"A"}, set()])
async def test_collection_scope_allowlist_is_enforced(env, allowed):
    """R3: a global authorization grant cannot bypass a collection allowlist."""
    bridge = KnowledgeBridge(
        collections={"default": CollectionConfig("default", env.store, allowed_scopes=allowed)},
        allow_management=True,
    )
    for operation in (
        lambda: bridge.search_knowledge("content", scope="B"),
        lambda: bridge.get_source("repo", document_id="a", scope="B"),
        lambda: bridge.knowledge_status(scope="B"),
        lambda: bridge.sync_collection("default", "repo", [], scope="B"),
        lambda: bridge.revoke_document("default", "a", scope="B"),
    ):
        with pytest.raises(ScopeAuthorizationError):
            await operation()
    bridge.register_collection("empty", env.store, allowed_scopes=set())
    with pytest.raises(ScopeAuthorizationError):
        await bridge.search_knowledge("content", collection="empty", scope="A")


async def test_scoped_management_rejects_spoofing_and_preserves_colliding_ids(env):
    """R2: identical external IDs remain independent across tenants."""
    bridge = KnowledgeBridge(
        env.store, allow_management=True, authorizer=ScopeAuthorizer({"alice": {"A"}, "bob": {"B"}})
    )
    row = {"id": "same", "text": "A content", "embedding": [1.0, 0.0]}
    with pytest.raises(ScopeAuthorizationError):
        await bridge.sync_collection(
            "default",
            "repo",
            [{**row, "metadata": {"isolation_scope": "B"}}],
            scope="A",
            client_id="alice",
        )
    assert await env.store.count() == 0
    for scope, client in (("A", "alice"), ("B", "bob")):
        await bridge.sync_collection(
            "default", "repo", [{**row, "text": f"{scope} content"}], scope=scope, client_id=client
        )
        span = await bridge.get_source("repo", document_id="same", scope=scope, client_id=client)
        assert span.text == f"{scope} content"
    assert await env.store.count() == 2
    assert len(await env.store.get(filters={"isolation_scope": "A"})) == 1
    assert len(await env.store.get(filters={"isolation_scope": "B"})) == 1


async def test_revocation_blocks_old_citations_but_retains_history(env):
    """R4: retained bytes do not imply permission to read revoked content."""
    snapshots = MemorySnapshotStore()
    lookup = SourceLookup(snapshots)
    bridge = KnowledgeBridge(env.store, source_lookup=lookup, allow_management=True)
    await bridge.sync_collection(
        "default",
        "repo",
        [{"id": "a", "text": "sensitive content", "embedding": [1.0, 0.0]}],
        scope="A",
    )
    assert (await bridge.get_source("repo", document_id="a", scope="A")).text
    assert (await bridge.revoke_document("default", "a", "repo", scope="A"))["chunks_removed"] == 1
    assert (await bridge.get_source("repo", document_id="a", scope="A")).text == ""
    # A fresh lookup over the same store also enforces the durable revocation.
    assert SourceLookup(snapshots).lookup_span("repo", "a", "v1", "A").text == ""
    assert snapshots.has_revision("repo", "a", "v1")


async def test_public_and_tenant_revisions_coexist_and_batch_conflicts_do_not_publish(env):
    """R2/R26: immutable revision checks use write identity rather than public fallback."""
    bridge = KnowledgeBridge(env.store, allow_management=True)
    row = {"id": "a", "text": "public", "embedding": [1.0, 0.0]}
    await bridge.sync_collection("default", "repo", [row])
    await bridge.sync_collection("default", "repo", [{**row, "text": "private"}], scope="A")
    assert (await bridge.get_source("repo", document_id="a")).text == "public"
    assert (await bridge.get_source("repo", document_id="a", scope="A")).text == "private"
    count = await env.store.count()
    with pytest.raises(ValueError, match="same batch"):
        await bridge.sync_collection(
            "default", "repo", [{**row, "id": "b"}, {**row, "id": "b", "text": "conflicting"}]
        )
    assert await env.store.count() == count


async def test_expired_worker_cannot_corrupt_winner_after_delayed_vector_write(env, monkeypatch):
    """R5: a write already in flight remains isolated when its lease is superseded."""
    clock = [1000.0]
    monkeypatch.setattr("ragit.sync.state.time.time", lambda: clock[0])
    entered, resume = asyncio.Event(), asyncio.Event()
    original_upsert = env.store.upsert
    first = True

    async def delayed_upsert(rows, **kwargs):
        nonlocal first
        if first:
            first = False
            entered.set()
            await resume.wait()
        await original_upsert(rows, **kwargs)

    monkeypatch.setattr(env.store, "upsert", delayed_upsert)
    slow = asyncio.create_task(env.coordinator.run_sync("scope", [doc(text="stale content")]))
    await entered.wait()
    clock[0] += 100
    winner = DefaultSyncCoordinator(
        env.manifest,
        env.state,
        env.store,
        env.embedder,
        SlidingWindowChunker(100, 0),
        env.settings,
        worker_id="winner",
    )
    await winner.run_sync("scope", [doc(text="winning content")])
    resume.set()
    with pytest.raises(StaleWorkerError):
        await slow
    assert [row["document"] for row in await visible(env)] == ["winning content"]
    assert await env.store.count() == 1


async def test_collection_failure_preserves_active_content_and_retry_recovers(env):
    """R6: no staged fingerprint or status is mistaken for committed content."""
    env.coordinator._activation_mode = ActivationMode.COLLECTION
    await env.coordinator.run_sync("scope", [doc("a", "old A content"), doc("b", "old B content")])
    env.embedder.fail = "FAIL"
    with pytest.raises(RuntimeError):
        await env.coordinator.run_sync("scope", [doc("a", "new A content"), doc("b", "FAIL")])
    assert {row["document"] for row in await visible(env)} == {"old A content", "old B content"}
    assert all(
        record.status == ManifestStatus.ACTIVE for record in await env.manifest.list("scope")
    )
    env.embedder.fail = None
    await env.coordinator.run_sync("scope", [doc("a", "new A content"), doc("b", "new B content")])
    assert {row["document"] for row in await visible(env)} == {"new A content", "new B content"}
    assert await env.store.count() == 2


async def test_collection_readers_keep_old_snapshot_until_atomic_publication(env, monkeypatch):
    """R6: prepublished vectors cannot expose a partial collection release."""
    env.coordinator._activation_mode = ActivationMode.COLLECTION
    await env.coordinator.run_sync("scope", [doc("a", "old A content"), doc("b", "old B content")])
    entered, resume = asyncio.Event(), asyncio.Event()
    original_commit = env.manifest.compare_and_swap_many

    async def pause_commit(updates):
        entered.set()
        await resume.wait()
        return await original_commit(updates)

    monkeypatch.setattr(env.manifest, "compare_and_swap_many", pause_commit)
    update = asyncio.create_task(
        env.coordinator.run_sync("scope", [doc("a", "new A content"), doc("b", "new B content")])
    )
    await entered.wait()
    assert {row["document"] for row in await visible(env)} == {"old A content", "old B content"}
    resume.set()
    await update
    assert {row["document"] for row in await visible(env)} == {"new A content", "new B content"}


async def test_fenced_commit_excludes_takeover_even_when_lease_expires(env, monkeypatch):
    """R5: a separate token check cannot substitute for the commit guard."""
    clock = [1000.0]
    monkeypatch.setattr("ragit.sync.state.time.time", lambda: clock[0])
    token = await env.state.acquire_lease("scope", "first", 1)
    async with env.state.fenced(token):
        clock[0] += 2
        takeover = asyncio.create_task(env.state.acquire_lease("scope", "second", 30))
        await asyncio.sleep(0)
        assert not takeover.done()
        assert await env.state.save_checkpoint("scope", {"phase": "completed"}, token)
    assert (await takeover).epoch > token.epoch


async def test_long_preparation_renews_the_lease(env, monkeypatch):
    """The preparation phase stays eligible for activation after lease renewal."""
    monkeypatch.setattr("ragit.sync.state.time.time", lambda: 1000.0)
    renewed = asyncio.Event()
    original_renew = env.state.renew_lease

    async def renew(token, ttl):
        result = await original_renew(token, ttl)
        renewed.set()
        return result

    async def embed(texts):
        await asyncio.wait_for(renewed.wait(), 1)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(env.state, "renew_lease", renew)
    env.coordinator._lease_ttl_seconds = 0.03
    env.coordinator._embedder = embed
    assert (await env.coordinator.run_sync("scope", [doc()])).success


async def test_pdf_pages_are_one_document_with_distinct_chunks_and_locations(env):
    """R7: extracting multiple pages must not lose all but the last page."""

    class Pages:
        id = "pdf"

        def extract_pages(self, data):
            return [(1, "first content"), (2, "second content"), (10, "last content")]

    result = extract_documents(b"", "manual.pdf", "manual.pdf", {}, {}, Pages())
    report = await env.coordinator.run_sync("scope", result)
    rows = await env.store.get(metadata_fields=["doc_id", "location", "page"])
    assert report.added_count == 1
    assert {row["document"] for row in rows} == {"first content", "second content", "last content"}
    assert {row["metadata"]["page"] for row in rows} == {1, 2, 10}
    assert all(row["metadata"]["doc_id"] == "manual.pdf" for row in rows)
    assert (await env.coordinator.run_sync("scope", result)).unchanged_count == 1


async def test_location_fields_cannot_override_sync_ownership_or_generation(env):
    location = {
        "page": 1,
        "scope": "other",
        "doc_id": "forged",
        "generation": 99,
        "source": "forged",
    }
    report = await env.coordinator.run_sync("scope", [doc(location=location)])
    assert report.success
    rows = await visible(env)
    assert len(rows) == 1 and rows[0]["document"] == "content"
    stored = await env.store.get(
        metadata_fields=["scope", "doc_id", "generation", "source", "page", "location"]
    )
    assert stored[0]["metadata"] == {
        "scope": "scope",
        "doc_id": "a",
        "generation": 1,
        "source": "a",
        "page": 1,
        "location": location,
    }


async def test_metadata_update_changes_filters_without_reembedding(env):
    """R8: business filters and returned evidence use the committed metadata."""
    await env.coordinator.run_sync("scope", [doc(metadata={"category": "public", "old": "remove"})])
    old_ids = (await env.manifest.get("scope", "a")).chunk_ids
    vectors = await env.store.get_embeddings(old_ids)
    calls = env.embedder.calls
    report = await env.coordinator.run_sync("scope", [doc(metadata={"category": "private"})])
    assert report.metadata_only_count == 1 and report.embedded_count == 0
    assert env.embedder.calls == calls
    assert await env.store.get(filters={"category": "public"}) == []
    rows = await env.store.get(filters={"category": "private"}, metadata_fields=["old"])
    assert len(rows) == 1 and "old" not in rows[0]["metadata"]
    assert list((await env.store.get_embeddings([rows[0]["id"]])).values()) == list(
        vectors.values()
    )


async def test_typed_request_keeps_generation_capabilities_session_and_projection(env, monkeypatch):
    """R9: constraints survive the public HybridRetriever adapter."""
    import ragit.retrieval.retriever as module

    monkeypatch.setattr(module, "get_embedding_function", lambda _: env.embedder)
    monkeypatch.setattr(module, "get_vector_store", lambda **_: env.store)
    for generation in (1, 2):
        await env.store.upsert(
            [
                {
                    "id": str(generation),
                    "text": "content",
                    "embedding": [1.0, 0.0],
                    "metadata": {"generation": generation, "tenant_id": "A"},
                }
            ]
        )
    retriever = HybridRetriever(RetrieverSpec("docs"), env.settings)
    request = RetrievalRequest(
        "content", threshold=5, expected_generation=2, isolation_scope="A", session=object()
    )
    assert [row.id for row in await retriever.retrieve(request)] == ["2"]
    assert [row.id for row in await retriever.retrieve_typed(request)] == ["2"]
    with pytest.raises(CapabilityUnsupportedError):
        await retriever.retrieve(replace(request, mandatory_capabilities={"transactions"}))
    captured = []
    original = env.store.hybrid_search

    async def capture(**kwargs):
        captured.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(env.store, "hybrid_search", capture)
    await retriever.retrieve(request)
    assert captured[0]["session"] is request.session
    assert {"tenant_id", "generation"} <= set(captured[0]["metadata_fields"])


async def test_static_upsert_fields_override_source_owned_metadata(env):
    """R10: static ownership remains trusted after shared-helper refactoring."""
    await env.store.upsert(
        [
            {
                "id": "a",
                "text": "content",
                "embedding": [1.0, 0.0],
                "metadata": {"tenant_id": "untrusted"},
            }
        ],
        extra_fields={"tenant_id": "trusted"},
    )
    assert len(await env.store.get(filters={"tenant_id": "trusted"})) == 1
    assert await env.store.get(filters={"tenant_id": "untrusted"}) == []


async def test_complete_paginated_snapshot_removes_missing_documents(env):
    """R11: only the complete traversal, rather than each page, establishes authority."""

    class Pages:
        async def load_page(self, *, cursor=None, limit=None):
            if cursor is None:
                return LoadResult([doc("a")], has_more=True, next_cursor="second")
            return LoadResult([doc("b")], cursor=cursor)

    await env.coordinator.run_sync("scope", [doc("a"), doc("b"), doc("deleted")])
    report = await env.coordinator.run_sync("scope", Pages())
    assert report.authoritative and report.removed_count == 1
    assert (await env.manifest.get("scope", "deleted")).status == ManifestStatus.TOMBSTONED


@pytest.mark.parametrize(
    "result",
    [
        LoadResult(complete=False, status="transient", errors=[{}]),
        LoadResult(incremental=True),
        LoadResult(complete=False),
    ],
)
async def test_loaded_partial_or_incremental_result_never_authorizes_deletion(env, result):
    """R12: list coercion and explicit authority flags cannot bless failed coverage."""
    await env.coordinator.run_sync("scope", [doc()])
    report = await env.coordinator.run_sync("scope", result, is_authoritative=True)
    assert not report.authoritative
    assert (await env.manifest.get("scope", "a")).status == ManifestStatus.ACTIVE


async def test_legacy_document_source_and_nested_chunker_configuration(env):
    """R13/R14: legacy source signatures and protocol chunker configs remain usable."""
    source = ListDocumentSource([doc(text="one two three four five six")])
    env.coordinator._chunker = MarkdownSectionChunker(SlidingWindowChunker(100, 0))
    await env.coordinator.run_sync("scope", source)
    env.coordinator._chunker = MarkdownSectionChunker(SlidingWindowChunker(2, 0))
    report = await env.coordinator.run_sync("scope", source)
    assert report.changed_count == 1
    assert len((await env.manifest.get("scope", "a")).chunk_ids) > 1


async def test_failed_scan_cannot_replace_successful_sync_health(env):
    """R15: report and health retain the distinction between outage and no change."""
    await env.coordinator.run_sync("scope", [doc()])
    successful = (await env.state.get_checkpoint("scope"))["last_successful_sync"]

    class Offline:
        async def load_page(self, **kwargs):
            return LoadResult(status="transient", complete=False, errors=[{"reason": "offline"}])

    report = await env.coordinator.run_sync("scope", Offline())
    assert not report.success and report.errors
    assert (await env.state.get_checkpoint("scope"))["last_successful_sync"] == successful
    assert (
        await HealthChecker(state_store=env.state).check_sync_status("scope")
    ).state == HealthState.DEGRADED


@pytest.mark.parametrize(
    "kind", [DenseRetrieval, HybridRetrieval, KeywordRetrieval, HybridRetriever]
)
async def test_bridge_uses_real_retriever_threshold_and_scope_contract(env, monkeypatch, kind):
    """R16/R17: real adapters return the same scoped evidence as direct search."""
    await env.store.upsert(
        [{"id": "a", "text": "content", "embedding": [1.0, 0.0], "metadata": {"tenant_id": "A"}}]
    )
    if kind is HybridRetrieval:
        strategy = kind(env.store, env.store, env.embedder)
    elif kind is DenseRetrieval:
        strategy = kind(env.store, env.embedder)
    elif kind is KeywordRetrieval:
        strategy = kind(env.store)
    else:
        monkeypatch.setattr(
            "ragit.retrieval.retriever.get_embedding_function", lambda _: env.embedder
        )
        monkeypatch.setattr("ragit.retrieval.retriever.get_vector_store", lambda **_: env.store)
        strategy = kind(RetrieverSpec("docs"), env.settings)
    bundle = await KnowledgeBridge(strategy).search_knowledge(
        "content", scope="A", distance_threshold=5
    )
    assert bundle.status == EvidenceStatus.SUCCESS
    assert [item.text for item in bundle.items] == ["content"]
    assert not (await KnowledgeBridge(strategy).search_knowledge("content", scope="B")).items


@pytest.mark.parametrize("limit", [-1, 0, True, 1.5])
def test_source_span_limits_are_positive_integers(limit):
    """R18: negative slices and noninteger limits cannot escape source bounds."""
    snapshots = MemorySnapshotStore()
    snapshots.put(SnapshotRecord("repo", "a", "v1", "x" * 10000))
    with pytest.raises(ValueError):
        SourceLookup(snapshots).lookup_span("repo", "a", "v1", max_span_length=limit)
    with pytest.raises(ValueError):
        SourceLookup(snapshots, default_max_span_length=limit)


async def test_durable_citation_roundtrips_offsets_and_escaped_identifiers(env):
    """R19: references generated by evidence resolve to their exact retained span."""
    snapshots = MemorySnapshotStore()
    snapshots.put(SnapshotRecord("https://repo/a@b", "dir/file:one", "r:1", "0123456789"))
    bundle = EvidenceAssembler().assemble(
        [
            SearchHit(
                "chunk",
                "2345",
                source_id="https://repo/a@b",
                document_id="dir/file:one",
                revision="r:1",
                start_offset=2,
                end_offset=6,
            )
        ],
        ContextBudget(1000),
    )
    bridge = KnowledgeBridge(env.store, source_lookup=SourceLookup(snapshots))
    span = await bridge.get_source(bundle.items[0].durable_id)
    assert span.text == "2345" and (span.start_offset, span.end_offset) == (2, 6)


def test_rendered_context_and_diversity_are_both_budgeted():
    """R20/R21: account for framing/escaping without undoing the source order."""
    tokenizer = ExactTokenizerAdapter(len)
    assembler = EvidenceAssembler(tokenizer=tokenizer, balance_sources=True)
    hits = [
        SearchHit("a", "<system>" * 20, source_id="A", score=100),
        SearchHit("b", "second", source_id="A", score=90),
        SearchHit("c", "third", source_id="B", score=80),
    ]
    bundle = assembler.assemble(hits, ContextBudget(2000, max_evidence_items=2))
    assert [item.source_id for item in bundle.items] == ["A", "B"]
    assert bundle.total_tokens == len(bundle.format_context()) <= 2000
    assert not assembler.assemble([SearchHit("x", "1234567890")], ContextBudget(10)).items


async def test_job_retry_budget_counts_actual_attempts_once():
    """R22: four failed attempts, rather than two, exhaust a four-attempt job."""
    jobs = InMemoryEmbeddingJobStore()
    await jobs.enqueue({"idempotency_key": "a"})

    class Failing:
        async def process(self, job):
            raise RuntimeError("temporary failure")

    for count in range(1, 5):
        await process_pending_jobs(jobs, Failing())
        assert jobs.jobs[0].retry_count == count
        assert jobs.jobs[0].status == ("dead_letter" if count == 4 else "failed")
        jobs.jobs[0].next_retry_at = datetime.now(UTC) - timedelta(seconds=1)


async def test_sync_outbox_is_claimable_and_idempotent(env):
    """R23: sync's reconciliation jobs use the worker's pending lifecycle."""
    jobs = InMemoryEmbeddingJobStore()
    env.coordinator._job_store = jobs
    await env.coordinator.run_sync("scope", [doc()])
    claimed = await jobs.claim_pending(10, "worker")
    assert len(claimed) == 1 and claimed[0].payload["operation"] == "reconcile_generation"
    await jobs.enqueue(claimed[0].payload)
    assert len(jobs.jobs) == 1


@pytest.mark.parametrize("change", ["profile", "metadata"])
async def test_ingestion_reindexes_profile_and_metadata_changes(env, change):
    """R24/R28: real pipeline snapshots reflect all effective ingestion inputs."""

    class Versions:
        active = None

        async def find_active(self, session):
            return self.active

        async def close_active(self, session):
            pass

        async def delete_chunks(self, session, version_id):
            pass

        async def create_version(self, session, snapshot):
            self.active = VersionRecord("v1", snapshot)
            return self.active

    @asynccontextmanager
    async def sessions():
        class Session:
            async def commit(self):
                pass

        yield Session()

    env.embedder.profile = EmbeddingProfile("one", "review-regression", 2)
    pipeline = IngestionPipeline(
        env.store, env.embedder, SlidingWindowChunker(100, 0), sessions, env.settings
    )
    versions = Versions()
    source = ListDocumentSource([doc(metadata={"category": "public"})])
    await pipeline.run(source, versions)
    previous = versions.active.snapshot
    if change == "profile":
        env.embedder.profile = replace(env.embedder.profile, provider_id="two")
    else:
        source = ListDocumentSource([doc(metadata={"category": "private"})])
    await pipeline.run(source, versions)
    assert env.embedder.calls == 2 and versions.active.snapshot != previous
    assert versions.active.snapshot.embedding_profile_hash == env.embedder.profile.canonical_hash()


async def test_task_modes_reach_provider_and_reject_unsupported_modes():
    """R25: explicit query/document overrides must change provider execution."""
    seen = []

    async def provider(texts, *, task_mode):
        seen.append(task_mode)
        return [[1.0, 0.0] for _ in texts]

    profile = EmbeddingProfile("one", "review-regression", 2, task_mode=TaskMode.DOCUMENT)
    executor = EmbeddingExecutor(provider, profile)
    await executor.embed_documents(["content"])
    await executor.embed_query("query")
    assert seen == [TaskMode.DOCUMENT, TaskMode.QUERY]
    with pytest.raises(EmbeddingPermanentError):
        await EmbeddingExecutor(Embedder(), profile).embed(["content"])


def test_snapshot_revisions_are_immutable_scoped_and_defensively_copied():
    """R26: another tenant or a caller-owned dict cannot rewrite historical evidence."""
    store = MemorySnapshotStore()
    record = SnapshotRecord("repo", "a", "v1", "A", "A", {"nested": {"value": 1}})
    store.put(record)
    store.put(record)
    with pytest.raises(ValueError):
        store.put(replace(record, content="replacement"))
    record.metadata["nested"]["value"] = 2
    store.put(SnapshotRecord("repo", "a", "v1", "B", "B"))
    assert store.get_scoped("repo", "a", "v1", "A").metadata["nested"]["value"] == 1
    assert SourceLookup(store).lookup_span("repo", "a", "v1", "B").text == "B"


async def test_mcp_uses_trusted_identity_for_all_tools(env):
    """R27: tool callers cannot spoof the client ID used for scope grants."""
    bridge = KnowledgeBridge(
        env.store, allow_management=True, authorizer=ScopeAuthorizer({"alice": {"A"}})
    )
    server = create_mcp_server(
        bridge,
        server_class=MockMCPServer,
        enable_management_tools=True,
        client_id_provider=lambda: "alice",
    )
    await server.call_tool(
        "sync_collection",
        {
            "collection": "default",
            "source_id": "repo",
            "documents": [{"id": "a", "text": "content", "embedding": [1.0, 0.0]}],
            "scope": "A",
        },
    )
    assert (await server.call_tool("search_knowledge", {"query": "content", "scope": "A"}))["items"]
    assert (
        await server.call_tool(
            "get_source", {"source_ref": "repo", "document_id": "a", "scope": "A"}
        )
    )["text"]
    assert await server.call_tool("knowledge_status", {"scope": "A"})
    with pytest.raises(TypeError):
        await server.call_tool("search_knowledge", {"query": "content", "client_id": "admin"})
    with pytest.raises(ScopeAuthorizationError):
        await server.call_tool("search_knowledge", {"query": "content", "scope": "B"})
    await server.call_tool(
        "revoke_document",
        {"collection": "default", "source_id": "repo", "document_id": "a", "scope": "A"},
    )
    assert not (
        await server.call_tool(
            "get_source", {"source_ref": "repo", "document_id": "a", "scope": "A"}
        )
    )["text"]
