---
name: ragnor
description: >-
  Guide for integrating the ragnor domain-agnostic RAG toolkit: embeddings,
  chunking, vector stores, hybrid retrieval, ingestion, background jobs, and
  evaluation. Use when building RAG features, wiring a host application to
  ragnor, or choosing/adding a vector backend (memory, pgvector, qdrant,
  chroma, faiss).
---

# ragnor Integration Guide

ragnor is a RAG _subsystem_, not an application: chunking, embeddings, vector stores, hybrid retrieval, idempotent ingestion, background jobs, and evaluation. Every host-specific dependency (settings, session factory, version/job tables) is injected through protocols and case-insensitive provider registries — ragnor never imports the host.

## Quick start

The high-level convenience API (`ragnor.builder`) needs three lines — `RagnorBuilder.local(embedder=...).build()` → `await rag.ingest([Document(...)])` → `await rag.search("...")` — with fully offline defaults: the in-memory vector store, the sliding-window chunker, the built-in `InMemoryVersionStore`, and a null session provider (no database, no driver, no API key). ragnor ships no local embedder in core, so the builder requires an injected embedder; `examples/local_quickstart.py` defines a deterministic fake and runs with the core install only (`pip install ragnor`).

```python
from ragnor.builder import RagnorBuilder
from ragnor.types import Document

rag = await RagnorBuilder.local(embedder=my_embedder).build()
await rag.ingest([Document(id="policy", text="The refund policy covers water damage.")])
results = await rag.search("refund policy")
```

The builder sits **on the low-level seams**: it wires the existing `IngestionPipeline` and `HybridRetriever` unchanged (idempotent snapshot ingestion; version replacement that deletes retired chunks through the store's `delete` capability), and every part is overridable — `with_store`, `with_embedder`, `with_chunker`, `with_version_store`, `with_settings` — while `rag.store`, `rag.pipeline`, and `rag.retriever` keep the low-level seams reachable. For production databases, drop to those seams with your real session factory and version tables (the builder's defaults are the offline path), or pass your own store and `VersionStore` through the builder's overrides.

## Install

Install **`ragnor`** from PyPI. The Python import name is `ragnor`.

```bash
pip install ragnor

# With an extra
pip install "ragnor[pgvector]"
```

As a uv/poetry dependency in `pyproject.toml`:

```toml
ragnor
```

Extras: `[litellm]` (litellm>=1.84 — LiteLLM embeddings provider and LLM judge), `[pgvector]` (sqlalchemy[asyncio]), `[qdrant]` (qdrant-client), `[chroma]` (chromadb), `[faiss]` (faiss-cpu), `[local-embeddings]` (reserved, currently empty — future local-model backend), `[all]`, `[test]` (pytest, pytest-asyncio, aiosqlite, sqlalchemy[asyncio]). Requires **Python >= 3.11**. Core has no third-party dependencies — no database driver is imported until a backend is selected, and litellm is imported only when the LiteLLM integration is used.

## Core concepts

**Protocol seams** (host implements/injects, ragnor consumes): `RagSettings` (settings fields; `@runtime_checkable` — validate with `isinstance(settings, RagSettings)`), `SessionProvider` (async session factory), `VersionStore` (`find_active`, `close_active`, `create_version`, `delete_chunks`), `EmbeddingJobStore` (outbox storage), `DocumentSource` (`async load() -> list[RawDocument]`), `EmbeddingFunction` (async `__call__`, `embed_query`, `embed_documents`), `Chunker` (`chunk(text) -> list[Chunk]` with `version`/`config`), `VectorStore` (ABC: `hybrid_search`, `count`, `upsert`).

**Registries** — `embedding_registry`, `vector_store_registry`, `chunker_registry`: case-insensitive name lookup; unknown names raise `KeyError` listing registered providers.

**Two abstract factories**: `SQLFactory` (every SQL operation — sessions, queries, mutations, advisory locking; built-in `SQLAlchemyFactory`) and `VectorStoreFactory` (creates stores; built-in `RegistryVectorStoreFactory` reads `settings.vector_store_provider`).

**Design invariants** (pinned by tests): ragnor never imports the host; no module-level singletons (host creates every registry/engine/session factory per process); idempotent ingestion via snapshot comparison; outbox atomicity (job row committed in the same transaction as the row it refers to); store failures swallowed unless `strict=True`; all SQL identifiers pass `validate_identifier` (`^[A-Za-z_][A-Za-z0-9_]*$`) and all values are bound parameters.

## Choose a vector backend

Selected via `settings.vector_store_provider`. **Default is `"memory"`** (not pgvector). Backend modules load lazily — driver dependencies stay optional until a backend is selected.

Every built-in backend satisfies the three capability Protocols structurally: `VectorIndex` (`vector_search`), `LexicalIndex` (`keyword_search`), and `DocumentStore` (`upsert`/`count`/`delete`/`get`). The "lexical capability" column says whether the keyword stage is a **native full-text index** (production-scale) or an **in-process fallback** (full-corpus token-overlap scan — local development, small corpora, and testing only). At scale, combine a pgvector `VectorIndex` with an external lexical engine adapted to `LexicalIndex`.

Distance metric: the memory, pgvector, and FAISS stores support `l2` (default) and `cosine`, selectable per store with the `metric=` constructor argument or declaratively through the optional `rag_distance_metric` host settings field; the Qdrant and Chroma stores reject `cosine` at construction time and keep the `l2` default.

| Provider     | Class                 | Use                        | Lexical capability                          | Notes                                                                                               |
| ------------ | --------------------- | -------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `"memory"`   | `InMemoryVectorStore` | dev/test, zero-dep         | in-process fallback (token overlap)         | brute-force search over `l2` (default) or `cosine` via `metric=`; token-overlap keyword approximation; nothing persisted |
| `"pgvector"` | `PgVectorStore`       | PostgreSQL production      | **native FTS** (`ts_rank`, production-scale) | real pgvector `<->` (l2, default) or `<=>` (cosine) via `metric=`; `ts_rank` full-text search; needs `[pgvector]` extra |
| `"qdrant"`   | `QdrantStore`         | Qdrant server              | in-process fallback (token overlap)         | native vector search + in-process token-overlap keyword; `url`/`api_key` or `settings.database_url` |
| `"chroma"`   | `ChromaStore`         | embedded/persistent/server | in-process fallback (token overlap)         | `path=` for persistent, else ephemeral; sync client wrapped in `asyncio.to_thread`                  |
| `"faiss"`    | `FaissStore`          | in-process exact L2        | in-process fallback (token overlap)         | `IndexIDMap` over `IndexFlatL2` (l2, default) or normalized cosine via `metric=`; nothing persisted |

```python
from ragnor.stores import get_vector_store

store = get_vector_store(
    table_name="policy_chunks",
    embedding_dim=1536,
    settings=settings,
    embedding_fn=embedder,  # auto-embed queries
)
```

Session ownership rule: when the caller supplies a `session`, the caller owns the transaction (commit/rollback); when omitted, the store opens its own session and commits.

Retrieval strategies (`ragnor.retrieval.strategies`) compose the capability Protocols instead of the fused store: `DenseRetrieval(vector_index, embedder)`, `KeywordRetrieval(lexical_index)` (keyword-only, **no embedding dependency**), and `HybridRetrieval(vector_index, lexical_index, embedder, reranker=None)` — the two stages may be different objects (e.g. a pgvector `VectorIndex` plus an external lexical engine adapted to `LexicalIndex`). Inject a strategy into `HybridRetriever(spec, settings, strategy=...)` to bypass eager embedder/store construction.

## Wire ingestion

```python
from ragnor.chunking import MarkdownSectionChunker, SlidingWindowChunker
from ragnor.ingestion import FileDocumentSource, IngestionPipeline

pipeline = IngestionPipeline(
    store=store,
    embedder=embedder,
    chunker=chunker,
    session_factory=my_session_factory,
    settings=settings,
)

# lock_key: PostgreSQL advisory lock serializes concurrent ingestion
count = await pipeline.run(
    FileDocumentSource("docs/policy.md"),
    version_store,
    lock_key="docs/policy.md",
)
```

- **Idempotent**: the pipeline builds an `IngestionSnapshot` (source hash, embedding model/dimension, chunker version/tunables); if it matches the active version's snapshot, ingestion is skipped — never re-embedded.
- **Version lifecycle** on change: close active version → delete retired version's chunks (chunk ids are position-based) → create new version row → embed and upsert.
- **Abort safety**: if chunking yields no chunks, the pipeline aborts before touching the version table.
- **Advisory locking** (`lock_key`) is PostgreSQL-only (`pg_advisory_lock(hashtext(key))`); on other stacks pass `lock_key=None` and serialize ingestion in your own infrastructure.
- Host implements `VersionStore`; all methods receive the pipeline's session, so every mutation joins the ingestion transaction.

## Wire retrieval

```python
from ragnor.retrieval import HybridRetriever, RetrieverSpec, as_dicts

retriever = HybridRetriever(
    spec=RetrieverSpec(table_name="policy_chunks", metadata_fields=["section", "source"]),
    settings=settings,
    session_factory=my_session_factory,
)

results = await retriever.retrieve(
    "What is the refund policy?",
    n_results=5,
    distance_threshold=0.5,  # defaults to settings.rag_distance_threshold
    owner_id="user-123",  # any equality filter, applied to both stages
)
rows = as_dicts(results)
```

Two-stage search: dense vector search under the distance threshold + sparse keyword search, each ranked and truncated to `n_results`, fused with Reciprocal Rank Fusion: `score = 1/(60 + vector_rank) + 1/(60 + keyword_rank)`. A document found by both stages outranks a single-stage hit. `RetrieverSpec` validates every SQL identifier at construction; `embedding_dim` defaults to the dimension registered for `settings.embedding_model`.

## Background jobs

```python
from ragnor.jobs import enqueue_job, process_pending_jobs, reclaim_stale_jobs
from ragnor.jobs.drainer import drain_forever, main

# Outbox: with a session the job row is written in the SAME transaction
await enqueue_job(store, {"text": "...", "metadata": {...}}, session=my_session)

processed = await process_pending_jobs(store, processor, limit=10, worker_id="worker-1")
await reclaim_stale_jobs(store, stale_after_seconds=300.0)

# Programmatic drainer (runs until the stop event) or console entry point
await drain_forever(
    store, processor, interval=5.0, batch_size=10, worker_id="worker-1", stop=stop_event
)
main(store, processor, interval=5.0, batch_size=10, worker_id="worker-1")
```

Claiming uses `FOR UPDATE SKIP LOCKED` (concurrent drainers scale horizontally without coordination). Failed jobs retry with backoff (1m, 5m, 30m); jobs exhausting `MAX_ATTEMPTS` (4) move to terminal `dead_letter`. `EmbeddingJobStore` and `JobProcessor` are host-implemented protocols.

## Evaluation

```python
from ragnor.evaluation import LabeledQuery, RagEvaluationHarness, run_evaluation, EvalConfig

harness = RagEvaluationHarness(
    [
        LabeledQuery(query="refund policy?", expected_chunk_ids={"chunk_0"}),
    ]
)
metrics = await harness.evaluate(retrieval_fn)  # async (query: str) -> list[result dicts]
# metrics: recall_at_k, precision_at_k, mrr, latency, errors

result = await run_evaluation(
    EvalConfig(mode="end-to-end", judge="heuristic", n_results=5),
    retrieval_fn,
    dataset,
    settings=settings,
    answer_fn=answer_fn,  # async (query: str, context: str) -> answer
)
```

Answer-quality judges: `evaluate_answer_heuristic` (free, deterministic, token-overlap; pass `domain_terms` to boost completeness) vs `evaluate_answer_llm` (LiteLLM calls; **falls back to the heuristic on any failure**). `run_evaluation` requires injected `retrieval_fn` and `dataset` (raises `ValueError` otherwise) — ragnor never imports a host retriever. Modes: `retrieval-only` (retrieval quality only), `context` (scores the joined retrieved context against the gold answer — the legacy behavior under an honest name; measures whether the chunks contain the answer, not whether a model answers faithfully), or `end-to-end` (judges a **real model output** from the injected `answer_fn`; the judges receive the actual answer and the actual retrieved context as separate inputs). `end-to-end` **requires** `answer_fn` and raises `ValueError` naming the three modes otherwise — a loud failure beats a silently wrong metric. Per-sample results carry `context`, `generated_answer`, `gold_answer`, and `chunks_found` separately.

## Register custom providers

**Embedding provider** (selected via `settings.embedding_provider`):

```python
from ragnor.embeddings import embedding_registry
from ragnor.embeddings.base import EmbeddingFunction


@embedding_registry.register("my-provider")
class MyEmbeddingProvider:
    def __init__(self, settings) -> None:
        self._settings = settings

    def create(self) -> EmbeddingFunction:
        return MyEmbedder(self._settings.openai_api_key)
```

**Custom vector backend** (selected via `settings.vector_store_provider`). Two paths:

**Path A — implement the capability Protocols structurally** (recommended). Implement `vector_search`, `keyword_search`, `upsert`, `count`, `delete`, and `get` — no ragnor inheritance needed, `isinstance(your_store, VectorIndex)` etc. hold automatically because the Protocols are `@runtime_checkable`. For a backend that wants the RRF-fused `hybrid_search`, subclass `HybridVectorStore` and implement the two search stages; the base provides the composite `hybrid_search`, `delete`/`get` semantics (at least one selector, empty `ids` rejected, conjunction when both supplied), and the uniform registry constructor kwargs:

```python
from ragnor.stores import HybridVectorStore, vector_store_registry


@vector_store_registry.register("my-backend")
class MyVectorStore(HybridVectorStore):
    def __init__(
        self,
        table_name="documents",
        id_field="id",
        embedding_dim=None,
        session_factory=None,
        settings=None,
        embedding_fn=None,
    ):
        super().__init__(
            table_name=table_name,
            id_field=id_field,
            embedding_dim=embedding_dim,
            session_factory=session_factory,
            settings=settings,
            embedding_fn=embedding_fn,
        )

    async def vector_search(
        self,
        embedding,
        n_results,
        threshold,
        text_field="text",
        metadata_fields=None,
        session=None,
        **filters,
    ): ...  # always raises on failure

    async def keyword_search(
        self, query, n_results, text_field="text", metadata_fields=None, session=None, **filters
    ): ...

    async def upsert(
        self,
        documents,
        session=None,
        text_field="text",
        embedding_field="embedding",
        id_field=None,
        extra_fields=None,
    ): ...
    async def count(self, session=None, strict=False, **filters) -> int: ...
    async def delete(self, ids=None, filters=None, session=None) -> int: ...
    async def get(
        self, ids=None, filters=None, session=None, text_field="text", metadata_fields=None
    ) -> list[dict]: ...
```

**Path B — keep the legacy fused interface.** If your backend already implements only `hybrid_search`/`count`/`upsert` (a fused hybrid result cannot be decomposed into independent vector and keyword stages), wrap it in `LegacyHybridStoreAdapter` — the adapter delegates the three legacy methods and deliberately claims **no** capabilities (`isinstance(adapter, VectorIndex)` is `False`), so it never pretends to satisfy the Protocols:

```python
from ragnor.stores import LegacyHybridStoreAdapter, vector_store_registry


@vector_store_registry.register("my-legacy-backend")
class MyLegacyStore(LegacyHybridStoreAdapter):
    def __init__(
        self,
        table_name="documents",
        id_field="id",
        embedding_dim=None,
        session_factory=None,
        settings=None,
        embedding_fn=None,
    ):
        super().__init__(backend=_MyLegacyBackend(...))
```

The constructor **must accept the registry factory kwargs** (`table_name`, `id_field`, `embedding_dim`, `session_factory`, `settings`, `embedding_fn`) because `get_vector_store` instantiates it with them.

## Tool Bridge and Model Context Protocol (MCP)

ragnor provides a single **framework-neutral tool bridge** (`KnowledgeBridge`) that unifies retrieval, bounded evidence assembly, immutable citation resolution, and health monitoring. Framework integrations (MCP servers, agent runtimes, APIs) call this bridge rather than duplicating retrieval and authorization logic.

### Skill vs MCP Distinction

- **Skill (Instructions & Guidelines)**: Human- and LLM-readable documentation detailing when to retrieve, how to formulate queries, how to interpret provenance metadata and citation labels, and what security boundaries apply.
- **MCP Server (Executable Surface)**: A live, machine-callable JSON-RPC 2.0 integration surface exposing tools (`search_knowledge`, `get_source`, `knowledge_status`) over standard input/output (stdio) or remote network transports.

### Direct Async Python Usage

```python
from ragnor.tools import KnowledgeBridge
from ragnor.evidence import SourceLookup, MemorySnapshotStore, EvidenceAssembler

bridge = KnowledgeBridge(
    retriever=my_retriever,
    default_collection="support",
    source_lookup=SourceLookup(MemorySnapshotStore()),
    assembler=EvidenceAssembler(),
    allow_management=False,  # Read-only by default
)

# Search knowledge with budget accounting and filters
bundle = await bridge.search_knowledge(
    query="refund guidelines",
    collection="support",
    filters={"locale": "en-US"},
    budget_limit=2000,
    scope="tenant_123",
)

# Fetch bounded authorized source span
span = await bridge.get_source(
    source_ref="kb_repo",
    document_id="faq_refund",
    revision="v1",
    start_offset=0,
    end_offset=500,
    scope="tenant_123",
)

# Inspect sanitized collection status
status = await bridge.knowledge_status(collection="support", scope="tenant_123")
```

### Running the MCP Adapter

Install with the `mcp` extra:

```bash
pip install "ragnor[mcp]"
```

Launch a local stdio server for desktop clients or CLI agents:

```python
import asyncio
from ragnor.mcp import create_mcp_server, run_stdio_server

# Read tools are exposed by default; management tools require explicit opt-in
server = create_mcp_server(bridge, name="ragnor-knowledge", enable_management_tools=False)

# Run over local stdio
asyncio.run(run_stdio_server(bridge))
```

For remote network transports, configure `RemoteTransportConfig` with host-provided authentication, rate limiting, and scope mapping:

```python
from ragnor.tools import RemoteTransportConfig, RateLimiter, ScopeAuthorizer

config = RemoteTransportConfig(
    auth_token_validator=lambda token: token == "secret-host-token",
    rate_limiter=RateLimiter(requests_per_minute=120, max_concurrent=20),
    scope_authorizer=ScopeAuthorizer(scope_mapping={"client_1": {"tenant_123"}}),
    host="0.0.0.0",
    port=8080,
)
```

### Tool Semantics

| Tool | Inputs | Output | Description |
| ---- | ------ | ------ | ----------- |
| `search_knowledge` | `query`, `collection`, `filters`, `candidate_k`, `final_k`, `budget_limit`, `scope` | Structured `EvidenceBundle` | Executes hybrid search with filter pushdown, deduplication, optional source diversity balancing, and budget constraints. |
| `get_source` | `source_ref`, `document_id`, `revision`, `start_offset`, `end_offset`, `collection`, `scope` | `BoundedSpanResult` | Fetches an authorized, bounded span from an immutable revision snapshot. Never substitutes newer content for old citations. |
| `knowledge_status` | `collection`, `scope`, `include_expensive` | Sanitized status summary | Reports system readiness (`ready`, `degraded`, `unavailable`, `unknown`), active document/chunk counts, and freshness. |
| `sync_collection` *(Opt-in)* | `collection`, `source_id`, `documents`, `scope` | Ingestion report | Ingests document batches. Guarded by `allow_management=True`. |
| `revoke_document` *(Opt-in)* | `collection`, `document_id`, `source_id`, `scope` | Deletion report | Authoritatively removes document and active chunks from search visibility. |

### Citations, Provenance & Immutability

- **Citation Labels vs Durable Identity**: Labels like `[1]` or `[2]` are ephemeral, deterministic presentation markers inside an individual `EvidenceBundle`. Durable citation identity is composed of `source_id@revision:document_id[start:end]`.
- **Immutable Historical Snapshots**: When looking up evidence from an earlier search, `get_source` requires the exact stated revision. If that revision is no longer retained, it returns `revision_unavailable` (or raises `RevisionUnavailableError`). ragnor **never** silently substitutes current content for historical citations.

### Error Handling & Boundaries

- **Distinct Statuses**: `EvidenceStatus` explicitly distinguishes `SUCCESS`, `EMPTY_SEARCH` (no hits matching query/filters), `BACKEND_FAILURE` (store or network timeout), `INSUFFICIENT_BUDGET` (budget too small for even one chunk), `STALE_INDEX`, and `REVISION_UNAVAILABLE`. An empty search result does NOT equate to an unanswerable question; host applications decide whether to answer, abstain, or ask questions.
- **Input Sanitization**: Tool arguments cannot supply arbitrary filesystem paths (`/etc/`, `../`, `file://`), credentials/tokens (`Bearer ...`, private keys, API keys), or unrestricted fetch/SSRF URLs (e.g. `localhost`, `127.0.0.1`, cloud metadata `169.254.169.254`). Unsafe inputs raise `SecurityValidationError`.
- **Scope Authorization**: Tenant boundaries (`ScopeAuthorizer`) are enforced before retrieval and before source lookup. Mismatched scopes fail closed with `ScopeAuthorizationError`.
- **Permission Guards**: Read tools are enabled by default; management tools (`sync_collection`, `revoke_document`) are disabled unless `allow_management=True` is explicitly configured.

### Tested Starter Recipes

Runnable offline recipes with deterministic fake embeddings and memory storage, followed by production profiles:

- `examples/recipes/policy_knowledge.py`: Policy and compliance knowledge with page citations and immutable revision updates.
- `examples/recipes/customer_service_faq.py`: Customer service FAQs with product and locale filters and authoritative FAQ revocation.
- `examples/recipes/multi_collection_diversity.py`: Cross-collection search with source diversity balancing.

## Gotchas

- **Register embedding dimensions** with `register_dimension("my-model", 768)`. Unknown models warn and fall back to `FALLBACK_DIMENSION` (1536) — a silent mismatch surfaces later as `ValueError: embedding has N dimensions; expected M`.
- **All SQL identifiers must pass `validate_identifier`** (`^[A-Za-z_][A-Za-z0-9_]*$`) — table names, column names, filter keys, metadata keys. Identifiers are never quoted or escaped; anything else is rejected.
- **Store failures are swallowed** into `[]` (or `0` for `count`) unless `strict=True`, which raises `ragnor.types.RetrievalError`. Use strict mode in fail-fast pipelines and tests.
- **The pgvector store needs the `pgvector` extra** — `ModuleNotFoundError: No module named 'sqlalchemy'` means it is missing.
- **Advisory locks are PostgreSQL-only** — pass `lock_key=None` on other stacks.
- `PgVectorStore` requires `embedding_dim` and either a `session_factory` or `settings` carrying a `database_url`.
- **MCP management tools are opt-in**: `sync_collection` and `revoke_document` will raise `PermissionError` unless `allow_management=True` is passed to `KnowledgeBridge`.
- Collection names must be registered exactly; unknown names fail instead of falling back. Configure `allowed_scopes` and the physical `scope_field` on each collection. For authenticated MCP, bind `client_id` in trusted server configuration or supply `client_id_provider`; remote SDK handlers also read the authenticated access-token context. See `docs/tool-bridge.md`.
- Sync host adapters must implement `SyncStateStore.fenced(token)` and atomic `ManifestStore.compare_and_swap_many(updates)`. Metadata-only sync copies stored vectors through `get_embeddings`, without re-embedding. See `docs/sync_coordination.md` for migration requirements.

## Pointers

- Full guide: `README.md` — key sections: Installation, Configuration (the `RagSettings` field table), Chunking, Embeddings, Vector stores, Hybrid retrieval, Ingestion, Job outbox and drainer, Evaluation, Extending ragnor: provider registries, Host integration checklist, Security model, Troubleshooting, Design invariants.
- `examples/host_integration.py` — complete runnable host-integration template (domain settings, a registered embedder provider, a registered vector backend, pipeline, retriever, evaluation harness); runs fully offline: `python examples/host_integration.py`.
- `examples/minimal_app.py` — zero-host-code FastAPI app (in-memory store, fake embedder unless `OPENAI_API_KEY` is set): `python examples/minimal_app.py`, then `curl "http://127.0.0.1:8000/search?query=refund"`.
- `examples/recipes/` — starter framework recipes: `policy_knowledge.py`, `customer_service_faq.py`, `multi_collection_diversity.py`.
