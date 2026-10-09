---
name: ragit
description: >-
  Guide for integrating the ragit domain-agnostic RAG toolkit: embeddings,
  chunking, vector stores, hybrid retrieval, ingestion, background jobs, and
  evaluation. Use when building RAG features, wiring a host application to
  ragit, or choosing/adding a vector backend (memory, pgvector, qdrant,
  chroma, faiss).
---

# ragit Integration Guide

ragit is a RAG _subsystem_, not an application: chunking, embeddings, vector stores, hybrid retrieval, idempotent ingestion, background jobs, and evaluation. Every host-specific dependency (settings, session factory, version/job tables) is injected through protocols and case-insensitive provider registries — ragit never imports the host.

## 1. Install

**The PyPI name `ragit` is FREE** (no distribution exists) but the package is **not yet published there** — install from GitHub, never from PyPI. The PyPI names `ragit` (an ingest4llm-based kit by HSPK) and `raglib` (an openai-agents-based Spanish RAG library by Harrison Chacon) are taken by unrelated third-party packages, and the GitHub repository is still named `ragit` while the Python package is `ragit`.

```bash
pip install git+https://github.com/eslam-aboutaleb/ragit.git

# With an extra
pip install "ragit[pgvector] @ git+https://github.com/eslam-aboutaleb/ragit.git"
```

As a uv/poetry git dependency in `pyproject.toml`:

```toml
ragit @ git+https://github.com/eslam-aboutaleb/ragit.git
```

Extras: `[pgvector]` (sqlalchemy[asyncio]), `[qdrant]` (qdrant-client), `[chroma]` (chromadb), `[faiss]` (faiss-cpu), `[all]`, `[test]` (pytest, pytest-asyncio, aiosqlite, sqlalchemy[asyncio]). Requires **Python >= 3.11**. Core dependencies are only `litellm` and `pydantic-settings` — no database driver is imported until a backend is selected.

## 2. Core concepts

**Protocol seams** (host implements/injects, ragit consumes): `RagSettings` (settings fields; `@runtime_checkable` — validate with `isinstance(settings, RagSettings)`), `SessionProvider` (async session factory), `VersionStore` (`find_active`, `close_active`, `create_version`, `delete_chunks`), `EmbeddingJobStore` (outbox storage), `DocumentSource` (`async load() -> list[RawDocument]`), `EmbeddingFunction` (async `__call__`, `embed_query`, `embed_documents`), `Chunker` (`chunk(text) -> list[Chunk]` with `version`/`config`), `VectorStore` (ABC: `hybrid_search`, `count`, `upsert`).

**Registries** — `embedding_registry`, `vector_store_registry`, `chunker_registry`: case-insensitive name lookup; unknown names raise `KeyError` listing registered providers.

**Two abstract factories**: `SQLFactory` (every SQL operation — sessions, queries, mutations, advisory locking; built-in `SQLAlchemyFactory`) and `VectorStoreFactory` (creates stores; built-in `RegistryVectorStoreFactory` reads `settings.vector_store_provider`).

**Design invariants** (pinned by tests): ragit never imports the host; no module-level singletons (host creates every registry/engine/session factory per process); idempotent ingestion via snapshot comparison; outbox atomicity (job row committed in the same transaction as the row it refers to); store failures swallowed unless `strict=True`; all SQL identifiers pass `validate_identifier` (`^[A-Za-z_][A-Za-z0-9_]*$`) and all values are bound parameters.

## 3. Choose a vector backend

Selected via `settings.vector_store_provider`. **Default is `"memory"`** (not pgvector). Backend modules load lazily — driver dependencies stay optional until a backend is actually selected.

| Provider     | Class                 | Use                        | Notes                                                                                               |
| ------------ | --------------------- | -------------------------- | --------------------------------------------------------------------------------------------------- |
| `"memory"`   | `InMemoryVectorStore` | dev/test, zero-dep         | brute-force L2 + token-overlap keyword approximation; nothing persisted                             |
| `"pgvector"` | `PgVectorStore`       | PostgreSQL production      | real pgvector L2 + `ts_rank` full-text search; needs `[pgvector]` extra                             |
| `"qdrant"`   | `QdrantStore`         | Qdrant server              | native vector search + in-process token-overlap keyword; `url`/`api_key` or `settings.database_url` |
| `"chroma"`   | `ChromaStore`         | embedded/persistent/server | `path=` for persistent, else ephemeral; sync client wrapped in `asyncio.to_thread`                  |
| `"faiss"`    | `FaissStore`          | in-process exact L2        | `IndexIDMap` over `IndexFlatL2`; nothing persisted                                                  |

```python
from ragit.stores import get_vector_store

store = get_vector_store(
    table_name="policy_chunks",
    embedding_dim=1536,
    settings=settings,
    embedding_fn=embedder,   # auto-embed queries
)
```

Session ownership rule: when the caller supplies a `session`, the caller owns the transaction (commit/rollback); when omitted, the store opens its own session and commits.

## 4. Wire ingestion

```python
from ragit.chunking import MarkdownSectionChunker, SlidingWindowChunker
from ragit.ingestion import FileDocumentSource, IngestionPipeline

pipeline = IngestionPipeline(
    store=store, embedder=embedder, chunker=chunker,
    session_factory=my_session_factory, settings=settings,
)

# lock_key: PostgreSQL advisory lock serializes concurrent ingestion
count = await pipeline.run(
    FileDocumentSource("docs/policy.md"), version_store,
    lock_key="docs/policy.md",
)
```

- **Idempotent**: the pipeline builds an `IngestionSnapshot` (source hash, embedding model/dimension, chunker version/tunables); if it matches the active version's snapshot, ingestion is skipped — never re-embedded.
- **Version lifecycle** on change: close active version → delete retired version's chunks (chunk ids are position-based) → create new version row → embed and upsert.
- **Abort safety**: if chunking yields no chunks, the pipeline aborts before touching the version table.
- **Advisory locking** (`lock_key`) is PostgreSQL-only (`pg_advisory_lock(hashtext(key))`); on other stacks pass `lock_key=None` and serialize ingestion in your own infrastructure.
- Host implements `VersionStore`; all methods receive the pipeline's session, so every mutation joins the ingestion transaction.

## 5. Wire retrieval

```python
from ragit.retrieval import HybridRetriever, RetrieverSpec, as_dicts

retriever = HybridRetriever(
    spec=RetrieverSpec(table_name="policy_chunks", metadata_fields=["section", "source"]),
    settings=settings,
    session_factory=my_session_factory,
)

results = await retriever.retrieve(
    "What is the refund policy?",
    n_results=5,
    distance_threshold=0.5,  # defaults to settings.rag_distance_threshold
    owner_id="user-123",     # any equality filter, applied to both stages
)
rows = as_dicts(results)
```

Two-stage search: dense vector search under the distance threshold + sparse keyword search, each ranked and truncated to `n_results`, fused with Reciprocal Rank Fusion: `score = 1/(60 + vector_rank) + 1/(60 + keyword_rank)`. A document found by both stages outranks a single-stage hit. `RetrieverSpec` validates every SQL identifier at construction; `embedding_dim` defaults to the dimension registered for `settings.embedding_model`.

## 6. Background jobs

```python
from ragit.jobs import enqueue_job, process_pending_jobs, reclaim_stale_jobs
from ragit.jobs.drainer import drain_forever, main

# Outbox: with a session the job row is written in the SAME transaction
await enqueue_job(store, {"text": "...", "metadata": {...}}, session=my_session)

processed = await process_pending_jobs(store, processor, limit=10, worker_id="worker-1")
await reclaim_stale_jobs(store, stale_after_seconds=300.0)

# Programmatic drainer (runs until the stop event) or console entry point
await drain_forever(store, processor, interval=5.0, batch_size=10,
                    worker_id="worker-1", stop=stop_event)
main(store, processor, interval=5.0, batch_size=10, worker_id="worker-1")
```

Claiming uses `FOR UPDATE SKIP LOCKED` (concurrent drainers scale horizontally without coordination). Failed jobs retry with backoff (1m, 5m, 30m); jobs exhausting `MAX_ATTEMPTS` (4) move to terminal `dead_letter`. `EmbeddingJobStore` and `JobProcessor` are host-implemented protocols.

## 7. Evaluation

```python
from ragit.evaluation import LabeledQuery, RagEvaluationHarness, run_evaluation, EvalConfig

harness = RagEvaluationHarness([
    LabeledQuery(query="refund policy?", expected_chunk_ids={"chunk_0"}),
])
metrics = await harness.evaluate(retrieval_fn)  # async (query: str) -> list[result dicts]
# metrics: recall_at_k, precision_at_k, mrr, latency, errors

result = await run_evaluation(
    EvalConfig(mode="end-to-end", judge="heuristic", n_results=5),
    retrieval_fn, dataset, settings=settings,
)
```

Answer-quality judges: `evaluate_answer_heuristic` (free, deterministic, token-overlap; pass `domain_terms` to boost completeness) vs `evaluate_answer_llm` (LiteLLM calls; **falls back to the heuristic on any failure**). `run_evaluation` requires injected `retrieval_fn` and `dataset` (raises `ValueError` otherwise) — ragit never imports a host retriever. Modes: `retrieval-only`, or `end-to-end` (scores the joined retrieved context against the gold answer; the runner does not generate answers).

## 8. Register custom providers

**Embedding provider** (selected via `settings.embedding_provider`):

```python
from ragit.embeddings import embedding_registry
from ragit.embeddings.base import EmbeddingFunction

@embedding_registry.register("my-provider")
class MyEmbeddingProvider:
    def __init__(self, settings) -> None:
        self._settings = settings

    def create(self) -> EmbeddingFunction:
        return MyEmbedder(self._settings.openai_api_key)
```

**Custom vector backend** (selected via `settings.vector_store_provider`): subclass the `VectorStore` ABC. The constructor **must accept the registry factory kwargs** (`table_name`, `id_field`, `embedding_dim`, `session_factory`, `settings`, `embedding_fn`) because `get_vector_store` instantiates it with them:

```python
from ragit.stores import VectorStore, vector_store_registry

@vector_store_registry.register("my-backend")
class MyVectorStore(VectorStore):
    def __init__(self, table_name="documents", id_field="id", embedding_dim=None,
                 session_factory=None, settings=None, embedding_fn=None):
        ...

    async def hybrid_search(self, query, embedding, n_results, threshold,
                            text_field="text", metadata_fields=None,
                            session=None, strict=False, **filters): ...
    async def count(self, session=None, strict=False, **filters) -> int: ...
    async def upsert(self, documents, session=None, text_field="text",
                     embedding_field="embedding", id_field=None, extra_fields=None): ...
```

## 9. Gotchas

- **Register embedding dimensions** with `register_dimension("my-model", 768)`. Unknown models warn and fall back to `FALLBACK_DIMENSION` (1536) — a silent mismatch surfaces later as `ValueError: embedding has N dimensions; expected M`.
- **All SQL identifiers must pass `validate_identifier`** (`^[A-Za-z_][A-Za-z0-9_]*$`) — table names, column names, filter keys, metadata keys. Identifiers are never quoted or escaped; anything else is rejected.
- **Store failures are swallowed** into `[]` (or `0` for `count`) unless `strict=True`, which raises `ragit.types.RetrievalError`. Use strict mode in fail-fast pipelines and tests.
- **The pgvector store needs the `pgvector` extra** — `ModuleNotFoundError: No module named 'sqlalchemy'` means it is missing.
- **Advisory locks are PostgreSQL-only** — pass `lock_key=None` on other stacks.
- `PgVectorStore` requires `embedding_dim` and either a `session_factory` or `settings` carrying a `database_url`.

## 10. Pointers

- Full guide: `README.md` — key sections: Installation, Configuration (the `RagSettings` field table), Chunking, Embeddings, Vector stores, Hybrid retrieval, Ingestion, Job outbox and drainer, Evaluation, Extending ragit: provider registries, Host integration checklist, Security model, Troubleshooting, Design invariants.
- `examples/host_integration.py` — complete runnable host-integration template (domain settings, a registered embedder provider, a registered vector backend, pipeline, retriever, evaluation harness); runs fully offline: `python examples/host_integration.py`.
- `examples/minimal_app.py` — zero-host-code FastAPI app (in-memory store, fake embedder unless `OPENAI_API_KEY` is set): `python examples/minimal_app.py`, then `curl "http://127.0.0.1:8000/search?query=refund"`.
