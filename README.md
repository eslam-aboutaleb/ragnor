# Ragnor

[![CI](https://github.com/eslam-aboutaleb/ragnor/actions/workflows/ci.yml/badge.svg)](https://github.com/eslam-aboutaleb/ragnor/actions/workflows/ci.yml)

**Domain-agnostic RAG toolkit** — chunking, embeddings, vector stores, hybrid retrieval, idempotent ingestion, background jobs, and evaluation.

ragnor is a RAG _subsystem_, not an application. so **any** host application can plug its own domain bindings into the same machinery. It ships no application code: every host-specific dependency (settings, session factory, version tables, job tables, datasets) is injected through protocols and case-insensitive provider registries.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Your first index](#your-first-index)
- [Configuration](#configuration)
- [Chunking](#chunking)
- [Embeddings](#embeddings)
- [Vector stores](#vector-stores)
- [Vector-store guide](docs/vector-stores.md)
- [Sync coordination and adapter requirements](docs/sync_coordination.md)
- [Knowledge bridge, evidence budgets, and MCP identity](docs/tool-bridge.md)
- [Hybrid retrieval](#hybrid-retrieval)
- [Ingestion](#ingestion)
- [Job outbox and drainer](#job-outbox-and-drainer)
- [Evaluation](#evaluation)
- [Extending ragnor: provider registries](#extending-ragnor-provider-registries)
- [Host integration checklist](#host-integration-checklist)
- [Security model](#security-model)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)
- [Examples](#examples)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Design invariants](#design-invariants)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Area                 | What you get                                                                                                                                                                                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chunking**         | Sliding-window chunker with configurable overlap; Markdown section-aware chunker that preserves `##` boundaries                                                                                                                                                |
| **Embeddings**       | LiteLLM-backed embedder; dimension registry with warn-and-fallback; `EmbeddingFunction` protocol                                                                                                                                                               |
| **Vector stores**    | Any vector database behind one `VectorStore` ABC — five built in (`memory`, `pgvector`, `qdrant`, `chroma`, `faiss`), all selectable via `settings.vector_store_provider` and loaded lazily so driver deps stay optional; bring your own (Milvus, Weaviate, …) |
| **Hybrid retrieval** | Dense vector similarity + sparse keyword search, fused with Reciprocal Rank Fusion (RRF)                                                                                                                                                                       |
| **Ingestion**        | Content-hash snapshot comparison (idempotent re-ingest), versioned chunks, optional advisory-lock serialization                                                                                                                                                |
| **Jobs**             | Outbox pattern with atomic claiming, retry/backoff, dead-letter, stale-job reclaim                                                                                                                                                           |
| **Evaluation**       | Recall@K / Precision@K / MRR retrieval metrics; heuristic and LLM-as-judge answer scoring                                                                                                                                                                      |
| **Extensibility**    | Case-insensitive provider registries for embedders, chunkers, and vector stores                                                                                                                                                                                |
| **Security**         | Strict SQL-identifier allowlist; all values bound as parameters                                                                                                                                                                                                |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Host application                         │
│  (settings, session factory, version/job tables, datasets)   │
└───────────────┬──────────────────────────────────┬───────────┘
                │ injects                          │ injects
                ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│                          ragnor                              │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ chunking │  │ embeddings│  │  stores  │  │ retrieval │  │
│  │ sliding  │  │ litellm   │  │ memory   │  │ hybrid    │  │
│  │ markdown │  │ dimensions│  │ pgvector │  │ RRF merge │  │
│  │          │  │           │  │ qdrant   │  │           │  │
│  │          │  │           │  │ chroma   │  │           │  │
│  │          │  │           │  │ faiss    │  │           │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ingestion │  │   jobs    │  │evaluation│  │  config   │  │
│  │ pipeline │  │  outbox   │  │ metrics  │  │ protocols │  │
│  │ version  │  │  drainer  │  │ judges   │  │ registry  │  │
│  │ locking  │  │  reclaim  │  │ runner   │  │ validation│  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
└──────────────────────────────────────────────────────────────┘
                │ uses                          │ uses
                ▼                              ▼
        ┌───────────────┐             ┌───────────────┐
        │ any vector DB │             │   LiteLLM /   │
        │ memory        │             │  any provider │
        │ pgvector      │             │               │
        │ qdrant        │             │               │
        │ chroma        │             │               │
        │ faiss         │             │               │
        └───────────────┘             └───────────────┘
```

**How the pieces fit.** A document enters through a `DocumentSource` and flows through the ingestion pipeline: the pipeline hashes the content, compares it against the active version's snapshot, chunks the text, embeds the chunks, validates the vectors, and upserts them into a vector store — inside one transaction, serialized by an advisory lock when a lock key is supplied. At query time, `HybridRetriever` embeds the query, delegates the two-stage search (vector similarity + keyword search) to the store, and fuses the results with RRF. Embedding work can be deferred to a background drainer through the job outbox. Finally, the evaluation harness scores retrieval quality against a curated test set. Every arrow in that flow is a protocol seam: the host injects settings, sessions, version storage, job storage, and datasets, and ragnor never imports the host in return.

**Two abstract factories.** Every SQL operation ragnor performs — sessions, queries, mutations, and advisory locking — goes through the `SQLFactory` interface, implemented by `SQLAlchemyFactory` (any SQLAlchemy dialect). Vector stores are created through the `VectorStoreFactory` interface, implemented by `RegistryVectorStoreFactory` (registry-based selection, including the zero-SQL in-memory store). Hosts subclass either factory to target another stack.

**Key design decisions.** ragnor depends on the host through protocols and never imports it: settings via `RagSettings`, sessions via `SessionProvider`, version and job storage via `VersionStore` / `EmbeddingJobStore`. Providers register under names and are looked up case-insensitively, so swapping a provider never edits ragnor. `Chunk`, `Document`, and `SearchResult` are the canonical value types, with the dict-based store API working alongside them. There are no module-level singletons — the host creates every registry, engine, and session factory per process.

---

## Installation

Upgrading from an earlier release? See the [Ragnor migration guide](docs/migration-ragnor.md).

ragnor requires **Python ≥ 3.11**. The core install has **zero third-party dependencies** — every capability (LiteLLM embeddings, vector backends) is an optional extra, imported lazily only when used.

The distribution is published on PyPI as **`ragnor`**:

```bash
# Install from PyPI
pip install ragnor

# With a backend extra
pip install "ragnor[pgvector]"
```

As a uv/poetry dependency in `pyproject.toml`:

```toml
ragnor
# or with an extra:
ragnor[pgvector]
```

Local development install from a checkout:

```bash
# From the repository root — the core install has ZERO third-party dependencies
pip install -e .

# With the LiteLLM integration (embeddings provider and LLM judge)
pip install -e ".[litellm]"  # litellm>=1.84

# With a specific vector backend (each driver is optional)
pip install -e ".[pgvector]"   # PostgreSQL: sqlalchemy[asyncio]
pip install -e ".[qdrant]"     # Qdrant: qdrant-client
pip install -e ".[chroma]"     # Chroma: chromadb
pip install -e ".[faiss]"      # FAISS: faiss-cpu

# Every backend at once
pip install -e ".[all]"

# With the test extras
pip install -e ".[test]"

# For the example app
pip install -e . fastapi uvicorn
```

The core install has **zero third-party packages** — `import ragnor.*` never requires litellm, a database driver, or a vector-DB client. Every capability ships as an optional extra and is imported lazily, only when actually used:

| Extra              | Adds                                          | Notes                                                              |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| `litellm`          | `litellm>=1.84`                               | LiteLLM embeddings provider and the LLM judge                      |
| `pgvector`         | `sqlalchemy[asyncio]>=2.0.30`                 | PostgreSQL pgvector store and the advisory lock                    |
| `qdrant`           | `qdrant-client>=1.9`                          | Qdrant vector store                                                |
| `chroma`           | `chromadb>=0.5`                               | Chroma vector store                                                |
| `faiss`            | `faiss-cpu>=1.8`                              | FAISS vector store                                                 |
| `local-embeddings` | *(reserved — currently empty)*                | Does not install a local-model provider |
| `test`             | `pytest`, `pytest-asyncio`, `aiosqlite`, `sqlalchemy[asyncio]`, `numpy`, `litellm` | Full test suite, including the LiteLLM integration check |
| `all`              | `ragnor[litellm,pgvector,qdrant,chroma,faiss]` | Every shipped capability at once                           |

The `pdf` and `web` extras enable PDF text extraction and HTTP document loading.

---

## Quick start

The high-level API needs three lines: build a `Ragnor` with `RagnorBuilder.local` (fully offline defaults — in-memory store, no database, no API key), ingest documents, and search. ragnor ships no local embedder in core, so you inject one; the example below uses a deterministic fake (the runnable version lives in `examples/local_quickstart.py`).

```python
from ragnor.builder import RagnorBuilder
from ragnor.types import Document

rag = await RagnorBuilder.local(embedder=my_embedder).build()
await rag.ingest([Document(id="policy", text="The refund policy covers water damage.")])
results = await rag.search("refund policy")
```

Every component is overridable (`with_store` / `with_embedder` / `with_chunker` / `with_version_store` / `with_settings`), and the low-level seams stay reachable through `rag.store`, `rag.pipeline`, and `rag.retriever`. The builder wires the existing `IngestionPipeline` and `HybridRetriever` unchanged — idempotent ingestion, snapshot comparison, and version replacement (retired chunks deleted through the store's `delete` capability) all behave exactly as in the low-level API.

Details the three lines hide: `build()` raises `ValueError` without an embedder (ragnor ships no local embedder in core); `embedding_dim=` is auto-registered for the embedding model via `register_dimension`; `rag.ingest(..., purge_on_empty=True)` treats an empty document list as authoritative — the active version is retired and its chunks deleted from the search index; `rag.count(**filters)` counts documents through the store; and `rag.search` defaults to `n_results=5`. The defaults are `SlidingWindowChunker(chunk_size=600, overlap=100)`, `BuilderSettings(embedding_model="local", rag_distance_threshold=1.3)` (a plain dataclass satisfying the `RagSettings` protocol — no pydantic-settings in core), and `InMemoryVersionStore` wired to the store; `BuilderSettings` and `InMemoryVersionStore` are public names, so hosts can construct or subclass them directly.

---

## Your first index

This walkthrough traces one document through the whole subsystem using the offline in-memory path — no database, no API key. The complete runnable version (with the fake embedder, null session factory, and in-memory version store) lives in `examples/host_integration.py`.

1. **Load.** A `DocumentSource` yields `RawDocument` objects; the built-in `FileDocumentSource` reads one file (a missing file is a no-op with a warning).
2. **Chunk.** The chunker splits the text — `MarkdownSectionChunker` splits on `##` headers and delegates each section to an inner chunker, stamping every chunk with its section and a position-based `chunk_id`.
3. **Snapshot.** The pipeline builds an `IngestionSnapshot` (source hash, embedding model and dimension, chunker version and tunables); if the active version carries the same snapshot, ingestion is skipped.
4. **Embed and upsert.** Changed content is chunked, embedded, validated against the expected dimension, and upserted into the store, stamped with the new version id.
5. **Retrieve.** `HybridRetriever` embeds the query and runs the two-stage hybrid search, returning `SearchResult` dataclasses.
6. **Evaluate.** A curated `LabeledQuery` test set gates deployment through `RagEvaluationHarness`.

The wiring looks like this:

```python
settings = MySettings()  # any object with the RagSettings fields
store = InMemoryVectorStore()
embedder = get_embedding_function(settings)
chunker = MarkdownSectionChunker(SlidingWindowChunker())
pipeline = IngestionPipeline(
    store=store,
    embedder=embedder,
    chunker=chunker,
    session_factory=null_session_factory,
    settings=settings,
)
count = await pipeline.run(FileDocumentSource("docs/policy.md"), version_store)

retriever = HybridRetriever(
    spec=RetrieverSpec(table_name="documents", metadata_fields=["chunk_id", "section"]),
    settings=settings,
    session_factory=null_session_factory,
)
results = await retriever.retrieve("What is the refund policy?")
```

Run `python examples/host_integration.py` for the full offline demo: it ingests the bundled document, re-ingests it (demonstrating the idempotent skip), retrieves, and evaluates — all in process.

---

## Configuration

ragnor reads its configuration through the **`RagSettings` protocol** — your existing pydantic `Settings` class already works when it exposes these fields:

| Field                              | Type    | Used by                                                                                                                |
| ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `embedding_model`                  | `str`   | Embedder model name; dimension lookup                                                                                  |
| `rag_distance_threshold`           | `float` | Default retrieval distance cutoff                                                                                      |
| `vector_store_provider`            | `str`   | Store provider name (default `"memory"`; also `pgvector`, `qdrant`, `chroma`, `faiss`, or your own registered backend) |
| `openai_api_key`                   | `str`   | LiteLLM provider credentials                                                                                           |
| `openai_api_base`                  | `str`   | LiteLLM base URL override                                                                                              |
| `embedding_drain_interval_seconds` | `float` | Drainer idle interval                                                                                                  |
| `embedding_drain_batch_size`       | `int`   | Drainer batch size                                                                                                     |
| `job_stale_after_seconds`          | `float` | Stale-job reclaim threshold                                                                                            |
| `worker_id`                        | `str`   | Drainer identity stamped into `locked_by`                                                                              |
| `ingest_on_startup`                | `bool`  | Host startup ingestion flag                                                                                            |

Two fields are optional and read via `getattr`: `embedding_provider` (default `"litellm"`, selects the embedder provider) and `llm_model` (read by the LLM judge). A third optional field, `rag_distance_metric` (default `"l2"`, selects the vector-store distance metric — see [Distance metrics](docs/vector-stores.md#distance-metrics)), is also read via `getattr`, so a settings object keeps satisfying the protocol whether or not it declares it. The protocol is `@runtime_checkable`, so host applications validate their settings at start-up with a plain `isinstance(settings, RagSettings)`. A `SettingsProvider` protocol (any object with a `get_settings()` method) is also available for accessor-style hosts.

---

## Chunking

Chunkers split raw text into `Chunk` objects. Every chunker reports a `version` tag and a `config` dict, which feed the ingestion snapshot comparison that makes re-ingestion idempotent.

### Sliding-window chunker

Splits text into overlapping chunks by **word count**: each chunk holds `chunk_size` words and consecutive chunks overlap by `overlap` words, preserving context across chunk boundaries. Use it for plain text with no structure to respect.

```python
from ragnor.chunking import SlidingWindowChunker, sliding_window_chunk

chunks = sliding_window_chunk(text, chunk_size=600, overlap=100)  # functional

chunker = SlidingWindowChunker(chunk_size=600, overlap=100)  # protocol form
chunks = chunker.chunk(text)  # -> list[Chunk] with uuid ids
chunker.version  # "sliding-window-v1"
```

`chunk_size >= 1` and `0 <= overlap < chunk_size`; otherwise `ValueError`. Empty or whitespace-only text yields `[]`.

### Markdown section chunker

Splits a Markdown document on `##` headers to preserve semantic boundaries, reconstructs each section with its H2 header (so the title leads the chunk text), then delegates to an inner chunker. The two chunkers **compose** — Markdown handles structure, the inner chunker handles sizing:

```python
chunker = MarkdownSectionChunker(
    inner=SlidingWindowChunker(),
    source_name="policy.md",
)
chunks = chunker.chunk(markdown_text)
```

Every chunk's metadata carries the frozen key set `{chunk_id, section, source, chunk_index, sub_chunk_index}` — `chunk_id` is position-based and unique per document, which is why ingestion deletes a retired version's chunks before inserting the new ones. A preamble without an H1 title defaults to the section name `"Introduction"`. The chunker reports `version = "section-aware-v1"` and delegates `config` to the inner chunker.

### Choosing a chunker

| Your document               | Choose                   | Why                                                                |
| --------------------------- | ------------------------ | ------------------------------------------------------------------ |
| Plain text, no structure    | `SlidingWindowChunker`   | Simple overlapping windows                                         |
| Markdown with `##` sections | `MarkdownSectionChunker` | Preserves section boundaries; delegates sizing to an inner chunker |

Both are registered in `chunker_registry` under `"sliding-window"` and `"markdown"` (lookups are case-insensitive), so hosts can select by name.

---

## Embeddings

### The EmbeddingFunction protocol

Every embedder implements the same seam: an async `__call__` for batches, plus `embed_query` and `embed_documents` conveniences.

```python
class EmbeddingFunction(Protocol):
    async def __call__(self, input: list[str]) -> list[list[float]]: ...
    async def embed_query(self, input: str | list[str]) -> list[float]: ...
    async def embed_documents(self, input: list[str]) -> list[list[float]]: ...
```

### LiteLLM provider (built-in)

The default provider awaits `litellm.aembedding`. Cancellation reaches native
async provider requests instead of leaving a synchronous request running in
an SDK worker thread. Qualify cancellation separately for providers that use
blocking fallbacks internally.

```python
from ragnor.embeddings import LitellmEmbeddingFunction

embedder = LitellmEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small",
)
vectors = await embedder(["hello", "world"])
```

Two optional keyword arguments remove the wrappers hosts used to write around this class, both defaulting to today's behavior: `cache_max_entries` (default `0`, off) turns on a bounded in-process LRU cache keyed by the model name and the text, so unchanged texts are never re-embedded; `max_batch_size` (default `None`) splits large batches into per-call chunks for provider-side input limits. When the cache is on, only the misses are embedded and the results are reassembled in the original input order.

For OpenAI/Azure paths, pass a configured native async client with `client=`
to control connection settings, retries and timeouts. The client is borrowed:
keep it open while the embedder is used and close it in the host lifecycle.

```python
embedder = LitellmEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small",
    cache_max_entries=4096,  # bounded LRU; 0 disables caching
    max_batch_size=512,  # per-call input cap; None sends one call
)
```

### Dimension registry

The dimension registry is the single source of truth mapping model names to vector dimensions, so changing the embedding model cannot silently desynchronize the dimension checks at insert time. Known models return their dimension; **unknown models warn and fall back to `FALLBACK_DIMENSION` (1536)** instead of crashing start-up — register your model if the fallback is wrong.

```python
from ragnor.embeddings import get_embedding_dimension, register_dimension

get_embedding_dimension("text-embedding-3-small")  # 1536
get_embedding_dimension("text-embedding-3-large")  # 3072
get_embedding_dimension("text-embedding-ada-002")  # 1536
register_dimension("my-model", 768)  # custom models
```

### Getting an embedder

`get_embedding_function(settings)` reads `settings.embedding_provider` (default `"litellm"`) and instantiates the registered provider. Construct `LitellmEmbeddingFunction` directly when you know the concrete class; use the factory when you want provider selection to be a settings change.

---

## Vector stores

ragnor ships five interchangeable stores behind the `VectorStore` interface. The core install stays database-free; install only the extra for the backend you select.

| Backend | Install extra | Best fit | Important trade-off |
| --- | --- | --- | --- |
| `memory` | None | Tests and local development | In-process and non-persistent; keyword search is token overlap |
| `pgvector` | `ragnor[pgvector]` | PostgreSQL-backed production retrieval | The only built-in backend with native PostgreSQL full-text ranking |
| `qdrant` | `ragnor[qdrant]` | Vector search through a Qdrant server | Keyword search is an in-process token-overlap fallback |
| `chroma` | `ragnor[chroma]` | Embedded or persistent Chroma collections | Keyword search is an in-process token-overlap fallback |
| `faiss` | `ragnor[faiss]` | An exact in-process vector index | No persistence or server; keyword search is token overlap |

The fallback keyword path in `memory`, `qdrant`, `chroma`, and `faiss` is not equivalent to PostgreSQL full-text search, and should not be used to tune production ranking thresholds. For backend setup examples, supported distance metrics, strict mode, and lifecycle operations, see the [vector-store guide](docs/vector-stores.md).

---

## Hybrid retrieval

`HybridRetriever` orchestrates retrieval: it resolves the embedder and store from settings, embeds the query, delegates the two-stage search to the store, and returns `SearchResult` dataclasses.

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
    distance_threshold=0.5,  # defaults to settings.rag_distance_threshold;
    # a distance under the store's metric (see
    # Distance metrics) -- metric-relative
    owner_id="user-123",  # any equality filter, applied to both stages
)
rows = as_dicts(results)  # back to dicts for legacy consumers
```

`RetrieverSpec` validates every SQL identifier at construction time; `embedding_dim` is optional and defaults to the dimension registered for `settings.embedding_model`.

**How the fusion works.** The store runs two independent stages — a dense vector search under the distance threshold and a sparse keyword search — ranks each, truncates both to `n_results`, and merges them with Reciprocal Rank Fusion: `score = 1/(60 + vector_rank) + 1/(60 + keyword_rank)`. A document found by both stages outranks one found by only one, and a query that fails one stage still returns results from the other. The pgvector backend implements the stages as SQL CTEs (`embedding <-> query` L2 distance and `tsvector @@ plainto_tsquery`); a custom backend implements the same contract with its native similarity and keyword search.

### Retrieval strategies

`ragnor.retrieval.strategies` composes the capability Protocols instead of the fused store: `DenseRetrieval(vector_index, embedder)`, `KeywordRetrieval(lexical_index)` (keyword-only, **no embedding dependency** — a keyword-only application needs no embedding-provider configuration at all), and `HybridRetrieval(vector_index, lexical_index, embedder, reranker=None)` — the two stages may be different objects (e.g. a pgvector `VectorIndex` plus an external lexical engine adapted to `LexicalIndex`). `reranker` is an optional `Reranker` protocol (`async rerank(query, results) -> list[dict]`; no built-in implementation ships — hosts inject a cross-encoder, an LLM reranker, ...). Dense and hybrid retrieval **require** a `threshold` kwarg (the distance cutoff) and raise `ValueError` without it. Every strategy applies the legacy swallow/raise contract: `strict=False` (the default) swallows failures into `[]`, `strict=True` raises `ragnor.types.RetrievalError`. Inject a strategy into `HybridRetriever(spec, settings, strategy=...)` to bypass the retriever's eager embedder/store construction — the builder wires its `HybridRetrieval` strategy this way, so the facade searches the same store instance the pipeline writes to.

---

## Ingestion

PDF connector ingestion defaults to strict extraction: unverified visual or
textless content rejects the PDF revision and preserves existing indexed content.
Traversal completeness and extraction quality are checked separately. See
[PDF policies and migration](docs/connectors.md#extraction-and-provenance) for legacy text views,
explicit partial extraction, and the limits of the built-in pypdf adapter.

For paginated directory, object-storage, HTTP/API, and website ingestion,
including extraction, completeness/deletion safety, and security requirements,
see the [source connector guide](docs/connectors.md).

### The pipeline

The pipeline runs in two phases. The **prepare phase** opens no transaction and takes no lock: **load source → hash the source manifest → compare snapshot → chunk → embed → validate**. The **critical section** is short: **re-check snapshot → retire old version → create version → upsert**, inside one transaction, serialized by the host transaction’s source row lock when a `lock_key` is supplied (see below); omit it and serialize ingestion in your own infrastructure.

The split keeps a slow remote embedding request from holding a transaction and an advisory lock open, and it lets an unchanged source skip before any embedding request is made.

```python
pipeline = IngestionPipeline(
    store=store,
    embedder=embedder,
    chunker=chunker,
    session_factory=my_session_factory,
    settings=settings,
)

# With a lock key: the pipeline owns the session, locks it, commits it
count = await pipeline.run(
    FileDocumentSource("docs/policy.md"),
    version_store,
    lock_key="docs/policy.md",
)

# With a caller-owned session: the pipeline never commits
count = await pipeline.run(source, version_store, session=my_session)
```

The constructor also accepts `version_id_key` and `source_id_key` (defaults `"version_id"` / `"source_id"`), the metadata keys the version and source ids are stamped under.

### Idempotency and snapshots

The source hash is a **manifest hash**: the pipeline records every document's `source_name`, `external_id`, and content hash in a canonical JSON manifest and hashes that. Document boundaries, source names, external ids, content, and document order all feed the hash — concatenating `"ab"` + `"c"` no longer hashes like `"a"` + `"bc"`, so a changed document identity or metadata forces a re-ingest.

**Upgrade note:** the manifest hash replaced the raw content-concatenation hash, so every stored snapshot mismatches once after upgrading — run ingestion once per source to re-ingest (a one-time cost).

The pipeline builds an `IngestionSnapshot` from the source hash, the embedding model and dimension, and the chunker version and tunables. When it matches the active version's stored snapshot, ingestion is skipped and the stored chunk count is returned — re-running ingestion on an unchanged document is a no-op, never re-embedding it. The comparison runs in the lock-free prepare phase, so an unchanged source never triggers an embedding request; the critical section re-checks the snapshot under the lock (double-checked locking), so a concurrent process that ingested the source meanwhile still wins.

**Abort safety.** If chunking yields no chunks, the pipeline aborts _before_ touching the version table — otherwise a version row carrying the current snapshot would make every later run skip the source forever.

**Version lifecycle.** On a changed source the pipeline:

1. Chunks, embeds, and validates outside any transaction
2. Closes the active version (sets its end timestamp)
3. Deletes the retired version's chunks (chunk ids are position-based, so they would violate the unique constraint)
4. Creates a new version row carrying the current snapshot
5. Upserts the new chunks, stamped with the version id

### Document sources

`FileDocumentSource` reads a single file (a missing file is a no-op with a warning). For anything else — S3, databases, APIs — implement the `DocumentSource` protocol (`async load() -> list[RawDocument]`).

**Source contract.** Implementations should **raise** on load failure (network error, permission denied, malformed response) and return `[]` **only** for an authoritative empty dataset — a source that genuinely has no documents. The pipeline treats an empty load as a no-op by default; pass `purge_on_empty=True` to `run` to treat it as authoritative: the active version is retired and its chunks deleted from the search index, so deleted content stops being searchable. This gives sync processes explicit deletion semantics — a source that empties out purges its index entries instead of leaving stale content searchable indefinitely.

### Advisory locking

`SQLAlchemyFactory.advisory_lock(session, key)` serializes ingestion of a source with a row in the host-owned `ragnor_ingestion_locks` table. The update holds a write lock for the caller's transaction. The host must apply the schema before ingestion. On failure the factory rolls back the transaction so the original error remains visible.

### Version storage (host-implemented)

The host implements the `VersionStore` protocol — `find_active`, `close_active`, `create_version`, `delete_chunks` — against its version tables. All methods receive the pipeline's session, so every mutation joins the ingestion transaction.

**External stores.** `delete_chunks` is the single host hook for retiring a version's chunks. Implement it with the portable delete capability: `await store.delete(filters={version_id_key: version_id})`. The delete-before-upsert order is load-bearing: chunk ids are position-based, so the new version's chunks would violate the unique constraint if the retired version's chunks stayed.

Hosts whose search index cannot share the version-table transaction should use the **staged-generation pattern**: write index mutations through the job outbox (`ragnor.jobs`) when they cannot join the ingestion transaction, and let the version-table transition (`close_active` + `create_version`) be the atomic active-version switch. A legacy backend wrapped in `LegacyHybridStoreAdapter` deliberately exposes no capabilities — hosts using legacy backends must retire chunks through their own mechanism (their version tables, or a native delete the backend provides outside the adapter). The pipeline itself only ever calls `version_store.delete_chunks`, so it stays agnostic to whether the underlying store implements the capability.

---

## Job outbox and drainer

The outbox pattern decouples row insertion from background processing: a job row is written in the **same transaction** as the row it refers to, and a drainer worker atomically claims pending jobs, processes them, and records the outcome. The built-in relational adapter uses conditional SQLAlchemy Core updates with fenced leases; the host supplies the engine, session factory, and migrations.

### Enqueue

```python
from ragnor.jobs import enqueue_job

# Inside your transaction: the job and the row land atomically
await enqueue_job(store, {"text": "...", "metadata": {...}}, session=my_session)

# Standalone: the store opens its own transaction and commits
await enqueue_job(store, {"text": "..."})
```

With a session, `enqueue_job` only flushes — the caller owns the commit, which is what makes the outbox atomic.

### Process

```python
from ragnor.jobs import process_pending_jobs, reclaim_stale_jobs

processed = await process_pending_jobs(store, processor, limit=10, worker_id="worker-1")
```

**Failure semantics:**

- A failed job is retried with backoff (`1m`, `5m`, `30m`)
- Jobs that exhaust `MAX_ATTEMPTS` (4) move to the terminal `dead_letter` state
- Claimed jobs are stamped `locked_at`/`locked_by`; a worker that dies mid-pass leaves a visible lock
- `reclaim_stale_jobs` resets abandoned `processing` rows (older than `stale_after_seconds`, or with a NULL lock timestamp) back to `pending`

### Drainer

```python
from ragnor.jobs.drainer import drain_forever, main

# Programmatic: runs until the stop event is set
await drain_forever(
    store, processor, interval=5.0, batch_size=10, worker_id="worker-1", stop=stop_event
)

# Console entry point: installs logging, handles SIGINT/SIGTERM
main(store, processor, interval=5.0, batch_size=10, worker_id="worker-1")
```

Multiple drainers run concurrently: atomic claims keep two workers from processing the same job. A failing pass is logged and retried on the next tick — a transient outage does not terminate the worker.

The job table itself is host-owned: `EmbeddingJobStore` (claim, mark, enqueue, reclaim) and `JobProcessor` (the per-job work) are protocols the host implements.

For the built-in SQLAlchemy Core adapter, see [host-owned SQL outbox](docs/sql-outbox.md)
for one-time database binding, migrations, supported transaction settings,
tenant scope, worker retries, and retention.

---

## Evaluation

### Retrieval metrics

`RagEvaluationHarness` runs a curated test set of `LabeledQuery` objects through any async retrieval function and aggregates Recall@K, Precision@K, MRR, average latency, and error counts.

```python
from ragnor.evaluation import LabeledQuery, RagEvaluationHarness

harness = RagEvaluationHarness(
    [
        LabeledQuery(query="refund policy?", expected_chunk_ids={"chunk_0"}),
    ]
)
metrics = await harness.evaluate(retrieval_fn)
# retrieval_fn: async (query: str) -> list[result dicts]
```

### Answer quality

Two strategies, both producing `AnswerScore` (faithfulness, relevance, completeness, conciseness, overall — all in `[0, 1]`, overall weighted `0.30·F + 0.25·R + 0.30·C + 0.15·Conc`):

| Strategy                    | Choose when                     | Trade-off                                                               |
| --------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| `evaluate_answer_heuristic` | CI, offline runs, no API budget | Deterministic and free; token-overlap based                             |
| `evaluate_answer_llm`       | Richer judgment in dev/staging  | Costs LiteLLM API calls; **falls back to the heuristic on any failure** |

The heuristic judge is domain-neutral by default: key facts are numbers, plus any host-supplied vocabulary passed via `domain_terms` (which boosts completeness). The LLM judge reads `settings.llm_model` and falls back to the heuristic when the setting is absent or the call fails.

### The runner

`run_evaluation(config, retrieval_fn, dataset, settings=None, answer_fn=None)` orchestrates a full run. It **requires** an injected `retrieval_fn` and `dataset` (it raises `ValueError` otherwise) — ragnor never imports a host retriever or dataset.

```python
result = await run_evaluation(
    EvalConfig(mode="end-to-end", judge="heuristic", n_results=5),
    retrieval_fn,  # async (query: str) -> list[result dicts]
    dataset,  # list[EvalSample], pre-filtered
    settings=settings,
    answer_fn=answer_fn,  # async (query: str, context: str) -> answer
)
print(result.retrieval_metrics)  # recall@K, precision@K, MRR, latency
print(result.answer_metrics)  # avg faithfulness/relevance/...
print(result.summary)  # human-readable report
```

**Modes.** Three modes, each measuring something different:

- `retrieval-only` — retrieval quality alone (Recall@K, Precision@K, MRR, latency).
- `context` — scores the **joined retrieved context** against the gold answer. This is the legacy behavior under an honest name: it measures whether the retrieved chunks *contain* the answer, not whether a model answers faithfully. A cheap upper-bound proxy when no answer generator is wired.
- `end-to-end` — also scores answer quality, but on a **real model output**: the injected `answer_fn` (an async `(query, context) -> answer` callable) generates the answer, and the judges receive the actual answer and the actual retrieved context as separate inputs. `end-to-end` **requires** `answer_fn` and raises `ValueError` naming the three modes otherwise — a silently wrong metric is worse than a loud failure. Per-sample results carry the retrieved `context`, the `generated_answer`, the `gold_answer`, and `chunks_found` separately.

`EvalSample.category` is a free-form string (hosts define their own taxonomy); `difficulty` is an `easy`/`medium`/`hard` enum. The CLI entry point (`ragnor.evaluation.runner:main`) exposes `--mode` (all three choices), `--judge`, `--n-results`, `--category`, `--difficulty`, and `--output`; hosts pass their own `retrieval_fn`, `dataset`, and optional `runner` callable to inject domain dependencies — including the `answer_fn` answer generator that `end-to-end` mode requires — behind the flags.

---

## Extending ragnor: provider registries

Providers register themselves in ragnor's registries — swap a provider without editing ragnor. A provider is a factory class that (1) accepts a `RagSettings` object in its constructor and (2) exposes a `create()` method returning the protocol instance:

```python
from ragnor.embeddings import embedding_registry
from ragnor.embeddings.base import EmbeddingFunction


@embedding_registry.register("my-provider")
class MyEmbeddingProvider:
    def __init__(self, settings) -> None:
        self._settings = settings

    def create(self) -> EmbeddingFunction:
        return MyEmbedder(self._settings.openai_api_key)


# Selected via settings.embedding_provider = "my-provider"
```

The same pattern applies to every registry:

| Registry                | Module                      | Selected by                      |
| ----------------------- | --------------------------- | -------------------------------- |
| `embedding_registry`    | `ragnor.embeddings.registry` | `settings.embedding_provider`    |
| `vector_store_registry` | `ragnor.stores.registry`     | `settings.vector_store_provider` |
| `chunker_registry`      | `ragnor.chunking.registry`   | explicit construction            |

All lookups are case-insensitive; unknown names raise `KeyError` listing the registered providers. For a vector backend, subclass the `VectorStore` ABC and make the constructor accept the registry factory kwargs (`table_name`, `id_field`, `embedding_dim`, `session_factory`, `settings`, `embedding_fn`), because `get_vector_store` instantiates it with them — this is the seam for targeting Milvus, Weaviate, or any other vector database not shipped in the box.

### Abstract factories

Two abstract factories sit above the registries. `SQLFactory` is the seam for queries, mutations, and source locking; inject a custom implementation via `sql_factory=` on `IngestionPipeline` and `PgVectorStore` to run ragnor on another stack. The host creates and injects its own session factory separately:

```python
from contextlib import asynccontextmanager

from ragnor.db.sql_factory import SQLFactory


class MySqlFactory(SQLFactory):
    async def fetch_all(self, session, statement, params=None): ...
    async def fetch_scalar(self, session, statement, params=None): ...
    async def execute(self, session, statement, params=None): ...

    @asynccontextmanager
    async def advisory_lock(self, session, key):
        yield  # your stack's locking primitive
```

`VectorStoreFactory` creates stores for any vector database — subclass it to customize backend construction, with `RegistryVectorStoreFactory` (the default) reading `settings.vector_store_provider` and instantiating the registered provider. Built-in backend modules are imported lazily, so their driver dependencies stay optional until the backend is actually selected.

---

## Host integration checklist

ragnor is host-agnostic by design — any domain application integrates by completing the checklist below. A complete runnable template lives in `examples/host_integration.py`: it wires settings, a registered embedder provider, a registered vector backend, the ingestion pipeline, the hybrid retriever, and the evaluation harness end to end, and runs fully offline (in-memory store + fake embedder).

1. **Define domain settings.** A settings class exposing the `RagSettings` fields (see the table in [Configuration](#configuration)). Validate at start-up with `isinstance(settings, RagSettings)`.
2. **Provide an embedder.** Either construct an `EmbeddingFunction` implementation directly and inject it into `IngestionPipeline`, or register a provider factory under a name (see [Extending ragnor](#extending-ragnor-provider-registries)) and select it via `settings.embedding_provider`.
3. **Choose the vector backend.** Use a built-in (`"memory"` for dev/test, `"pgvector"` for PostgreSQL, `"qdrant"`, `"chroma"`, or `"faiss"`) or bring your own for any other vector database: subclass the `VectorStore` ABC, register it, and select it via `settings.vector_store_provider`. The constructor must accept the registry factory kwargs because `get_vector_store` instantiates it with them.
4. **Register the embedding dimension.** Call `register_dimension("my-model", 768)` so insert-time dimension validation matches the model's real output; unknown models warn and fall back to 1536 instead of crashing start-up.
5. **Wire ingestion.** Construct `IngestionPipeline(store, embedder, chunker, session_factory, settings)` plus a host-implemented `VersionStore`.
6. **Wire retrieval.** Construct `HybridRetriever(spec=RetrieverSpec(table_name=..., metadata_fields=[...]), settings=settings, session_factory=...)`; retrieve with `await retriever.retrieve(query, n_results=5, **filters)`.
7. **Validate retrieval quality before serving.** Curate a `LabeledQuery` test set and run `RagEvaluationHarness` (Recall@K / Precision@K / MRR); gate deployment on the metrics.

The design invariants below are pinned by tests — see [Design invariants](#design-invariants) for the authoritative list.

---

## Security model

The built-in pgvector store builds raw SQL (the hybrid query needs CTEs, window functions, and pgvector operators impractical to express in the ORM); the `VectorStore` ABC itself is database-agnostic. Where raw SQL is used, the security model compensates:

1. **All user-supplied values are bound parameters.** They never appear in the SQL string, so they cannot inject SQL.
2. **All SQL identifiers pass a strict allowlist** (`^[A-Za-z_][A-Za-z0-9_]*$`) via `validate_identifier` before interpolation — table names, column names, filter keys, metadata keys. Identifiers are never quoted or escaped; anything else is rejected.
3. **Parameter namespacing.** Filter parameters are prefixed (`filter_<key>`), column parameters use `_col_<key>__<row>`, and extra fields use `_extra_<key>__<row>`, so a metadata key can never shadow a reserved bind parameter or a column value.
4. **Embedding validation.** Every embedding is checked for the expected dimension and finite values before it reaches the database (`validate_embedding`).

---

## Troubleshooting

**`KeyError: Unknown provider 'x'. Registered: ...`** — You misspelled a provider name in `settings.embedding_provider` or `settings.vector_store_provider`. The error message lists the registered names — for the built-in registries those are `litellm` (embeddings) and `memory`, `pgvector`, `qdrant`, `chroma`, `faiss` (stores). Lookups are case-insensitive, so check for typos, not case.

**`ValueError: embedding_dim is required`** — You constructed a `PgVectorStore` without `embedding_dim`. Pass the dimension configured for the active embedding model — `get_embedding_dimension(settings.embedding_model)` resolves it.

**`ValueError: embedding has N dimensions; expected M`** — A dimension mismatch at insert or query time. The usual cause is an unregistered embedding model that silently fell back to `FALLBACK_DIMENSION` (1536) while your backend produces a different size. Register the real dimension with `register_dimension()` and check the logs for the unknown-model warning.

**Warning: `No embedding dimension registered for model '...'`** — The model is not in the dimension registry, so ragnor assumed 1536. If that is wrong, call `register_dimension(model, dimension)` at start-up; otherwise the fallback is safe.

**`RuntimeError: PgVectorStore needs a session_factory`** — Pass the host-owned session factory.

**`RuntimeError: QdrantStore needs an explicit url or injected client`** — Pass `url` (and `api_key` for authenticated servers) or inject a host-owned `client`.

**`ModuleNotFoundError: No module named 'sqlalchemy'`** (or `'qdrant_client'` / `'chromadb'` / `'faiss'`) — A vector backend was used without installing its extra. Run `pip install -e ".[pgvector]"` (or `.[qdrant]`, `.[chroma]`, `.[faiss]`; `pip install -e ".[test]"` includes the pgvector dependency).

**Empty results (or `count` of 0) with no error** — Non-strict mode swallows store failures into `[]` / `0` by design. Pass `strict=True` to surface the underlying failure as a `RetrievalError` — the logged error line above the empty result names the real cause.

**Jobs stuck in `dead_letter`** — A job exhausted `MAX_ATTEMPTS` (4) and moved to the terminal `dead_letter` state. Inspect the dead-letter rows' `status_detail` for the recorded error, fix the cause, and re-enqueue.

**`InFailedSQLTransactionError` around advisory locks** — A failed statement left the session in an aborted transaction, and a later command on it raised this error. `advisory_lock` rolls back before unlocking precisely so the _original_ ingestion failure surfaces instead — look for the earlier log line naming the real error.

---

## Glossary

- **RRF (Reciprocal Rank Fusion)** — the merge formula `1/(60 + vector_rank) + 1/(60 + keyword_rank)` that combines the vector and keyword result rankings; a hit found by both stages outranks a single-stage hit.
- **Outbox** — the pattern of writing a job row in the same transaction as the row it refers to, so background work is triggered atomically with the data change.
- **Drainer** — the background worker that claims pending outbox jobs, processes them, and records the outcome.
- **Snapshot** — an `IngestionSnapshot` capturing everything that determines whether a stored index is current (source hash, embedding model and dimension, chunker version and tunables); comparing snapshots makes ingestion idempotent.
- **Strict mode** — opt-in `strict=True` behavior where store failures raise `RetrievalError` instead of being swallowed into empty results.
- **Atomic claim** — a storage adapter's concurrency-safe transition from pending to processing; the relational adapter uses conditional updates and fenced leases.
- **Source lock** — a host-database row lock used to serialize ingestion of the same source across instances and tasks.
- **SQL factory** — the `SQLFactory` interface for queries, mutations, and source locking; `SQLAlchemyFactory` is the built-in implementation, while the host owns session creation.
- **Vector store factory** — the `VectorStoreFactory` interface that creates vector stores; `RegistryVectorStoreFactory` is the default, selecting the backend from the provider registry.
- **tsvector / pgvector** — PostgreSQL extensions for full-text search (tokenized text matched with `tsquery`) and vector similarity (L2 distance via the `<->` operator), respectively.
- **retrieval_fn** — the injected async callable `(query: str) -> list[result dicts]` that evaluation runs against; the host supplies it, ragnor never imports it.

---

## Examples

`examples/local_quickstart.py` is the offline three-line quickstart for the high-level builder API — build, ingest, search with a deterministic fake embedder, fully in process (no network, no database, no API key):

```bash
python examples/local_quickstart.py
```

`examples/minimal_app.py` is a complete FastAPI app using ragnor with zero host code — in-memory store, fake embedder (unless `OPENAI_API_KEY` is set), no database:

```bash
pip install -e . fastapi uvicorn
python examples/minimal_app.py
curl "http://127.0.0.1:8000/search?query=refund"
```

The example also demonstrates the `VersionStore` and `SessionProvider` protocols with in-memory implementations (`MemoryVersionStore`, `_null_session_factory`) — a useful template for host adapters.

`examples/host_integration.py` is the full host-integration template for any domain — domain settings, a registered embedder provider, a registered vector backend, end-to-end wiring, and retrieval-quality validation. It runs offline (no database, no API key):

```bash
python examples/host_integration.py
```

---

## Testing

GitHub Actions runs the full suite on Python 3.11 and 3.12, checks Ruff linting and formatting, validates the built distributions, and tests optional vector-store extras in separate jobs.

```bash
# Install the test extras
pip install -e ".[test]"

# Run the full suite
python -m pytest -q

# Run a single module
python -m pytest tests/test_chunking.py -q
```

The suite is **512 passed, 7 skipped**. It covers:

| Module                                      | Test file(s)                                                                                                                                                                 |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chunking`                                  | `test_chunking.py`                                                                                                                                                           |
| `embeddings`                                | `test_embeddings.py`                                                                                                                                                         |
| `builder`                                   | `test_builder.py`                                                                                                                                                         |
| `stores` (memory)                           | `test_memory_store.py`, `test_vector_store_coverage.py`                                                                                                                      |
| `stores` (pgvector)                         | `test_pgvector_store.py`, `test_pgvector_store_coverage.py`, `test_pgvector_strict_mode.py`, `test_hybrid_search_projection.py`, `test_hybrid_search_session_injection.py`   |
| `stores` (qdrant, chroma, faiss)            | `test_vector_backends.py`, `test_qdrant_coverage.py`, `test_faiss_coverage.py`                                                                                                                                                    |
| `stores` (capability Protocols)             | `test_store_capabilities.py`                                                                                               |
| `retrieval`                                 | `test_retriever.py`, `test_retriever_strict.py`                                                                                                                                                          |
| `ingestion`                                 | `test_ingestion_pipeline.py`, `test_ingestion_sources.py`, `test_ingestion_versioning.py`, `test_ingestion_locking.py`                                                       |
| `jobs`                                      | `test_embedding_jobs.py`, `test_embedding_jobs_coverage.py`, `test_embedding_drain.py`, `test_job_reclaim.py`                                                                |
| `evaluation`                                | `test_rag_eval.py`, `test_evaluation_coverage.py`, `test_eval_runner_coverage.py`, `test_eval_runner_modes.py`, `test_eval_dataset_coverage.py`, `test_answer_evaluator_coverage.py`                      |
| `examples/host_integration.py`              | `test_host_integration.py`                                                                                                                                                   |
| domain-agnostic seams                       | `test_agnostic_seams.py` (parametrized over two synthetic domains: settings, providers, dimensions, chunkers, stores, retrieval, ingestion, evaluation, pgvector validation) |
| `registry`, `validation`, `types`, `config` | `test_registry.py`, `test_validation.py`, `test_types.py`, `test_config.py`                                                                                                  |
| `db`                                        | `test_session_provider.py`                                                                                                                                                   |
| factories (`db`, `stores`)                  | `test_factories.py`, `test_factory_coverage.py`                                                                                                                              |
| invariants                                  | `test_no_app_imports.py` (ragnor must never import the host `app` package), `test_dependency_hygiene.py` (zero-dependency invariant: subprocess-based litellm import blocking)                                                                                                    |

**Database-dependent tests** (pgvector strict mode, session injection) skip automatically unless `RAGNOR_TEST_DATABASE_URL` points at a PostgreSQL instance with pgvector installed — that is 3 of the 7 skipped tests in a default run (the other 4 need the `chroma` extra):

```bash
RAGNOR_TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/ragnor_test" \
    python -m pytest -q
```

**Test doubles** for the job subsystem live in `tests/job_doubles.py` (`FakeJobStore`, `FakeJobProcessor`).

---

### Continuous integration

[CI](.github/workflows/ci.yml) inherits Python quality, security analysis, and workflow validation from
[ci-workflows](https://github.com/eslam-aboutaleb/ci-workflows), pinned to a reviewed commit SHA.
The Python call explicitly uses pip and the package's `test` extra on Python 3.11 and 3.12.
It requires Ruff lint/format checks, at least 85% coverage of `src/ragnor`, and valid wheel/sdist metadata.

Ragnor's optional backend matrix also runs in CI. `CI Status Check` requires the shared jobs and
every backend job to succeed. SonarCloud runs through its dedicated workflow.
GitHub Actions Dependabot updates cover the shared workflow references; review a dependency update
and its resulting checks before adopting a new revision.

## Project structure

```
ragnor/
├── pyproject.toml            # packaging, pytest, ruff config
├── README.md                 # this file
├── examples/
│   ├── local_quickstart.py   # offline three-line quickstart (builder API)
│   ├── minimal_app.py        # zero-dependency FastAPI example
│   ├── host_integration.py   # host integration template (any domain)
│   └── sample_document.md    # bundled example document
├── src/ragnor/
│   ├── core/                # types, settings, exceptions, validation, registries
│   ├── client/              # asynchronous builder/facade and synchronous client
│   ├── types.py             # compatibility import for core.types
│   ├── config.py            # compatibility import for core.config
│   ├── builder.py           # compatibility import for client.builder
│   ├── chunking/             # sliding-window + Markdown section chunkers
│   ├── embeddings/           # LiteLLM function, dimension registry
│   ├── stores/               # VectorStore ABC, five built-in backends
│   │   ├── metrics.py        # vector distance metrics and normalization
│   │   ├── memory.py         # in-memory store (dev/test)
│   │   ├── pgvector.py       # PostgreSQL pgvector + tsvector store
│   │   ├── qdrant.py         # Qdrant store
│   │   ├── chroma.py         # Chroma store
│   │   ├── faiss.py          # FAISS store
│   │   ├── _scoring.py       # shared RRF merge, token overlap, L2 distance
│   │   ├── registry.py       # vector_store_registry, get_vector_store
│   │   └── store_factory.py  # VectorStoreFactory ABC + registry factory
│   ├── retrieval/            # HybridRetriever, RetrieverSpec
│   │   └── strategies.py     # DenseRetrieval, KeywordRetrieval, HybridRetrieval, Reranker
│   ├── ingestion/            # pipeline, versioning, locking, sources
│   ├── jobs/                 # outbox, drainer
│   ├── evaluation/           # metrics, judges, runner
│   └── db/                   # SessionProvider protocol, session factory
│       └── sql_factory.py    # SQLFactory ABC + SQLAlchemyFactory
└── tests/                    # test suite, no host dependencies
```

---

## Design invariants

These invariants are pinned by tests and must hold for any change:

1. **ragnor never imports the host application** — enforced by `test_no_app_imports.py`.
2. **No module-level singletons** — registries, engines, and session factories are created per process by the host.
3. **Chunk metadata keys are frozen** — the Markdown chunker's metadata set is exactly `{chunk_id, section, source, chunk_index, sub_chunk_index}`.
4. **Ingestion is idempotent** — an unchanged source is skipped via snapshot comparison, never re-embedded.
5. **The outbox is atomic** — a job row is committed in the same transaction as the row it refers to.
6. **Store failures are opt-in** — swallowed by default, raised as `RetrievalError` under `strict=True`.
7. **SQL identifiers are allowlisted** — every interpolated identifier passes `validate_identifier`; every value is a bound parameter.

---

## Contributing

1. Fork and branch from `main`.
2. Add tests for any new behavior (the suite runs without a database; pgvector-specific tests skip gracefully).
3. Run `python -m pytest -q` and `ruff check .` before pushing.
4. Keep the design invariants above intact — especially the no-host-imports rule.

### Pre-commit

Install the hooks to run linting, formatting, and whitespace fixes automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

The hooks run `ruff check`, `ruff format`, and the standard whitespace/EOF/YAML fixes defined in `.pre-commit-config.yaml`.

---

## License

Ragnor is source-available under the **Elastic License 2.0 (ELv2)**
(SPDX: `Elastic-2.0`), which is not an OSI-approved open-source license.
Commercial integration is permitted subject to the license terms, including
restrictions on offering substantial Ragnor functionality as a hosted or
managed service. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the
[licensing guide](docs/licensing.md) for terms and illustrative examples.
Previously distributed releases retain their original license terms.
