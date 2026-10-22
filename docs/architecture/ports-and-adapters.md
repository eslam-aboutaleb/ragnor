# Architecture: ports, adapters and dependency rules

**Status:** Accepted for SDK integration (issue #24). This document describes the existing SDK structure and incremental constraints; it does not assert that every provider currently has production-grade adapters.

## Context and decision (ADR-001)

Ragnor is a library embedded into a host application, not a framework that owns the host's API server or database. Preserve a **zero mandatory third-party dependency** core, host-controlled infrastructure, structural Python `Protocol` seams and an async-first public API.

The composition root (`ragnor.builder.RagnorBuilder` or the host application) selects concrete adapters. Application orchestration depends on protocols and dataclasses; it must not depend directly on vendor SDKs. Concrete infrastructure adapters depend *inward* on contracts, not the reverse.

```text
Host app / entrypoints (FastAPI, CLI, MCP, workers)
                         |
              Composition root / builder
                         |
         Application workflows / use cases
                         |
          Core dataclasses and Protocols
               ^                ^
               |                |
         Optional adapters / host implementations
       PostgreSQL / vector DBs / LiteLLM / OTel
```

Arrows toward core indicate source-code dependencies, **not** direction of data flow. Composition roots may import adapters lazily. Infrastructure modules are not automatically ports simply because they implement an interface.

## Port and adapter inventory

| Concern | Core port or contract | Existing implementation(s) | Integration rule |
|---|---|---|---|
| Source loading | `ingestion.sources.DocumentSource` | list/file and connector sources | Inject a source into ingestion |
| Extraction | extractor/connector interfaces | file/PDF/HTTP/object-storage integrations | Keep provider-specific libraries optional |
| Chunking | `chunking.base.Chunker` | sliding-window words, markdown sections | Inject strategy or resolve via registry |
| Token budgets | `evidence.tokenizer.Tokenizer` | approximate/exact adapter | Injection required for exact model limits |
| Embeddings | `embeddings.base.EmbeddingFunction` | LiteLLM adapter, host implementations | Inject with embedding profile |
| Retrieval | retrieval strategy/capability contracts | dense/keyword/hybrid, reranker | Explicit capability negotiation |
| Vector + document storage | `stores.base.VectorIndex`, `LexicalIndex`, `DocumentStore`, `VectorStore` | memory, pgvector, Qdrant, Chroma, FAISS | Adapters loaded only when requested |
| Version storage | `ingestion.pipeline.VersionStore` | in-memory, host adapters | Transaction-compatible versioning |
| Database sessions | `db.session.SessionProvider` + `db.sql_factory.SQLFactory` | host factory, SQLAlchemy | Host owns transaction if injected |
| Distributed sync | `sync.state.SyncStateStore`, manifest contracts | in-memory, host adapters | Durable leases/fencing required in production |
| Job execution | `jobs.outbox.EmbeddingJobStore`, `JobProcessor` | in-memory + host adapters | Protocol alone does NOT guarantee atomic claiming |
| Evidence assembly | `evidence.tokenizer.Tokenizer`, `EvidenceCompressor` | deterministic assembler, pluggable compressor | Preserve provenance and token budgets |
| Diagnostics | `telemetry.events.EventSink` | null/logging/OTel sinks | Opt-in, redact content and secrets |
| Scope/authorization | tools/bridge authorizer | supplied host authorizer | Fail closed on unsupported isolation |

Each concern must admit **a separately supplied concrete object or factory** without changing Ragnor source code. Registries are optional conveniences; avoid global mutable registration as the only path. Existing backwards-compatible interfaces remain supported.

## Rules enforced by automated architecture tests

1. Core port, domain type and use-case modules must not statically import a host `app`, FastAPI/Django or vendor libraries (`litellm`, `sqlalchemy`, `qdrant_client`, `chromadb`, `tiktoken`, etc.). Vendor imports belong to concrete adapters.
2. Core use cases cannot statically import vendor-specific concrete adapter modules (`ragnor.stores.pgvector`, `ragnor.stores.memory`, `ragnor.embeddings.litellm`, etc.). **`builder.py` is a composition root and explicitly exempt**, but its defaults must not be used implicitly as durable production guarantees.
3. No import-time backend connections, credential access or driver requirements in the public core.
4. Use small composable protocols. The host constructs/owns externally supplied session factories, stores, processors and event sinks unless ownership is explicitly transferred.
5. Backend capability absence must be reported rather than allowing a silent authorization, durability, transaction or model-limit downgrade.
6. Refactors should be incremental and backwards-compatible. Prefer new composition seams over replacing existing working components.

These checks do **not** prove DB atomicity or tenant isolation; those require adapter conformance and real integration tests in #26.

## Resource and transaction ownership

- **Core-only local builder:** temporary in-memory vector/version stores and a no-op session; suitable for demos, not durable production.
- **Injected resources:** remain host-owned. Ragnor must not close host pools by default.
- **Owned resources:** must have explicit async close/context-manager behavior, bounded cancellation and deterministic cleanup (see #21).
- **Caller-provided database session:** caller commits/rolls back. A pipeline-owned session commits its own transaction. Locks must be held on the same transaction that mutates versions and indexes.
- **Cross-storage consistency:** do not claim that PostgreSQL transactions encompass a separate external vector DB.
- **Sync and jobs:** fencing/atomic job claiming are adapter responsibilities with conformance tests; Protocol documentation is not proof.

## Extension recipe

```python
from ragnor.builder import RagnorBuilder
from ragnor.types import Document

# Your own implementation of EmbeddingFunction, plus a custom
# Chunker and VectorStore implementation when needed.
builder = RagnorBuilder.local(embedder=my_embedder).with_chunker(my_chunker)
builder = builder.with_store(my_store).with_version_store(my_version_store)
rag = await builder.build()
await rag.ingest([Document(id="one", text="Knowledge source")])
```

Replace the placeholders with host-provided implementations of the documented protocols. An external backend must implement the search capabilities required by the selected retrieval strategy and provide the session/transaction semantics appropriate to its storage engine.

## Open integration work / decisions

- [#20](https://github.com/eslam-aboutaleb/ragnor/issues/20): stable public SDK facade and import surface.
- [#21](https://github.com/eslam-aboutaleb/ragnor/issues/21): production builder injection for SQL sessions and proper lifecycle; prevent null-session fallback.
- [#26](https://github.com/eslam-aboutaleb/ragnor/issues/26): implement and *test* durable, atomic multi-host outbox and leases.
- [#27](https://github.com/eslam-aboutaleb/ragnor/issues/27): exact tokenizer adapter/budget and snapshot fingerprints.
- [#25](https://github.com/eslam-aboutaleb/ragnor/issues/25): thin synchronous outer facade after async lifecycle is stable.

**Architecture gate:** New adapters must satisfy the protocol, no-vendor-in-core dependency checks, an external consumer example, and capability-specific integration tests. A green static import test alone is insufficient to close issue #24.
