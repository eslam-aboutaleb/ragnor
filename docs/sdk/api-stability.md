# Ragnor Python SDK — public API and compatibility policy

The minimal supported root imports are:

```python
from ragnor import Ragnor, RagnorBuilder, Document, Chunk, SearchResult, RetrievalError
```

| Root public symbol | Canonical implementation | Compatibility |
|---|---|---|
| `Ragnor`, `RagnorBuilder` | `ragnor.client.builder` | `ragnor.builder` imports remain valid |
| `Document`, `Chunk`, `SearchResult`, `RetrievalError` | `ragnor.core.types` | `ragnor.types` imports remain valid |

The `ragnor.core` package contains shared types, configuration, exceptions
and validation. The `ragnor.client` package contains the async builder and
synchronous facade. Existing imports from `ragnor.builder`, `ragnor.types`,
`ragnor.config` and the other established module paths continue to resolve
to the same implementations, including their type annotations.

All root exports are **lazy**. Importing `ragnor` must not initialize databases, model clients, telemetry pipelines, or optional vendor dependencies. The root exports are deliberately small. Power users can import the established extension protocols from `ragnor.chunking.base`, `ragnor.embeddings.base`, `ragnor.stores.base`, `ragnor.jobs.outbox`, and `ragnor.evidence.tokenizer`.

## Minimal local example (ephemeral, for demos only)

```python
import asyncio
from ragnor import RagnorBuilder, Document

class DemoEmbedder:
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float("refund" in text.lower()), 1.0] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)

async def main() -> None:
    rag = await RagnorBuilder.local(DemoEmbedder(), embedding_dim=2).build()
    await rag.ingest([Document(id="policy-1", text="Refund policy")])
    print(await rag.search("refund"))
    print(await rag.count())

asyncio.run(main())
```

For real deployment inject a durable store, version store, session factory, and restrictive tenant policy. `with_production_mode()` rejects known ephemeral defaults but **does not attest** the truth of a custom backend's storage guarantees; adapters require conformance tests and operational validation.

## Versioning and compatibility

Ragnor is a **0.x** package. The public names above are intended to stay stable within compatible minor-version releases, but this is not a promise of 1.0 semantic stability. Prefer additive changes. Breaking public signatures, return types, or import paths require a documented migration and release notes. Deprecated paths, if ever introduced, remain available for an announced transition period with explicit warnings rather than silent deletion.

**Async remains canonical.** `Ragnor.ingest()`, `search()`, `count()` retain existing semantics. Storage capability requirements are evaluated by the injected backend and retrieval policies. Unsupported capabilities fail explicitly.

`SearchResult` is a dataclass used at the high-level API; internal vector backends may continue passing dictionaries. Don't import concrete vendor stores from the public root, and don't assume the debug/local embedder is suitable for production.

## Core Method & Result Semantics

- **Document IDs & Duplicate IDs**: Each `Document(id=..., text=..., metadata=...)` maps to a source record identified by `Document.id` (stamped into `chunk.metadata["source"]` and `chunk.metadata["external_id"]`). Re-ingesting an unchanged document list is an idempotent no-op via canonical manifest hashing. Re-ingesting a modified document set retires the previous version's chunks and upserts the new version. Atomicity requires stores that participate in the same supplied transaction; a separate vector service requires immutable staging and manifest-based publication. The local builder does not provide a cross-storage transaction.
- **Filtering & Capability Enforcement**: Keyword filter arguments passed to `rag.search(query, n_results=5, **filters)` and `rag.count(**filters)` are validated against the backend's declared `StoreCapabilities` and pushed down to both vector and lexical search stages. When a backend lacks a required filter operator (e.g., `IN` list filters on an equality-only adapter) or a mandatory capability constraint, Ragnor fails closed by raising `CapabilityUnsupportedError` (`UnsupportedCapabilityError`, a subclass of `RagnorError` and `RetrievalError`) rather than silently dropping filters.
- **Response Ordering & Empty Results**: `rag.search()` returns `list[SearchResult]` ordered deterministically by descending Reciprocal Rank Fusion score (`rrf_score`), or by reranker order when a `Reranker` is configured. Queries with zero matching documents return an empty list `[]`. In `production_mode=True` (or when `strict=True` is passed), backend outages raise `BackendUnavailableError` so an unavailable store is never confused with `[]`.
- **Metadata Propagation**: User-supplied `Document.metadata` fields are preserved on generated chunks alongside system provenance (`source`, `external_id`, `version_id`).
- **Facade vs Low-Level Seam Boundary**: High-level document ingestion, hybrid search, counting, evidence assembly (`assemble_evidence`), and readiness checks (`check_health`) live on `Ragnor`. Direct chunk-level CRUD (`rag.store.get(ids=...)`, `rag.store.delete(ids=..., filters=...)`) and custom `DocumentSource` pipeline runs (`rag.pipeline.run(source, version_store)`) remain reachable via `rag.store`, `rag.pipeline`, and `rag.retriever`.
