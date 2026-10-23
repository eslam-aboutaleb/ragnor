# Pluggable immutable-vector staging and real Qdrant conformance

`ImmutableGenerationVectorAdapter` lives in
`ragnor.jobs.adapters.generation_vectors` and depends on a **port**, not
Qdrant/pgvector packages. Supply an existing `QdrantStore`, `PgvectorStore`
or host implementation of `GenerationDocumentStore.get/upsert`.

```python
from ragnor.jobs.adapters.generation_vectors import ImmutableGenerationVectorAdapter
from ragnor.jobs.publication import ActiveGenerationReader

vectors = ImmutableGenerationVectorAdapter(store=host_vector_store, scope=tenant_scope)
await vectors.stage(job.payload["generation_key"], embedded_chunks)
# Only after ALL writes succeed:
await durable_postgres_outbox.activate_generation(job)
reader = ActiveGenerationReader(durable_postgres_outbox, vectors)
texts = await reader.read(tenant_scope, "policy")
```

The adapter stamps tenant/generation metadata, computes stable per-generation
document IDs, detects replay of altered content, and makes each read use both
scope AND generation filters. The **real** optional Qdrant local-engine CI
suite verifies staged-visibility, replays, revision switching and two tenants
with identical logical IDs. It cannot prove deployed remote Qdrant over
HTTP behaves identically under dropped connections or concurrent writes.

**Concurrency limitation:** cross-process check-and-upsert is not atomic
across arbitrary vector providers. A host must serialize conflicting
producers or use content-addressed immutable generations. Do not treat
direct `QdrantStore.get`, `vector_search` or `hybrid_search` as safe
production reads unless scoped to the active-generation manifest. The
manifest gate must be mandatory in host query paths. Stage retention,
cleanup, timeout/replay and remote-provider load tests still need
deployment-specific evidence.
