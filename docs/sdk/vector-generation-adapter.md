# Pluggable immutable-vector staging and real Qdrant conformance

`ImmutableGenerationVectorAdapter` lives in
`ragnor.jobs.adapters.generation_vectors` and depends on a **port**, not
Qdrant/pgvector packages. Supply an existing `QdrantStore`, `PgvectorStore`
or host implementation of `GenerationDocumentStore.get/upsert`.

```python
from ragnor.jobs.adapters.generation_vectors import ImmutableGenerationVectorAdapter
from ragnor.jobs.adapters.generation_reservations import PostgresGenerationReservations
from ragnor.jobs.publication import ActiveGenerationReader

# Apply migrations 001, 002 and 003 before starting workers.
reservations = PostgresGenerationReservations(host_sessions, scope=tenant_scope)
vectors = ImmutableGenerationVectorAdapter(
    store=host_vector_store, scope=tenant_scope, reservations=reservations,
)
await vectors.stage(job.payload["generation_key"], embedded_chunks)
# Only after ALL writes succeed:
await durable_postgres_outbox.activate_generation(job)
reader = ActiveGenerationReader(durable_postgres_outbox, vectors)
texts = await reader.read(tenant_scope, "policy")
```

The adapter stamps tenant/generation metadata, computes stable per-generation
document IDs, detects replay of altered content, and makes each read use both
scope AND generation filters. The optional Qdrant local-engine tests verify staged-visibility, replays, revision switching and two tenants
with identical logical IDs. The PostgreSQL CI job also provisions Qdrant HTTP, races independent
reservation sessions, tests partial writes and retries, and exercises a real
TCP disconnect/reconnect. These tests qualify those service versions and
paths; production credential, authorization and load configuration remain
host responsibilities.

The required `GenerationReservations` port atomically binds a generation
key to a SHA-256 digest of the complete batch before any vector access.
The PostgreSQL adapter commits the binding independently of external writes.
Two conflicting producers cannot both reserve a generation, including
producers proposing different chunk sets. Identical retries use stable IDs
and can safely complete a partial write; input order does not change identity.
A failed or cancelled vector write deliberately leaves the reservation intact.
Never remove it while the generation is readable or can be replayed. Existing
unreserved generations must be migrated or rebuilt before using this contract.

There is no implicit in-memory fallback. A host-supplied reservation adapter
must provide the same durable atomic CAS semantics and run real concurrency
tests. A hand-written reference adapter is only unit-test evidence. Do not treat
direct `QdrantStore.get`, `vector_search` or `hybrid_search` as safe
production reads unless scoped to the active-generation manifest. The
manifest gate must be mandatory in host query paths. Stage retention,
cleanup, timeout/replay and remote-provider load tests still need
deployment-specific evidence.
