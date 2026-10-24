# Immutable vector-generation staging

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
scope AND generation filters. Configure authentication, authorization and
connection settings for your selected vector provider.

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
must provide the same durable atomic compare-and-swap semantics. Do not treat
direct `QdrantStore.get`, `vector_search` or `hybrid_search` as safe
production reads unless scoped to the active-generation manifest. The
manifest gate must be mandatory in host query paths. Configure stage retention,
cleanup, provider timeouts and retries for your deployment.
