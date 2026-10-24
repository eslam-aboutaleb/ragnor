# Durable vector publication, visibility and crash consistency

Ragnor's host-bound relational outbox offers an **opt-in**
`RelationalEmbeddingJobStore.activate_generation(job)` and
`get_active_generation(document_id)`, and a pluggable
`ActiveGenerationReader`. These do **not** replace the regular job store
contract or add mandatory SQLAlchemy dependencies.

## Required contract for external vector providers

1. Enqueue an embedding job with non-empty authenticated `scope`,
   unique `idempotency_key`, `document_id`, non-negative monotonic
   `revision`, and an immutable `generation_key` that includes the scope,
   document and model/content revision.
2. Atomically claim the job with a lease and generation token. Stage vectors
   under that immutable generation key; **never overwrite a currently active
   generation or reuse a key for different content**.
3. After a confirmed stage operation, call `activate_generation(job)`.
   The SQL database validates scope, claim, lease and persisted
   payload; conditional updates serialize active publication per document and commit the
   active manifest and job completion **within the same transaction**.
4. Route searches through the injected generation-scoped reader, which
   fetches the authoritative active key from the manifest and directs the
   vector provider to search only that key. Direct unfiltered provider
   queries **do not** satisfy the publication safety contract.
5. A crash after staging but before activation leaves the old generation
   visible. On lease expiry, reclaim and replay the same immutable stage
   under the same key. Old owner activation fails. A later source revision
   always wins over earlier revisions. Orphan generations can be garbage
   collected after a separately defined retention period.

This preserves **single active manifest generation** for cooperating
readers and immutable-generation adapters. It does **not** make external
provider writes transactional or globally exactly once. Providers must
honor immutable keys and generation filters; deployment must test those
properties with each real Qdrant/pgvector/etc adapter.

The SQLAlchemy `metadata` in `ragnor.jobs.adapters.relational` defines the
job, reservation, active-generation and ingestion-lock tables. Apply the
equivalent schema through host-owned versioned migrations before deployment;
the SDK does not run migrations on import or startup. PostgreSQL deployments
can continue using the packaged SQL migrations. Use database authorization
in addition to SDK scoped predicates. Long-running
processing still requires heartbeat and graceful cancellation.

## Deployment responsibilities

Use a tenant-isolated database role or row-level security and enforce
authorization in the application. Configure provider timeouts, retry policy,
heartbeat scheduling and retention for orphaned generations. A separate
vector service does not participate in the SQL transaction.

## Enforcing the contract in production

Bind `RelationalJobStoreFactory(host_sessions)` and create its scoped job store
with `require_publication=True` when
job payloads stage external vectors. This explicitly **disables**
`mark_completed` as a bypass around the durable active-generation manifest.
The host must stage immutable vectors, then call `activate_generation(job)`.
The `process_pending_jobs` worker checks `require_publication` before
claiming and routes successful stages through `activate_generation(job)`
when required. It rejects a publication-required store without that port
before processing. Legacy stores retain `mark_completed` behavior.

For any multi-host worker using the generic processing API, pass
`require_atomic_claims=True` to reject local two-step stores up front.
These flags are opt-in for backward compatibility, so production
configuration validation must explicitly enable them.
