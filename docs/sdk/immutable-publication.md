# Durable vector publication, visibility and crash consistency

Issue #26 (P0): Ragnor's PostgreSQL outbox now offers an **opt-in**
`PostgresEmbeddingJobStore.activate_generation(job)` and
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
   PostgreSQL holds the job row lock, validates scope/claim/lease/persisted
   payload, serializes active publication per document, and commits the
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

The `DDL` sequence defines the new
`ragnor_active_generations(scope, document_id, revision, generation_key)`
table. Apply through a versioned, reviewed migration before deployment; the
SDK does not run migrations on import or startup. Use a tenant-isolated
database role or RLS in addition to SDK scoped predicates. Long-running
processing still requires heartbeat and graceful cancellation.

### Verified scenarios (when CI is green)

- Real PostgreSQL: staged-but-inactive vectors invisible to a gated reader
- Worker crash after stage / before ack, lease expiration and safe replay
- Delayed older revision cannot replace a newer visible revision
- Foreign-scope activation rejected
- Lost generation/lease cannot activate or acknowledge

Remaining independent gates: real production vector backend API-level
fencing; query authorization/RLS; DB disconnect midway through transaction;
fault injection across all provider write boundaries; performance/retention
baselines and human concurrency/security review.

## Enforcing the contract in production

Instantiate `PostgresEmbeddingJobStore(..., require_publication=True)` when
job payloads stage external vectors. This explicitly **disables**
`mark_completed` as a bypass around the durable active-generation manifest.
The host must stage immutable vectors, then call `activate_generation(job)`.
The generic legacy `process_pending_jobs` worker still calls
`mark_completed` and therefore is **not** a publication worker in strict
mode; do not silently route strict jobs through that generic loop.

For any multi-host worker using the generic processing API, pass
`require_atomic_claims=True` to reject local two-step stores up front.
These flags are opt-in for backward compatibility, so production
configuration validation must explicitly enable them.
