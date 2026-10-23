# PostgreSQL job outbox: atomic claiming and operational limits

Issue [#26](https://github.com/eslam-aboutaleb/ragnor/issues/26).

`PostgresEmbeddingJobStore` is an optional outer adapter backed by SQLAlchemy's async PostgreSQL driver. The SDK core `jobs.outbox` stays vendor-independent. The host must inject `async_sessionmaker` and run a versioned DB migration equivalent to the `DDL` tuple in `ragnor.jobs.adapters.postgres`. Schema creation is **never performed on ordinary SDK imports or job dispatch**. Do not create the schema automatically at runtime.

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from ragnor.jobs.adapters.postgres import PostgresEmbeddingJobStore
from ragnor.jobs.outbox import enqueue_job, process_pending_jobs

engine = create_async_engine(database_url)
sessions = async_sessionmaker(engine, expire_on_commit=False)
store = PostgresEmbeddingJobStore(sessions, scope="tenant-a", lease_seconds=300)

await enqueue_job(store, {"scope": "tenant-a", "idempotency_key": "doc/revision/chunk"})
processed = await process_pending_jobs(store, my_processor, worker_id="host-1")
await engine.dispose()
```

Use unique tenant/source/revision/chunk/model generation values in idempotency keys. A job with no matching scope is rejected. With a host-supplied session, enqueue executes in that **caller-owned transaction** without committing. Without one, enqueue commits atomically. Repeated identical scope/key inserts are ignored by the database constraint.

### Real guarantees

- Jobs are claimed and changed to `processing` atomically in one PostgreSQL transaction using `FOR UPDATE SKIP LOCKED`. The transaction is committed before the external embedding call.
- A claimant carries an incremented `claim_generation` and `claim_owner`. Completion and failure require exact ownership, processing status and an unexpired lease.
- On expiration, `reclaim_stale` revokes the previous generation and returns the job to pending. A previous claimant's ack is rejected.
- All storage operations are scoped by `scope`; worker identifiers are mandatory. Error details persisted in the database are sanitized to exception **types**, not raw messages.

### Known limitations: not yet production complete

- **At least once**, not exactly once. A timed-out worker can still perform an external embedding/vector write before its ack is rejected. The host processor MUST stage writes and validate generation/manifest before activating any index. No generic guarantee is possible across arbitrary external vector stores.
- Fenced `renew_lease(job, lease_seconds=...)` is available for host-driven heartbeats. Call it during long work, before lease expiry, at a cadence with margin for transient PostgreSQL latency. A stale or reclaimed worker is rejected. This adapter does not run a background heartbeat or cancel remote work on lease loss automatically.
- The end-to-end worker still requires host integration for heartbeat scheduling, cancellation and staged, generation-safe vector writes. Do not treat it as fully crash-safe until replay tests pass.
- The reference implementation requires PostgreSQL and SQLAlchemy async sessions; alternative adapters implement the same neutral ports and require their own real concurrency suites.
- Migration rollout, SIGTERM draining, metrics, dead-letter retry policy overrides, idempotent vector content identifiers, worker clock drift and multi-process kill/restart fault injection are additional #26 acceptance work.
- Do not use an in-memory implementation to claim distributed safety.

The GitHub Actions tests use a real PostgreSQL service and separate worker processes. They race two independent Python processes over 100 claims in two repeated trials, SIGKILL a worker after claim, reclaim its expired lease, and reject an obsolete acknowledgment. These tests do not certify side-effect idempotence across external vector databases or guarantee exactly-once remote execution. This is not a hosted job scheduler or message broker.
