# Host-owned SQL outbox

The SDK does not choose a SQL database, read a database URL from settings, create
an engine, or run migrations. The host creates one engine and async session
factory, applies its schema, and injects the same binding into ingestion and
durable jobs. Vector storage can be a separate service; its client is configured
explicitly by the host.

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from ragnor.jobs.adapters.relational import RelationalJobStoreFactory

engine = create_async_engine(host_database_url, pool_pre_ping=True)
sessions = async_sessionmaker(engine, expire_on_commit=False)
database = RelationalJobStoreFactory(sessions)

builder = builder.with_job_persistence(
    database, scope=authenticated_scope, require_publication=True
)
# On shutdown: await engine.dispose()
```

The host owns `host_database_url`, engine lifetime, driver installation, secrets,
connection pooling, backup, and migrations. `with_job_persistence` uses the same
binding for ingestion and the outbox; it rejects a different preconfigured
session provider. Pass `database.reservations(scope=authenticated_scope)` to
generation publication. The scope must come from trusted authentication.

Older releases exposed `ragnor.db.session.create_session_factory(database_url)`
and `SQLAlchemyFactory.create_session_factory(database_url)`. The former import
still exists but raises an actionable error without opening an engine; the
latter method is removed from the SQL operation factory. Move engine and
session creation into the host as shown above. `RagnorConfig`,
`BuilderSettings`, and the `RagSettings` protocol no longer carry
`database_url`. Pass an explicit `url` or host-owned client to `QdrantStore`,
and host-owned sessions to `PgVectorStore`.

## Schema and upgrade

Apply the tables and indexes in
[`ragnor.jobs.adapters.relational.metadata`](../src/ragnor/jobs/adapters/relational.py)
through your migration system before starting workers. PostgreSQL installations
already using the original outbox apply the packaged SQL migrations 001–004 in
order; inspect existing columns, constraints, and data before an upgrade. In
particular, the scoped `(scope, idempotency_key)` unique constraint, claim
generation, lease columns, and source lock table are required. For a new SQLite
deployment, translate the metadata into a versioned host migration. `create_all`
in tests is an example of the host action, not an SDK startup step. Claim
preflight checks required tables, columns, primary keys, scoped uniqueness,
driver row counts, dialect, and transaction settings before processing. For
SQLite it also performs a rollback probe against the host binding to reject
driver modes where a nested savepoint escapes the outer transaction.

Grant the worker role only the table permissions its operations need. Use a
separate migration role for DDL. Enforce tenant authorization at the host
boundary and, where available, database row policies; SDK scope predicates
are a safety boundary but do not authenticate the caller. Keep migration and
application roles from bypassing those policies during request handling.
For PostgreSQL, the worker's table grants are `SELECT, INSERT, UPDATE` on
`ragnor_embedding_jobs`, `ragnor_active_generations`, and
`ragnor_ingestion_locks`, and `SELECT, INSERT` on
`ragnor_generation_reservations`. The retention/maintenance role owns any
`DELETE` privilege. Apply equivalent least-privilege controls on other hosts;
SQLite relies on filesystem access and process isolation rather than SQL roles.

## Qualified deployments

The built-in distributed adapter is qualified with SQLAlchemy async
PostgreSQL/asyncpg at `READ COMMITTED` and file-backed SQLite/aiosqlite at
`SERIALIZABLE`, with `read_uncommitted=OFF` and a positive `busy_timeout`.
SQLite's driver transaction control must be configured so savepoints join the
outer transaction. For Python versions before the driver exposes modern
transaction control, the host can use SQLAlchemy connect/begin listeners:

```python
from sqlalchemy import event


@event.listens_for(engine.sync_engine, "connect")
def configure_sqlite(dbapi_connection, _record):
    dbapi_connection.isolation_level = None


@event.listens_for(engine.sync_engine, "begin")
def begin_sqlite(connection):
    connection.exec_driver_sql("BEGIN")
```

Use that pair only for a SQLite engine. SQLite serializes file writers; more
workers may increase contention instead of throughput. Claims use conditional
updates and fenced leases. Busy/operational claim failures retry with bounded
backoff: at most 20 claim attempts, exponential delay starting at 25 ms,
capped at 500 ms with jitter. After exhaustion the error reaches the host.
Set connection and statement timeouts in the host so one blocked attempt does
not outlive the worker lease. Set the worker's processing deadline below the
lease or renew it during long work. Monitor retry failures and rerun the real
database contract tests for your driver versions. A backend other than the two
qualified combinations needs a separately tested host adapter; it is not
silently accepted as distributed-safe merely because SQLAlchemy can connect.

The host decides retention. Do not purge pending, processing, or retryable
failed jobs. Purging completed jobs also removes their idempotency record, so
retain them for at least the maximum replay window or maintain a separate
deduplication record. Keep active generations and generation reservations for
as long as their keys can be replayed. A host using an external vector service
also cleans up orphaned staged vectors only after the corresponding job or
publication attempt can no longer win its lease/fence; validate readers against
the committed active generation until cleanup completes. Back up these tables
with the host's other authoritative state.

CI runs the same failure matrix on real PostgreSQL and file-backed SQLite,
including separate worker processes, killed-worker replay, fencing, duplicate
enqueue, rollback, reservations, publication order, tenant isolation, and
reconnection. `benchmarks/relational_outbox_profile.py` records 1/2/4-worker
throughput for each database without treating one machine's result as an SLA.
In a local 80-job run on 25 October 2026, observed jobs per second were:

| Host database | 1 worker | 2 workers | 4 workers |
| --- | ---: | ---: | ---: |
| PostgreSQL/asyncpg | 177 | 436 | 489 |
| SQLite/aiosqlite | 141 | 188 | 186 |

The SQLite two-worker run assigned all 80 jobs to one worker, illustrating
writer serialization and scheduling variability. Re-run the benchmark with
your storage, pool, workload, and worker placement before sizing a deployment.
