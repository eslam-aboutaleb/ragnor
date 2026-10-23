# Real PostgreSQL architecture verification

This GitHub Actions job validates `SQLFactory` with **real PostgreSQL connections**.

- A caller-selected SQLAlchemy factory reads and writes through independently acquired sessions.
- Two independent sessions contend for the same PostgreSQL advisory lock; the second cannot enter until the first releases it.
- A failed database statement inside a lock rolls back and leaves the injected session usable.
- The test service uses an ephemeral PostgreSQL container with no application credentials or persistent user data.

## GitHub Actions gate

`postgres-contracts` provisions PostgreSQL 16 as a CI service, installs asyncpg and the optional `pgvector` SQLAlchemy dependencies, runs `tests/integration/test_postgres_adapter_contract.py`, and gates `ci-success`.

These tests are deliberately **not** run with fake sessions. Locally, supply `RAGNOR_TEST_DATABASE_URL` pointing to a disposable PostgreSQL 16+ instance. Without the env var they skip, but the GitHub Actions job always supplies it.

## Claims that are NOT supported by this suite

PostgreSQL advisory-lock behavior here **does not prove** a durable `EmbeddingJobStore`, atomic `SELECT FOR UPDATE SKIP LOCKED` job dispatch, stale worker fencing, crash recovery, or idempotent vector writes. Those are independent #26 acceptance criteria. Nor does this test provide tenant isolation guarantees for Qdrant/Chroma/FAISS or transactional consistency across external vector stores.

See #24 for architecture conformance and #21 for explicit production session/lifecycle wiring.
