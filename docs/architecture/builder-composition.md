# Builder composition and dependency ownership

The normal `RagnorBuilder.local(embedder)` convenience retains ephemeral in-memory stores and a no-op session provider, suitable for offline development only.

The builder is a **composition root** (not domain/application business logic). Production applications supply their own objects implementing the existing Ragnor ports:

```python
from ragnor.builder import RagnorBuilder

# Hosts supply embedder, vector_store, version_store and
# async session provider; these names represent application objects.
rag = await (
    RagnorBuilder(embedder=embedder, store=vector_store, version_store=version_store)
    .with_session_factory(session_provider)
    .with_sql_factory(lock_factory)
    .with_event_sink(telemetry_sink)
    .with_production_mode()
    .build()
)

# Calls a host-provided advisory lock adapter if lock_key is supplied.
await rag.ingest(documents, lock_key="tenant:collection")
```

Alternative retrieval strategies implement `RetrievalStrategy` and can be installed with `with_retrieval_strategy(strategy)`, independently of the vector index used for ingestion. A strategy must provide its own index/embedding dependencies. The host owns supplied clients and sessions; the builder does not close them. Session factory context managers are used for pipeline operations, and the ingestion pipeline commits *its own* sessions, not caller-supplied transaction sessions.

The `with_production_mode()` check rejects missing session factory/store/version store and known in-memory adapters. **It is a necessary guardrail, not a production certification.** Host adapters remain responsible for durability, filtering, transaction affinity, per-tenant isolation, thread/loop safety and distributed lock guarantees. SQLAlchemy is only imported for advisory locks when the host did not provide a factory; no SQL backend is mandatory for core imports.

For resource registration and shutdown, see [async lifecycle](async-lifecycle.md).
