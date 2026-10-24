# Production SDK startup configuration

`RagnorBuilder.local(...)` stays an ephemeral memory-backed recipe. Use
`RagnorBuilder(..., store=..., version_store=..., session_factory=..., production_mode=True)`
for a host-controlled deployment with explicit adapter capabilities.

## Fail-before-write checks

- Host supplies persistent store, version store, session provider and an
  embedding provider; no production null/memory fallback.
- Storage declares `StoreCapabilities` with isolation scope support. This
  declaration is an interface contract, **not proof** that the backend obeys
  it; validate actual tenant filters in backend-specific integration tests.
- The embedding model name must be nonempty. Unknown model dimensions do
  **not** silently accept the registry's 1536-dimensional development fallback:
  pass explicit positive `embedding_dim` or register the model dimension.
- Where a host store advertises an `embedding_dim` attribute and the builder
  has an explicit dimension, they must match before SDK construction.
- Injected stores and sessions remain host-owned unless ownership is
  deliberately transferred with `with_owned_resource`.

The checks do **not** certify PostgreSQL/other backend durability, filter
semantics, lease correctness, or secrets handling inside host implementations.
Validate those properties against the adapters and infrastructure used by your application.

No credentials or raw connection URLs are interpolated into startup exceptions.
Use a restrictive `ScopeAuthorizer` when `KnowledgeBridge(production_mode=True)`
exposes scoped documents.


Production mode also applies when `RagnorConfig.deployment_mode` is
`PRODUCTION`, including settings supplied through `with_settings`.
The builder checks callable storage/version/session ports before writes;
plain objects or capability labels alone are insufficient. These structural
checks do not verify persistence or authorize a tenant. Supply tested durable
adapters and bind authenticated tenant scope in the host session/provider.

## Readiness before returning a client

Structural validation does not contact a provider. To require live readiness,
register one async host check. Compose checks for every dependency your
application needs, using the same clients, credentials and authenticated
tenant scope as normal operations:

```python
from ragnor import RagnorBuilder
from ragnor.exceptions import ConfigurationError

async def check_dependencies() -> None:
    # strict=True distinguishes an empty index from an unavailable backend.
    await vector_store.count(strict=True)
    vector = await embedder.embed_query("readiness")
    if len(vector) != configured_dimension:
        raise ConfigurationError("embedding provider returned an unexpected dimension")

builder = RagnorBuilder.production(
    embedder,
    embedding_model=model_name,
    embedding_dim=configured_dimension,
    store=vector_store,
    version_store=version_store,
    session_factory=tenant_sessions,
    sql_factory=host_sql,
).with_startup_check(check_dependencies, timeout_seconds=10.0)

async with await builder.build() as rag:
    results = await rag.search("policy")
```

The check takes no arguments and must return an awaitable. Raise an exception
to reject startup; returning normally means ready. A later registration
replaces the previous check. No check runs unless explicitly registered.
Keep probes read-only: the SDK cannot roll back arbitrary writes made by a
host callback. An embedding probe makes a provider request and may incur cost.

The startup sequence is:

```text
validate configuration -> wire injected components -> transfer owned resources
    -> await bounded host check -> return usable facade
                 failure/cancellation -> dispose owned resources -> raise
```

Timeouts must be positive finite seconds. Checks and their clients must
cooperate with async cancellation; blocking synchronous calls cannot be
interrupted by an asyncio deadline. Timeout raises `OperationTimeoutError`;
other provider failures become `BackendUnavailableError` without including
the provider payload. Existing SDK domain errors and cancellation propagate.
Original causes remain available for host diagnostics.

Injected clients remain borrowed. Resources explicitly registered with
`with_owned_resource` are closed in reverse order if startup fails, expires
or is cancelled. Failed structural validation occurs before ownership
transfer and leaves disposal to the host. After a failed check, recreate
disposed clients before retrying. Startup and disposal errors are both
preserved in an exception group when both fail.

Bind tenant identity from trusted authentication at the host boundary. For
PostgreSQL, apply the tenant setting transaction-locally, enforce RLS using a
restricted non-bypass role, and verify both unfiltered reads and spoofed writes.
For external vector services, verify their collection/filter authorization
separately. A successful readiness probe is point-in-time connectivity evidence;
it does not replace authorization or ongoing failure handling.
