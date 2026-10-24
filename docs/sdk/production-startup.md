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
