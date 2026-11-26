# Deployment credentials, authorization, and client qualification

This guide is the deployment-side companion to
[production startup](production-startup.md). It covers the two acceptance
areas of [SDK-02](https://github.com/eslam-aboutaleb/ragnor/issues/21) that
code alone cannot close: how a deployment configures credentials and
authorization, and how it qualifies its *actual* embedding/vector clients.

## Credential configuration

Ragnor core never loads `.env` files, never reads process environment
implicitly, and never invents credentials. Every secret is host-supplied:

| Secret | Where it enters | Rule |
|---|---|---|
| Embedding provider key | Host settings object satisfying `RagSettings` (e.g. `openai_api_key`), or the host's own adapter constructor | Passed by the host; never logged or repr'd by Ragnor |
| Database connection string | Host-composed SQLAlchemy engine / injected `SQLFactory` and `SessionProvider` | Ragnor never builds or stores DSNs in core |
| Object storage / web connector credentials | Host-configured optional adapter (`boto3`, `httpx`) | Live in the adapter layer, never in core |

Redaction is a tested contract, not a convention:

- `RagnorConfig` redacts credentials in `repr`/string output.
- All `RagnorError` subclasses sanitize messages, details and trace IDs
  (`sanitize_message` redacts DSN passwords, bearer tokens, `sk-...`
  provider keys and `api_key`/`token`/`secret`-style key values).
- Startup and readiness failures classify deadlines and sanitize provider
  payloads before re-raising.
- Regression coverage: `tests/test_sdk_secrets_contract.py`,
  `tests/test_sdk_startup_checks.py`, `tests/test_config.py`.

Misconfigured or incomplete production wiring fails before any write with a
typed `ConfigurationError` naming the missing setting, never its value.

## Authorization configuration

Authorization is a host-injected port, and production is fail-closed:

- `KnowledgeBridge` and retrieval paths take a host `ScopeAuthorizer`. In
  production mode a missing or permissive authorizer is rejected; there is
  no allow-all default. Coverage: `tests/test_production_authorizer.py`.
- Tenant isolation is a declared store capability
  (`BackendFeature.ISOLATION_SCOPE`); a production store that cannot filter
  by tenant scope is rejected at composition time instead of serving
  unscoped data. Coverage: `tests/test_production_capability_security.py`.
- Mandatory scopes are never widened or dropped because a provider cannot
  implement them; unsupported capabilities raise
  `UnsupportedCapabilityError` explicitly.

## Qualifying your deployed clients

The repository qualifies the bundled PostgreSQL/pgvector and Qdrant paths
with real services, and transport behavior against explicitly synthetic
local endpoints. **Those results do not attest your injected clients.** A
deployment must qualify its own embedding and vector clients for startup,
cancellation, disconnect and cleanup. Ragnor ships a pytest-agnostic
harness for this in `ragnor.qualification`:

```python
from ragnor.qualification import qualify_client_lifecycle, qualify_embedding_client

# Embedding client: protocol shape, startup deadline, batch/dimension
# contract, cancellation propagation, aclose idempotence, terminal disconnect.
report = await qualify_embedding_client(my_embedder, expected_dimension=1536)
report.require_passed()

# Any other deployed client (e.g. a vector store): supply a real probe.
report = await qualify_client_lifecycle(my_store, my_store.count)
report.require_passed()
```

The harness runs real calls with real deadlines and real cancellation
delivery against the client you hand it. Run it in your own CI or startup
checks against the exact client build you deploy; every check detail is
sanitized so provider payloads cannot leak credentials through a report.

A passing harness qualifies the *integration contract* only. It is not
model-quality evidence, and it does not replace the independent
deployment-security review tracked in the issue.

## Acceptance status

Covered by repository evidence:

- Credential redaction in config, errors, logs and startup failures.
- Fail-closed production authorization and capability declaration.
- Fail-before-write validation of incomplete/misconfigured production wiring.
- Startup readiness with bounded timeouts, cancellation-safe owned cleanup,
  and borrowed-resource survival for injected clients.
- A shipped, reusable qualification harness for deployed embedding/vector
  clients (`ragnor.qualification`, `tests/test_sdk_client_qualification.py`).

Still required per deployment (not closable from this repository):

- Provisioning the real tenant/credential/authorization configuration and
  having it independently reviewed.
- Running the qualification harness (or an equivalent procedure) against
  the actual deployed embedding/vector clients and recording the result.
