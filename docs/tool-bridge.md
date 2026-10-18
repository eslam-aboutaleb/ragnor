# Knowledge bridge integration

Register every collection under its actual name. Omitting the collection selects the bridge's declared `default_collection`. An unknown name raises `KeyError`; the bridge never redirects it to another collection. Every operation checks both the client grants in `ScopeAuthorizer` and the collection's `allowed_scopes`. An empty scope allowlist denies all scopes.

```python
from ragnor.tools import KnowledgeBridge, CollectionConfig, ScopeAuthorizer

bridge = KnowledgeBridge(
    collections={
        "support": CollectionConfig(
            name="support",
            retriever=retriever,
            allowed_scopes={"tenant_a"},
            scope_field="tenant_id",
        ),
    },
    authorizer=ScopeAuthorizer({"support_client": {"support:tenant_a"}}),
)
```

Built-in retrieval strategies and `HybridRetriever` receive typed requests with the threshold, generation, session, mandatory capabilities and isolation constraints intact. The default scope field is `tenant_id` for those retrievers, `scope` for `SyncRetriever`, and `isolation_scope` for direct vector stores. Set `CollectionConfig.scope_field` explicitly when the physical index uses another field. Custom retrievers remain responsible for applying the supplied scope filters.

Management operations require `allow_management=True`. `sync_collection` rejects document ownership that contradicts the authorized scope and namespaces stored IDs by collection, scope, source and external ID. Supply a new revision when changing document content or metadata: conflicting writes to an existing immutable revision raise `ValueError`. Public and tenant-specific snapshots have distinct identities.

`revoke_document` revokes retained source access before deleting indexed chunks. Retention keeps historical bytes but does not permit `get_source` to return revoked content. A custom snapshot store used for revocation must implement `revoke(source_id, document_id, scope)` and `is_revoked(source_id, document_id, scope)` persistently. Scoped stores can implement `get_scoped(source_id, document_id, revision, scope)` in addition to the legacy lookup methods.

Evidence references escape their identifiers and include optional character offsets. Pass an item's `durable_id` directly to `get_source(source_ref=...)` to retrieve that span from the exact retained revision. Span limits must be positive integers and are clamped to the lookup's configured maximum.

`EvidenceBundle.total_tokens` measures the complete `format_context()` output, including notices, citation labels, provenance, delimiters and escaping. `EvidenceItem.token_count` measures the item's text. Reserved host instructions, user input and answer tokens remain separate. Model-specific limits require an exact tokenizer; the default tokenizer remains an estimate. Custom separators or additional host formatting need their own budget allowance.

## MCP client identity

MCP handlers use the SDK's authenticated access-token context when available. A local stdio host can bind a fixed client identity, or supply a trusted provider that resolves identity from its own session context:

```python
from ragnor.mcp import create_mcp_server

server = create_mcp_server(
    bridge,
    client_id="support_client",  # Host configuration for a local process
)
```

`client_id_provider=lambda: current_authenticated_client()` overrides the default resolution when the host supplies its own authentication. These are server configuration options, never tool arguments. Configure and validate remote transport authentication before deriving an identity; `RemoteTransportConfig` alone does not install transport middleware. The stdio runner accepts the same identity options.
