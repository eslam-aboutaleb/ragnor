# Ragnor SDK Error Taxonomy, Retryability, and Fail-Closed Semantics

**Status:** Defined under [SDK-03 #22](https://github.com/eslam-aboutaleb/ragnor/issues/22).

This document establishes the typed failure contract across Ragnor's public API, preserving backward compatibility while enforcing fail-closed semantics and credential sanitization.

## Exception Hierarchy

All domain-specific errors inherit from `RagnorError`:

```
Exception
└── RagnorError (code, retryable, details)
    ├── ConfigurationError (ERR_CONFIG_INVALID, retryable=False)
    ├── BackendUnavailableError (ERR_BACKEND_UNAVAILABLE, retryable=True)
    ├── EmbeddingProviderError (ERR_EMBEDDING_FAILED, retryable=False)
    │   └── EmbeddingError
    │       ├── EmbeddingTransientError (ERR_EMBEDDING_TRANSIENT, retryable=True)
    │       └── EmbeddingPermanentError (ERR_EMBEDDING_PERMANENT, retryable=False)
    │           └── EmbeddingResponseError (ERR_EMBEDDING_RESPONSE, retryable=False)
    ├── IndexConsistencyError (ERR_INDEX_INCONSISTENT, retryable=False)
    │   ├── RevisionUnavailableError (ERR_REVISION_UNAVAILABLE)
    │   ├── StaleWorkerError (ERR_STALE_WORKER)
    │   ├── FencingError (ERR_FENCING)
    │   └── LostJobOwnershipError (ERR_LOST_JOB_OWNERSHIP)
    ├── UnsupportedCapabilityError (ERR_UNSUPPORTED_CAPABILITY, retryable=False)
    │   └── CapabilityUnsupportedError (ERR_CAPABILITY_UNSUPPORTED)
    ├── ScopeDeniedError (ERR_SCOPE_DENIED, retryable=False)
    │   ├── ScopeViolationError (ERR_SCOPE_VIOLATION)
    │   └── ScopeAuthorizationError (ERR_SCOPE_UNAUTHORIZED)
    ├── OperationTimeoutError (ERR_OPERATION_TIMEOUT, retryable=True)
    └── RetrievalError (ERR_RETRIEVAL)
```

## Failure Matrix and Diagnostics

| Public Exception | Error Code | Retryable? | Typical Origin | Recommended Caller Action |
|---|---|---|---|---|
| `ConfigurationError` | `ERR_CONFIG_INVALID` | No | Misconfigured dimensions, invalid options, missing provider extras | Fix configuration before retrying |
| `BackendUnavailableError` | `ERR_BACKEND_UNAVAILABLE` | Yes | Store connection timeout, unreachable PostgreSQL or Qdrant | Retry with exponential backoff and jitter |
| `EmbeddingProviderError` | `ERR_EMBEDDING_FAILED` | Depends | Upstream LLM/embedding API failure | Check child class (`EmbeddingTransientError` vs `EmbeddingPermanentError`) |
| `EmbeddingTransientError` | `ERR_EMBEDDING_TRANSIENT` | Yes | Rate limit (429), server 5xx, transient socket reset | Respect `retry_after` if provided; retry with backoff |
| `EmbeddingPermanentError` | `ERR_EMBEDDING_PERMANENT` | No | Auth failure (401/403), dimension mismatch, invalid model name | Correct API credentials or model identifier |
| `IndexConsistencyError` | `ERR_INDEX_INCONSISTENT` | No | Stale lease generation, snapshot mismatch, fencing failure | Re-evaluate active state, refresh lease |
| `UnsupportedCapabilityError` | `ERR_UNSUPPORTED_CAPABILITY` | No | Storage backend lacks requested filter or scoring feature | Choose compatible storage adapter or relax constraint |
| `ScopeDeniedError` | `ERR_SCOPE_DENIED` | No | Cross-tenant access attempt, missing partition authorization | Deny request; audit security boundary |
| `OperationTimeoutError` | `ERR_OPERATION_TIMEOUT` | Yes | Synchronous facade timeout, remote deadlocks | Retry or increase operation timeout |

## Fail-Closed Retrieval Semantics

In strict retrieval mode, backend unavailability or authorization failures **must never** return empty success:
1. When a vector database or index is unreachable, strict search raises `BackendUnavailableError` (or `RetrievalError`). An empty list `[]` represents genuine absence of matching records, not a failure.
2. Cross-tenant document hits raise `ScopeViolationError` immediately to prevent silent leakage.
3. Cancellation (`asyncio.CancelledError`) is strictly propagated and never swallowed as an empty result or generic error.

## Credential and Secret Sanitization

All exceptions automatically sanitize error messages and details dictionaries before rendering `str()` or `repr()`:
- Database connection strings: `postgresql://user:pass@host/db` is sanitized to `postgresql://user:***@host/db`.
- API keys, tokens, and bearer tokens: patterns matching `api_key`, `token`, `secret`, `password`, or `Bearer` have their values replaced with `***`.
