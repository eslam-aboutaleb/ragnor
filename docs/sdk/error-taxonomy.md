# SDK Error Taxonomy, Retryability & Fail-Closed Semantics

See [`docs/sdk/errors.md`](./errors.md) for the complete error contract matrix, strict vs non-strict retrieval semantics, sanitization policy, and FastAPI/CLI integration examples.

## Core Exception Hierarchy (`ragnor.exceptions`)

- `RagnorError(Exception)` — Base class with `.code`, `.retryable`, `.message`, `.details`, and `.trace_id`.
  - `ConfigurationError(RagnorError, ValueError)` — `ERR_CONFIG_INVALID` (`retryable=False`)
  - `RetrievalError(RagnorError)` — `ERR_RETRIEVAL` (`retryable=True`)
    - `BackendUnavailableError(RetrievalError)` — `ERR_BACKEND_UNAVAILABLE` (`retryable=True`)
    - `OperationTimeoutError(RetrievalError, TimeoutError)` — `ERR_OPERATION_TIMEOUT` (`retryable=True`)
  - `EmbeddingProviderError(RagnorError)` — `ERR_EMBEDDING_FAILED` (`retryable=False` by default; `True` for 429/5xx)
  - `IndexConsistencyError(RagnorError)` — `ERR_INDEX_INCONSISTENT` (`retryable=False`)
  - `UnsupportedCapabilityError(RagnorError)` — `ERR_UNSUPPORTED_CAPABILITY` (`retryable=False`)
  - `ScopeDeniedError(RagnorError, PermissionError)` — `ERR_SCOPE_DENIED` (`retryable=False`)
