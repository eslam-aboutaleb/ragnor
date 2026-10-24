# SDK Error Taxonomy, Retryability & Fail-Closed Semantics

Ragnor provides a unified, vendor-neutral exception hierarchy in `ragnor.exceptions`. All SDK domain errors inherit from `RagnorError` and expose:

- `message: str` — Sanitized human-readable description (secrets, bearer tokens, `sk-...` keys, DSNs, and raw document snippets are automatically redacted).
- `code: str` — Stable machine-readable error code (e.g., `ERR_CONFIG_INVALID`, `ERR_BACKEND_UNAVAILABLE`).
- `retryable: bool` — Deterministic indicator of whether the operation is safe to retry with backoff.
- `details: dict[str, Any]` — Sanitized structured diagnostic context.
- `trace_id: str | None` — Optional caller-supplied correlation identifier preserved across error translation.

## Exception Hierarchy & Contract Matrix

| Public Exception | Base Classes | Default Code | `retryable` | Origin / Trigger | Partial Side Effects? | Recommended Caller Action |
|---|---|---|---|---|---|---|
| `RagnorError` | `Exception` | `ERR_RAGNOR` | `False` | Base class for all Ragnor SDK errors | Operation-dependent | Inspect `.code` and `.retryable`; log `.trace_id` |
| `ConfigurationError` | `RagnorError`, `ValueError` | `ERR_CONFIG_INVALID` | `False` | Missing embedder, invalid `embedding_dim`, missing production adapters, event-loop conflict (`ERR_LOOP_CONFLICT`), closed client (`ERR_CLIENT_CLOSED`), prompt budget overflow (`PromptBudgetError`) | **None** (fails before I/O or write) | Fix configuration or caller lifecycle; do **not** retry |
| `RetrievalError` | `RagnorError` | `ERR_RETRIEVAL` | `True` | Base class for strict-mode retrieval failures | **None** (read-only) | Catch for broad retrieval error handling |
| `BackendUnavailableError` | `RetrievalError` | `ERR_BACKEND_UNAVAILABLE` | `True` | Vector store / SQL connection error, HTTP 429 (`ERR_RATE_LIMITED`) or 5xx (`ERR_UPSTREAM_5XX`), strict search/count backend failure | Read paths: **None**. Write paths: uncommitted SQL transactions roll back | Retry with exponential backoff or trip circuit breaker |
| `OperationTimeoutError` | `RetrievalError`, `TimeoutError` | `ERR_OPERATION_TIMEOUT` | `True` | `TimeoutError` / `asyncio.TimeoutError` during search, ingest, count, embedding, or `SyncRagnor` call | In-flight task cancelled; uncommitted DB transaction rolled back | Retry with backoff or increase timeout budget |
| `EmbeddingProviderError` | `RagnorError` | `ERR_EMBEDDING_FAILED` | `False` (or `True` on 429/5xx) | Embedding provider failure, dimension mismatch, NaN/Inf vector, empty batch response | **None** (embeddings run outside DB transaction before version mutation) | Retry only when `err.retryable is True`; alert on 401/403 or dimension mismatch |
| `IndexConsistencyError` | `RagnorError` | `ERR_INDEX_INCONSISTENT` | `False` | Stale generation (`RevisionStaleError`), expired sync lease (`FencingTokenError`), or lost outbox job ownership (`LostJobOwnershipError`) | Write rejected by fencing guard before commit | Re-acquire lease or refresh active generation before retrying |
| `UnsupportedCapabilityError` | `RagnorError` | `ERR_UNSUPPORTED_CAPABILITY` | `False` | Backend lacks mandatory capability or filter operator (`CapabilityUnsupportedError`) | **None** (rejected before query execution) | Use a backend supporting the required capability or adjust query constraints |
| `ScopeDeniedError` | `RagnorError`, `PermissionError` | `ERR_SCOPE_DENIED` | `False` | Unauthorized tenant scope (`ScopeAuthorizationError`) or cross-scope result leak (`ScopeViolationError`) | **None** (blocked before returning data) | Return HTTP 403 / permission denied; audit caller scope |

## Strict vs Non-Strict Retrieval

- **Local / Default Mode (`RagnorBuilder.local(...)`)**: `rag.search(query)` defaults to `strict=False` for backward compatibility, logging backend storage errors and returning `[]`, while **always** raising `CapabilityUnsupportedError`, `ScopeViolationError`, and `ConfigurationError` (fail-closed security and capability checks are never swallowed). Pass `strict=True` (`await rag.search(query, strict=True)`) to fail closed on backend outages.
- **Production Mode (`RagnorBuilder.production(...)` / `.with_production_mode()`)**: `rag.search(query)` and `rag.count()` default to `strict=True`. Legitimate zero-hit searches return `[]`, whereas backend outages raise `BackendUnavailableError` (or `OperationTimeoutError`), preventing silent ungrounded answers during a database or vector-store outage.

## Outbox Dead-Letter & Sanitization Policy

- `RagnorError` and `sanitize_message()` redact DSN passwords (`postgresql://user:***@host/db`), `Authorization: Bearer ***`, `sk-***` keys, `api_key=***`, `token=***`, `secret=***`, and `document_text=***` / `prompt_content=***` snippets.
- In `ragnor.jobs.outbox`, `status_detail` persisted to the job table on `failed` or `dead_letter` transitions records only the failure category and exception class name (e.g., `Permanent failure (ConfigurationError)` or `Exhausted 4 attempts (RuntimeError)`), never raw provider payloads or customer document text.

## Sample Host Error Handlers

### FastAPI Exception Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ragnor.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    OperationTimeoutError,
    RagnorError,
    ScopeDeniedError,
    UnsupportedCapabilityError,
)

app = FastAPI()


@app.exception_handler(RagnorError)
async def ragnor_exception_handler(request: Request, exc: RagnorError) -> JSONResponse:
    if isinstance(exc, ScopeDeniedError):
        status_code = 403
    elif isinstance(exc, (ConfigurationError, UnsupportedCapabilityError)):
        status_code = 400
    elif isinstance(exc, OperationTimeoutError):
        status_code = 504
    elif isinstance(exc, BackendUnavailableError):
        status_code = 503
    else:
        status_code = 500

    headers = {"Retry-After": "5"} if exc.retryable else {}
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "trace_id": exc.trace_id,
            }
        },
    )
```

### CLI Error Handler

```python
import sys
from ragnor.exceptions import RagnorError


def run_cli_command(fn) -> int:
    try:
        fn()
        return 0
    except RagnorError as exc:
        prefix = "[RETRYABLE]" if exc.retryable else "[FATAL]"
        sys.stderr.write(f"{prefix} {exc.code}: {exc.message}\n")
        return 75 if exc.retryable else 1
```


SDK-generated provider failure logs and wrapping error messages report the
operation and exception type, not the provider exception text. Original
causes remain available through `__cause__` for compatibility; application
logging of a complete chained traceback must apply the host's redaction
policy. SDK logs do not attach those tracebacks or raw query text.
