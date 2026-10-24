# Synchronous Python Facade — `SyncRagnor`

`SyncRagnor` provides a safe, optional synchronous API for blocking Python scripts, batch workers, and CLI utilities, wrapping Ragnor's canonical asynchronous core.

## Quickstart

```python
from ragnor.builder import RagnorBuilder
from ragnor.sync_client import SyncRagnor
from ragnor.types import Document

def make_builder():
    return RagnorBuilder.local(...)

with SyncRagnor.from_builder(make_builder) as client:
    count = client.ingest([Document(id="doc-1", text="Sample policy text")])
    results = client.search("policy", n_results=5)
    total = client.count()
```

## Architectural Design

1. **Persistent Worker Thread and Event Loop:**
   `SyncRagnor` manages a single dedicated background thread running an `asyncio` event loop. Calls to `ingest`, `search`, and `count` submit coroutines via `asyncio.run_coroutine_threadsafe` and block until complete. This avoids the overhead and state-clearing pitfalls of calling `asyncio.run()` repeatedly.

2. **Loop Safety and Fast Failure:**
   Attempting to instantiate `SyncRagnor` from inside a thread with an active running `asyncio` event loop (such as inside an `async def` function or Jupyter notebook) fails fast with `ConfigurationError(code='ERR_LOOP_CONFLICT')`. In asynchronous environments, use `await ragnor.search(...)` directly.

3. **Pluggable Dependency Injection:**
   `SyncRagnor.from_builder` accepts any configured `RagnorBuilder`. Custom embedders, chunkers, vector stores, session factories, and telemetry event sinks are injected into the underlying async instance without modifying core code.

4. **Resource Lifecycle and Timeouts:**
   - `client.close()` shuts down the underlying async resources via `await ragnor.aclose()`, stops the background event loop, and joins the worker thread.
   - `close()` is idempotent; multiple calls are safe no-ops.
   - Operations support an explicit `timeout` parameter and raise `OperationTimeoutError` upon expiration.

Owned resources must honor cancellation and their close contracts so that
timeouts and shutdown can release connections and other resources promptly.
