# Async SDK lifecycle and dependency ownership

Ragnor receives most infrastructure dependencies **borrowed** from the host.
The SDK must never implicitly dispose of an injected event sink, vector store,
session provider, database engine, embedding client or retriever.

Where a host intentionally transfers ownership of a resource implementing
`async def aclose(self) -> None`, register it explicitly:

```python
builder = RagnorBuilder.local(embedder)
builder.with_owned_resource(owned_async_client)
rag = await builder.build()

async with rag:
    await rag.ingest(documents)
# owned_async_client.aclose() was awaited by Ragnor; injected borrowed
# stores, embedders, sessions and sinks were not closed.
```

- Calls after `aclose()` are rejected (including re-entering a closed client).
- Repeated `aclose()` is idempotent.
- Owned resources close in reverse registration order, once by object identity.
- If cleanup fails, later resources are still attempted and errors are surfaced.
- Resource construction is still performed by the host. This facade is **not**
  a general-purpose dependency-injection framework.
- An application can keep the async facade open without `async with` and
  call `await rag.aclose()` during its own shutdown.
- Externally owned session/SQL factories do not automatically become owned
  when passed through `with_session_factory` or `with_sql_factory`.

Keep clients and sessions on their owning event loop. Finish direct calls
through low-level stores, pipelines and retrievers before closing their
resources; the facade tracks only operations performed through its own API.
