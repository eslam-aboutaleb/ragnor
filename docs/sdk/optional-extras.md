# Ragnor SDK optional extras

Ragnor has **zero mandatory third-party runtime dependencies**. Install the
extras needed by your application; optional adapters load their dependencies
when used.

| Extra | Provides |
| --- | --- |
| `litellm` | Hosted model embeddings through LiteLLM |
| `rerank` | Model-based reranking |
| `pgvector` | SQLAlchemy async storage and PostgreSQL integration |
| `database` | Host-bound SQLAlchemy outbox, reservations, and SQL transaction locks |
| `qdrant` | Qdrant vector storage |
| `chroma` | Chroma vector storage |
| `faiss` | Local FAISS indexing |
| `pdf` | PDF text extraction |
| `directory` | Local directory extraction |
| `watch` | File watching |
| `web` | HTTP document loading |
| `object-storage` | Object storage connectors |
| `opentelemetry` | Tracing integration |
| `mcp` | MCP integration |
| `tokenizers-tiktoken` | Exact counting with a selected tiktoken encoding |
| `test` | Repository test dependencies; not needed at runtime |
| `local-embeddings` | Reserved and empty; does not install a local-model provider |
| `all` | Provider extras, excluding `test` and reserved extras |

```bash
python -m pip install 'ragnor[qdrant]'
python -m pip install 'ragnor[pgvector]'
python -m pip install 'ragnor[tokenizers-tiktoken]'
```

Inject credentials and configure connections through your host application.
Provider consistency, tenant authorization and network availability depend
on the selected service and deployment configuration.
