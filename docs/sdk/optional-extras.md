# Ragnor SDK optional extras

Ragnor has **zero mandatory third-party runtime dependencies**. Install one
extra for the corresponding host-integrated adapter. The SDK core can be
imported without provider credentials, SQLAlchemy, hosted agents or LLM clients.

| Extra | Intended use | Qualification |
| --- | --- | --- |
| `litellm` | Hosted model embedding via a host-supplied LiteLLM configuration | `test_embeddings_litellm.py`, optional CI |
| `rerank` | Optional reranking model integration | `test_reranker_integration.py`, optional CI |
| `pgvector` | SQLAlchemy async store/transactions | Pgvector tests and real PostgreSQL service CI |
| `qdrant` | Qdrant vector store | Store tests and Qdrant local-client + PostgreSQL publication |
| `chroma` | Chroma store | Vector backend conformance tests |
| `faiss` | FAISS local index | Vector backend and FAISS tests |
| `pdf` | PDF text extraction | Connector tests with sample PDFs |
| `directory` | Host local directory extraction | Directory connector tests |
| `watch` | Optional file watching | Directory connector tests |
| `web` | HTTP document loading | HTTP connector tests |
| `object-storage` | Object store integration | Object connector tests |
| `opentelemetry` | Optional tracing | OTel bridge and event tests |
| `mcp` | MCP host/server adapter | MCP adapter/stdio tests |
| `tokenizers-tiktoken` | Exact BPE tokenizer | Multilingual optional tiktoken CI |
| `test` | Repository tests only; not needed by SDK runtime | Python 3.11/3.12 CI |
| `local-embeddings` | **Reserved only**; empty and not a working local-model provider | Never included in `all` |
| `all` | Supported provider extras together, excluding test and reserved | Metadata contract and independent backend CI |

Install examples:

```bash
python -m pip install 'ragnor[qdrant]'
python -m pip install 'ragnor[pgvector]'
python -m pip install 'ragnor[tokenizers-tiktoken]'
```

Each optional adapter remains independently injectable. Passing dependency
installation and local tests does not certify a third-party hosted provider's
network availability, credentials, consistency or row-level authorization.
Deployment tests must validate each selected real provider. Test/extras
compatibility evolves with the pinned `pyproject.toml` metadata and is
checked in GitHub Actions before a release.
