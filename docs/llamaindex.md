# LlamaIndex compatibility

`pip install "ragnor[llamaindex]"` adds `llama-index-core` (`>=0.12.0,<0.15`).
The core `ragnor` package stays zero-dependency and never imports
LlamaIndex; the adapters live in `ragnor.integrations.llamaindex`.

## Inbound: LlamaIndex into ragnor

```python
from llama_index.core.node_parser import SentenceSplitter
from ragnor.integrations.llamaindex import LlamaIndexChunker, LlamaIndexDocumentSource

source = LlamaIndexDocumentSource(my_reader)          # any BaseReader, or a list of Documents
chunker = LlamaIndexChunker(SentenceSplitter(chunk_size=256, chunk_overlap=20))
```

- `LlamaIndexDocumentSource` is a ragnor `DocumentSource`. Reader errors
  propagate. An empty list means the reader returned nothing.
- `LlamaIndexChunker` is a ragnor `Chunker`. Chunk ids are deterministic, and
  `version`/`config` change with the splitter and its settings.

## Outbound: ragnor into LlamaIndex

```python
from ragnor.integrations.llamaindex import RagnorRetriever, RagnorVectorStore

store = RagnorVectorStore(ragnor_store, isolation_scope="tenant-42")
retriever = RagnorRetriever(
    HybridRetrieval(ragnor_store, ragnor_store, embedder),
    isolation_scope="tenant-42",
    threshold=0.8,
)
```

Both require an `isolation_scope` (owner/tenant) and enforce it inside the
adapter:

- Writes are stamped with the scope. A node that sets the scope field to a
  different value is rejected.
- Queries, get, delete and clear are restricted to the scope. Filters naming the
  scope field are rejected. `clear()` only clears the adapter's own scope.
- Results are re-checked. A row from another scope raises
  `ScopeViolationError` (the retriever runs with `strict=True`).

`RagnorVectorStore` supports `DEFAULT`, `HYBRID` and `TEXT_SEARCH` query modes,
`EQ` and `IN` filters joined by `AND`, `add`, `delete(ref_doc_id)`,
`delete_nodes`, `get_nodes`, `clear` and the async variants. Anything else
raises `NotImplementedError` instead of being silently ignored. Sync methods
work inside a running event loop (they run on a worker thread).

## Verified against real backends

`tests/integration/test_llamaindex_backends.py` runs the same suite on the
in-memory store, a real Qdrant server and real PostgreSQL with pgvector
(CI service containers `qdrant/qdrant` and `pgvector/pgvector`). It covers
round trips, hybrid and keyword modes, get/delete/clear, idempotent upsert,
`IngestionPipeline.run` with the LlamaIndex source and chunker, and these
adversarial tenant-isolation cases: cross-scope read on every query path,
cross-scope write of the same node id, cross-scope delete and clear, filter key
and value injection, scope-field spoofing in filters and node metadata (any
letter case), interleaved async writers and readers across tenants.

Isolation notes:

- Row ids are namespaced `<scope>:<node_id>`. Without that, two tenants using the
  same node id would overwrite each other, because stores key rows by id.
  Nodes returned through `RagnorVectorStore` keep their original id.
  `RagnorRetriever` reads raw store rows, so over rows written by
  `RagnorVectorStore` it shows the namespaced id.
- When a pipeline writes rows, give the chunker `chunk_metadata={"tenant_id": ...}`.
  Chunk ids include that metadata, so identical text for two tenants does not collide.
- Filters are checked against the backend's declared capabilities and refused
  if unsupported (Qdrant is equality-only).
- Column-schema stores (pgvector) need a column for every metadata key a node or
  document carries, and for `ref_doc_id`, `node_content` and `node_type`.

## Known gaps

- Chroma and FAISS are not exercised through the adapter. The adapter was
  tested on memory, Qdrant (server 1.12.6 locally, 1.20.0 in CI) and pgvector.
- `IN` filters work only on the in-memory store. Qdrant refuses them (equality
  only). pgvector declares `IN` support but its SQL cannot bind a list, so the
  call raises instead of filtering.
- `distance_threshold` defaults to infinity (no cutoff). Distances are not
  comparable across metrics. Qdrant's store supports only the L2 metric.
  Similarities returned to LlamaIndex are `1 / (1 + distance)` (dense), the RRF
  score (hybrid) or the keyword score.
- No `NE`, `GT`, `LT`, `OR`, nested or text-match filters. No `MMR`, `SPARSE` or
  `SEMANTIC_HYBRID` modes. Query restriction by `node_ids`/`doc_ids` is not supported.
- Only scalar metadata with identifier-safe keys is filterable. Other metadata
  is kept in the stored node but cannot be filtered on.
- `IngestionPipeline.run` returns the store-wide count when a source is unchanged,
  so on a store shared by tenants the returned number is not per tenant (existing
  ragnor behavior, not changed here).
- Remote LlamaIndex readers are only as reliable as the reader. No network
  readers are tested.
- Tested with `llama-index-core` 0.12.0 (the lower bound) and the latest release
  in CI. The upper bound `<0.15` is a guard, not a promise.

## Environment note: NLTK data permissions

`SentenceSplitter` loads bundled NLTK data. Recent NLTK refuses data from a
directory that group or other accounts can write to, so on a host with a
permissive umask it raises `LookupError: Resource 'stopwords' not found`. If you
hit it, run `chmod -R go-w <site-packages>/llama_index/core/_static/nltk_cache`.
The test suite does this itself (`tests/llamaindex_env.py`).
