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

## Known gaps

- Only the in-memory store is exercised in tests. Qdrant, pgvector, Chroma and
  FAISS are not covered by this integration's tests. The adapter relies on
  `extra_fields`, equality/`IN` filters and `delete(filters=...)`, so a backend
  that handles those differently could behave differently.
- `distance_threshold` defaults to infinity (no cutoff). Distances are not
  comparable across metrics. Similarities returned to LlamaIndex are
  `1 / (1 + distance)` (dense), the RRF score (hybrid) or the keyword score.
- No `NE`, `GT`, `LT`, `OR`, nested or text-match filters. No `MMR`, `SPARSE` or
  `SEMANTIC_HYBRID` modes. Query restriction by `node_ids`/`doc_ids` is not supported.
- Only scalar metadata with identifier-safe keys is filterable. Other metadata
  is kept in the stored node but cannot be filtered on.
- The adapters are not run through `IngestionPipeline.run` in tests. The source
  and chunker are tested against the protocols and a store, not through the
  version-tracked pipeline.
- LlamaIndex readers that fetch from remote services are only as reliable as the
  reader. No network readers are tested.
- Tested with `llama-index-core` 0.12.0 (the lower bound) and the latest release
  in CI. The upper bound `<0.15` is a guard, not a promise.
