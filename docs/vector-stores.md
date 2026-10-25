# Vector stores

This page keeps the backend-specific setup and behavioral details together. Start with the [README overview](../README.md#vector-stores) to choose a store.

ragnor is vector-database agnostic: it ships no required database. Every backend — built-in or your own — implements the same `VectorStore` ABC with three methods: `hybrid_search`, `count`, and `upsert` (full signatures in `src/ragnor/stores/base.py`; the ABC itself has no database dependency). Storage is split from retrieval strategy: three `@runtime_checkable` capability Protocols describe what a backend can do — `VectorIndex` (`vector_search`), `LexicalIndex` (`keyword_search`), and `DocumentStore` (`upsert`/`count`/`delete`/`get`) — and a host adapts an existing search engine to a Protocol by implementing its methods, with no ragnor inheritance. Five backends ship in the box — `memory`, `pgvector`, `qdrant`, `chroma`, and `faiss` — each in its own module, imported lazily by the factory so its driver dependency stays optional until the backend is selected. Any other vector database plugs in through the registry.

**Session ownership rule:** when the caller supplies a `session`, the caller owns the transaction (commit/rollback). When omitted, the store opens its own session and commits where the operation is transactional.

### Choosing a store

| Need                                        | Choose                             | Lexical capability                          | Why                                                                                                                       |
| ------------------------------------------- | ---------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Tests, zero-dependency dev                  | `InMemoryVectorStore` (`"memory"`) | in-process fallback (token overlap)       | Brute-force **L2 or cosine** + token-overlap keyword approximation; same RRF merge; nothing persisted                     |
| Production on PostgreSQL                    | `PgVectorStore` (`"pgvector"`)     | **native FTS** (`ts_rank`, production-scale) | Real pgvector distance (`<->` L2 or `<=>` cosine) + `ts_rank` full-text search                                            |
| Production vector search on a Qdrant server | `QdrantStore` (`"qdrant"`)         | in-process fallback (token overlap)       | Qdrant's native nearest-neighbor search + in-process token-overlap keyword stage, merged with RRF (L2 only)               |
| Embedded or server Chroma                   | `ChromaStore` (`"chroma"`)         | in-process fallback (token overlap)       | Chroma's native vector search + in-process token-overlap keyword stage; ephemeral or persistent (L2 only)                 |
| In-process exact vector index               | `FaissStore` (`"faiss"`)           | in-process fallback (token overlap)       | Exact `IndexFlatL2` search over an in-process `IndexIDMap`; no server, nothing persisted (L2 or cosine)                   |
| Production on another vector DB             | Your own `VectorStore` subclass    | —                                           | Register it and select via `settings.vector_store_provider` — see [Extending ragnor](../README.md#extending-ragnor-provider-registries) |

The in-memory, Qdrant, Chroma, and FAISS stores' keyword score is a **token-overlap ratio, not a real `ts_rank`** — expect ranking differences versus the pgvector backend, so never tune production thresholds against it. Their `LexicalIndex` implementation is an **in-process fallback** (a full-corpus token-overlap scan) intended for local development, small corpora, and testing; implementing `LexicalIndex` does not guarantee production-scale keyword performance. pgvector is the only built-in **native lexical index** (PostgreSQL full-text search). Hosts at scale combine a pgvector `VectorIndex` with an external lexical engine adapted to `LexicalIndex` — a first-class composition the retrieval strategies (`ragnor.retrieval.strategies`) support out of the box.

### Distance metrics

Every store's vector stage ranks by one distance metric: **`l2`** (Euclidean, the historical default) or **`cosine`** (`1 - cosine_similarity`, in `[0, 2]`). The metric is selected per store — the `metric` constructor argument wins over the optional `rag_distance_metric` host settings field, which wins over the `"l2"` default; values are case-insensitive:

```python
from ragnor.stores import get_vector_store, resolve_distance_metric

store = get_vector_store(
    table_name="policy_chunks",
    embedding_dim=1536,
    session_factory=my_session_factory,
    embedding_fn=embedder,
    metric="cosine",  # or settings carrying rag_distance_metric
)

resolve_distance_metric("COSINE")  # "cosine"
resolve_distance_metric(None, my_settings)  # reads rag_distance_metric
resolve_distance_metric(None)  # "l2"
```

Cosine support per built-in backend:

| Backend     | Cosine | How                                                                                                   |
| ----------- | ------ | ----------------------------------------------------------------------------------------------------- |
| `memory`    | yes    | True cosine similarity (`ragnor.metrics.cosine_distance`) over the in-process records                    |
| `pgvector`  | yes    | The `<=>` operator; needs no pre-normalization. Match the index opclass (`vector_cosine_ops`) — DDL is host-owned |
| `faiss`     | yes    | Vectors and queries are normalized, then ranked on the existing `IndexFlatL2`; the reported distance is `l2² / 2`, exactly `1 - cosine_similarity` |
| `qdrant`    | no     | L2 only; constructing it with `metric="cosine"` raises `ValueError`                                    |
| `chroma`    | no     | L2 only; constructing it with `metric="cosine"` raises `ValueError`                                    |

Two conventions matter when you switch a store's metric:

1. **Thresholds are distances, and distances are metric-relative.** `rag_distance_threshold` keeps its meaning ("smaller is stricter") under both metrics, but a value tuned under `l2` means something different under `cosine` — convert a cosine *similarity* cutoff `s` to the distance `1 - s` and re-tune against a labeled query set (see [Evaluation](../README.md#evaluation)) before serving.
2. **The index must match the metric.** ragnor never creates indexes; a pgvector cosine deployment indexes with `vector_cosine_ops` and an L2 deployment with `vector_l2_ops`. Searching with `<=>` against a `vector_l2_ops` index still computes correctly, but without the HNSW acceleration.

`get_vector_store` forwards `metric` to the selected backend only when it is not `None`, and forwards the `settings` object only to backends whose constructor declares an explicit `settings` parameter — so a custom registered backend built on the documented registry kwargs keeps working unchanged.

### In-memory store

For tests and zero-dependency development — **not** a production backend. It accepts the same constructor kwargs as the pgvector store for drop-in compatibility; the database arguments are ignored. The vector stage ranks by the store's metric (`l2` by default, `cosine` opt-in — see [Distance metrics](#distance-metrics)), so a cosine host can pre-tune thresholds against it locally. Pass `embedding_fn` to auto-embed queries when `hybrid_search` is called with an empty embedding.

### pgvector store (built-in)

The built-in production backend for PostgreSQL tables with `vector` and `tsvector` columns. It requires the `pgvector` extra (`pip install -e ".[pgvector]"`), plus `table_name`, `embedding_dim`, and a host-owned `session_factory`. The optional `tsquery_config` (default `"english"`, a lowercase identifier) must match the tsvector column's configuration.

```python
from ragnor.stores import PgVectorStore

store = PgVectorStore(
    table_name="policy_chunks",
    embedding_dim=1536,  # required
    session_factory=my_session_factory,
    embedding_fn=embedder,  # auto-embed queries
)
```

The pgvector store keeps its optimized single-query `hybrid_search` (one round-trip, two SQL CTEs fused in the database). It also exposes the capability methods standalone: `vector_search` (vector CTE ordered by the store's distance expression — `embedding <-> :embedding` under the default `l2` metric, `embedding <=> :embedding` under `cosine`, see [Distance metrics](#distance-metrics) — with the distance threshold and equality filters as bound parameters), `keyword_search` (FTS CTE ordered by `ts_rank`), plus `delete` (`DELETE ... WHERE id = ANY(:ids)` and/or equality-filter predicates) and `get` (`SELECT ... WHERE id = ANY(:ids)`).

### Qdrant store (built-in)

Production vector search on a Qdrant server. Requires the `qdrant` extra (`pip install -e ".[qdrant]"`), plus `table_name` (the collection name) and `embedding_dim`. The client is created lazily from an explicit `url`/`api_key` on first use, or injected directly via `client` for hosts that manage their own.

```python
from ragnor.stores import QdrantStore

store = QdrantStore(
    table_name="policy_chunks",  # Qdrant collection name
    embedding_dim=1536,  # required
    url="http://localhost:6333",
    api_key="...",  # optional, for authenticated servers
    embedding_fn=embedder,  # auto-embed queries
    # client=my_client,               # or inject a pre-built AsyncQdrantClient
)
```

Qdrant identifies points by unsigned integers or UUIDs, so string document ids are mapped deterministically to `uuid5` point ids (the original id is kept in the payload under `id_field`). The vector stage uses Qdrant's native nearest-neighbor search (Euclid distance); the keyword stage scrolls the collection's payloads and scores token overlap in process, because Qdrant has no built-in full-text rank. Both stages are ranked independently and merged with RRF, so the result shape matches the other backends. Cosine distance is not supported by this backend — see [Distance metrics](#distance-metrics).

### Chroma store (built-in)

Embedded or server Chroma. Requires the `chroma` extra (`pip install -e ".[chroma]"`), plus `table_name` (the collection name) and `embedding_dim`. The client is created lazily on first use — an `EphemeralClient` by default, or a `PersistentClient` when `path` is given — or injected via `client`. The store talks to Chroma's synchronous client API and wraps every call in `asyncio.to_thread` so the event loop is never blocked.

```python
from ragnor.stores import ChromaStore

store = ChromaStore(
    table_name="policy_chunks",  # Chroma collection name
    embedding_dim=1536,  # required
    embedding_fn=embedder,  # auto-embed queries
    # path="./chroma_db",             # PersistentClient; omitted -> EphemeralClient
    # client=my_client,               # or inject a pre-built client
)
```

The vector stage uses Chroma's native nearest-neighbor search (L2 distance); the keyword stage scores token overlap in process over the collection's documents, because Chroma has no built-in full-text rank. Both stages are ranked independently and merged with RRF. Cosine distance is not supported by this backend — see [Distance metrics](#distance-metrics).

### FAISS store (built-in)

In-process exact L2 index — no server, nothing persisted. Requires the `faiss` extra (`pip install -e ".[faiss]"`), plus `embedding_dim`. The index is created lazily on first use — a `faiss.IndexIDMap` over `IndexFlatL2` — or injected via `index` for hosts and tests that manage their own.

```python
from ragnor.stores import FaissStore

store = FaissStore(
    table_name="policy_chunks",  # logical name (logging only)
    embedding_dim=1536,  # required
    embedding_fn=embedder,  # auto-embed queries
    # index=my_index,                 # or inject a pre-built IndexIDMap
)
```

FAISS identifies vectors by 64-bit integers, so string document ids are mapped to consecutive int64 ids (and back) inside the store; the original ids are returned in results. FAISS cannot filter natively, so the vector search over-fetches candidates (10× `n_results`) and filters them in process; the keyword stage scores token overlap over the in-process record map, because FAISS has no text search. Both stages are ranked independently and merged with RRF. Under the `cosine` metric the vectors and queries are normalized to unit length first and the reported distance is `l2² / 2` — exactly `1 - cosine_similarity`, so FAISS cosine ranking matches the memory and pgvector cosine paths (see [Distance metrics](#distance-metrics)).

All backends return result dicts with the same keys: `id`, `document`, `metadata`, `distance` (vector hits only), `keyword_score` (keyword hits only), and `_rrf_score`. They also share the same behaviors: `strict=True` raises `RetrievalError` instead of swallowing failures, an `embedding_fn` auto-embeds queries when `hybrid_search` is called with an empty embedding, and `**filters` applies equality filters to both search stages.

Every backend also implements the `DocumentStore` lifecycle: `delete(ids=None, filters=None, session=None) -> int` and `get(ids=None, filters=None, session=None, text_field="text", metadata_fields=None) -> list[dict]`. `delete` requires at least one selector (an empty `ids` list is rejected — an empty id list never means "delete everything"), applies both restrictions as a conjunction when both are supplied, and returns the number of deleted documents. `get` returns `id`/`document`/`metadata` dicts (score keys are meaningless for direct fetches) and returns every document when no selector is supplied.

### Strict mode

By default, store failures are logged and swallowed into an empty result list (or `0` for `count`) — the historical behavior. Pass `strict=True` to raise `ragnor.types.RetrievalError` instead, which you want for fail-fast pipelines and tests. The opt-in swallow contract lives only in the legacy `hybrid_search`/`count` and in the retrieval strategies: the capability methods (`vector_search`, `keyword_search`) and the lifecycle operations (`delete`, `get`) have no `strict` parameter and always raise on failure.

Stores are selected through `vector_store_registry` via `get_vector_store(...)`, which reads `settings.vector_store_provider` (default `"memory"`) and instantiates the provider with the remaining kwargs. To use another vector database, register your backend under a name and set `vector_store_provider` to it — no ragnor code changes. `get_vector_store` also accepts `metric="l2" | "cosine"` (forwarded to the selected backend only when it is not `None`, so backends built on the documented registry kwargs are untouched), and forwards the `settings` object to backends whose constructor declares an explicit `settings` parameter, so they can read their optional host fields.

---
