# Opt-in page-image retrieval fallback (EX-10)

For pages text extraction cannot represent (engineering drawings, multi-panel
figures, complex forms, scans), a host can route pages to a visual multi-vector
index instead of forcing repeated VLM repair. It is opt-in, provider neutral and
adds no core runtime dependency. No model weights ship with Ragnor.

Modules: `ragnor.retrieval.visual` (core, dependency free) and
`ragnor.stores.qdrant_visual` (optional, needs the `qdrant` extra).

## Contract

* `VisualModel(model_id, revision, dim)`: explicit identity. Every index write
  and query checks it; a mismatch raises `VisualRetrievalError("model_mismatch")`.
* `VisualPage`: immutable page reference (tenant, source, page, private artifact
  ref, size, SHA-256, classification, reason) with normalized/pixel coordinate
  transforms. Public URL schemes (`http`, `https`, `data`, `file`) are rejected
  as artifact refs.
* `VisualEmbeddingProvider` (`embed_pages`, `embed_query`, `remote`),
  `VisualIndex` (`begin`, `stage`, `commit`, `rollback`, `revoke_document`,
  `search`) and `PageAuthorizer` are host ports. Remote providers need
  `allow_remote=True`.
* `VisualIndexer` validates image bytes (hash, size, pixel count, PNG/JPEG
  header) before calling a provider, stages vectors in a generation and commits
  once; any failure, cancellation or out-of-memory rolls the generation back.
* `VisualRetriever.search` returns `VisualEvidence`: page-image location and
  source reference only, never text quotes. Authorization is checked per hit and
  fails closed. `fetch_image` re-checks authorization, size and hash on delivery.
* `VisualRoutingPolicy` routes designated families and unstructurable pages.
  Text representations are kept.
* `fuse_results` merges text and visual hits by (tenant, document, page) with
  empirical-CDF calibration (`ScoreCalibrator`), drops hits for superseded
  revisions, and returns one result per page.
* `derive_interpretation` wraps optional VLM output and flags it `derived`.

## Qdrant

`QdrantVisualIndex` uses Qdrant multi-vector storage with the `MAX_SIM`
comparator (same scoring as `maxsim`), one collection per model id, revision and
dimension, a verified model sentinel, optional int8 scalar quantization and
`on_disk` storage. Staged pages carry `state="staged"` and become visible only on
commit; commit retires older live copies of the same page. pgvector has no
multi-vector MaxSim operator, so it is not offered: keep a Qdrant collection for
visual pages or implement per-patch rows and SQL aggregation in the host.

Commit is two Qdrant operations (publish new, retire old), so replays may briefly
show both copies of a page; search de-duplicates by page id (best score wins). The qdrant-client local mode used in tests
ignores payload indexes; create them on a server (the adapter tries).

## Benchmark

`python -m benchmarks.visual_retrieval` builds text-free raster drawings and
reports Recall@K, MRR, latency, estimated index memory, patch storage cost and
false positives for text-only, visual and fused retrieval, and accepts extra
user-defined `(query, expected_page_id)` pairs. The bundled embedder is a
deterministic fake: it validates plumbing and metrics, not real model quality.
Pass a host provider to `run_benchmark(provider=...)` to measure a real model.
The optional real-model smoke test is gated by `RAGNOR_VISUAL_MODEL_SMOKE`.
