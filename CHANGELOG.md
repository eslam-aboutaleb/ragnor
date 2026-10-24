# Changelog

## Unreleased

- Adopt Elastic License 2.0 (`Elastic-2.0`) for this source revision and future
  releases, preserving project notices. Earlier published artifacts retain
  their original terms. Previously published release 0.4.1 artifacts are unchanged.


All notable changes to ragnor (published on PyPI as `ragnor`) are
documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Stability policy (pre-1.0).** While ragnor is pre-1.0, additive API changes
ship in minor releases and are safe to adopt. Behavioral changes to the
compatibility surface — the `VectorStore` ABC (`hybrid_search` / `count` /
`upsert`), the `RagSettings` protocol, and the capability protocols
(`VectorIndex` / `LexicalIndex` / `DocumentStore`) — require a minor version
bump and a changelog entry here, so hosts pinning `ragnor==x.y.z` get a
stability contract rather than a moving target.

## [Unreleased]

## [0.4.0] - 2026-10-11

### Added

- **Selectable vector-store distance metric** (`l2` by default, `cosine`
  opt-in). Every built-in store accepts a uniform `metric` constructor kwarg,
  hosts can select it declaratively through the optional
  `rag_distance_metric` host settings field, and
  `get_vector_store(..., metric=...)` passes it through to the selected
  backend. The new `ragnor.metrics` module exposes `DistanceMetric`,
  `DEFAULT_DISTANCE_METRIC`, `SUPPORTED_DISTANCE_METRICS`,
  `resolve_distance_metric`, `cosine_similarity`, `cosine_distance`, and
  `normalize`. Under the cosine metric, distance is defined as
  `1 - cosine_similarity`, which preserves ragnor's existing "smaller is
  stricter" threshold semantics.
  - **Memory store** — true cosine similarity (dot product over vector norms,
    with a zero-vector guard) when the metric is `cosine`; the `l2` path is
    unchanged.
  - **pgvector store** — the distance operator comes from an internal map,
    never from user input: `l2` → `<->`, `cosine` → `<=>` (both exact, no
    pre-normalization needed). Threaded through `hybrid_search`,
    `vector_search`, and the constructor. Host-owned DDL must match: a cosine
    index uses `vector_cosine_ops` instead of `vector_l2_ops` (ragnor never
    creates indexes).
  - **FAISS store** — cosine ranking over normalized vectors, with distances
    reported as `l2² / 2` so ranking is identical to cosine.
  - **Qdrant and Chroma stores** — reject the cosine metric at construction
    time with a clear `ValueError`; those backends keep the `l2` default.
  - `rag_distance_metric` is an **optional** host settings field read via
    `getattr`, exactly like the existing `embedding_provider` / `llm_model`
    fields — it is not a new `RagSettings` protocol member, so existing hosts
    are unaffected and keep satisfying `isinstance(settings, RagSettings)`.
  - Thresholds are **metric-relative**: `rag_distance_threshold` values are
    never comparable across metrics, so a host switching metrics must re-tune
    its threshold (a cosine similarity `s` corresponds to a threshold of
    `1 - s`).
- **`LitellmEmbeddingFunction` embedder options** — new `cache_max_entries`
  (bounded LRU cache keyed by model + text; default `0` = off) and
  `max_batch_size` (default `None` = no splitting) constructor kwargs. Both
  default to today's behavior, and results are returned in the original input
  order even with mixed cache hits and misses.
- **LiteLLM provider error classification** — hosts classify provider
  errors as transient or permanent when using these cache and batching options.

[0.4.0]: https://github.com/eslam-aboutaleb/ragnor/releases/tag/v0.4.0
[Unreleased]: https://github.com/eslam-aboutaleb/ragnor/compare/v0.4.0...HEAD
