# Extended adapter conformance — scope and evidence

This suite is an incremental deliverable for [SDK architecture #24](https://github.com/eslam-aboutaleb/ragnor/issues/24) and reuses the host adapter fixtures introduced in PR #30.

## Covered contracts

| Component | Out-of-tree implementation | Proven by CI | Not proven |
| --- | --- | --- | --- |
| Extractor / source | HostExtractor + HostSource | Loaded and indexed via existing pipeline | Remote connector pagination/credentials |
| Document lifecycle | HostStorageAdapter | Scoped get/count/delete and upsert through injected backend | Independent durable database semantics |
| Vector/lexical indexes | HostIndex | Independent index injection, filter pushdown, capability validation | Vector search quality / remote engine behavior |
| Reranker | HostReranker | Called within composed HybridRetrieval | Model correctness |
| Sync state / manifests | Host*Adapter wrappers | Port substitution, CAS rejection, lease fencing in memory | Multi-host atomicity or crash recovery |
| Job store | HostJobStoreAdapter wrapper | Port substitution | Distributed atomic job claims; see #26 |
| Scope authorization | Host callback + KnowledgeBridge | Unauthorized scope is rejected *before* retrieval | Exhaustive tenant isolation / transport auth |

These are **host-level test implementations outside `src/ragnor`**, not additional SDK runtime dependencies. They validate the boundary contracts, not production certifications.

## Execution

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_external_adapter_conformance.py tests/test_extended_adapter_conformance.py
```

GitHub Actions runs the same tests as a dedicated required `adapter-contracts` job, plus the shared full Python matrix. The aggregate status check depends on adapter conformance.

## Still required for full #24 closure

- Production lifecycle and ownership checks for injected async clients, pools, cancellation and shutdown (#21).
- Versioned source/extractor/embedding/index fingerprint semantics and key rotation without cache contamination.
- **Real** SQL transactions, concurrent lease acquisition, atomic compare-and-swap and failure injection (#26); memory tests cannot prove multi-host correctness.
- Typed public facade and artifact contract (#20/#23).
- Security and authorization negative E2E tests against supported real backends; no fallback to unfiltered results.
