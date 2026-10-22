# SDK architecture verification matrix

This is the CI-verifiable portion of [#24](https://github.com/eslam-aboutaleb/ragnor/issues/24). It records **what has been tested** and **what remains**, rather than treating a green import check as production proof.

| Contract | Verification | Scope and limit |
|---|---|---|
| Inward import direction | `tests/test_architecture_contracts.py` (PR #28) | Selected core modules; does not prove the entire graph is free from runtime dynamic imports |
| Protocol substitution | `tests/test_external_adapter_conformance.py` (PR #30) | Basic host implementations; not security/durability certification |
| Builder dependency injection | `tests/test_builder_dependency_injection.py` (PR #29) | Sessions, SQL locking adapter, retrieval strategy and sink; durable backend is not simulated as a production guarantee |
| Clean-wheel install | `examples/sdk_consumer_smoke.py` in isolated venv, this PR | Runs actual installed distribution with host embedder/chunker; no editable install or source-path imports |
| Distributed locking | Real PostgreSQL multi-worker tests — issue #26 | Pending; must not claim atomic multi-host processing until validated |
| Token budgets | Exact-tokenizer and snapshot tests — issue #27 | Pending; word-based chunking/approximate evidence remain documented limits |
| Full API/lifecycle | #20, #21, #22, #23 | Additional API and correctness work remains before full issue closure |

The `sdk-consumer` GitHub Actions job builds a wheel, installs it into a fresh environment **without third-party runtime dependencies**, moves outside the source checkout, and executes the script. It verifies installed imports, metadata dependencies, an injected embedder/chunker, ingestion, count and retrieval.

## Architecture release checklist

- [ ] Source dependency gates pass.
- [ ] Pluggable reference adapters pass contracts, including negative cases where needed.
- [ ] Real services test their actual declared capabilities (transactions, locking, filters, tenant isolation).
- [ ] Installed wheel and sdist are qualified on declared Python versions.
- [ ] No source checkout shortcuts mask missing package resources.
- [ ] Documentation identifies ephemeral defaults and does not imply production safety based on protocol conformance.
- [ ] Only close #24 after the port matrix, ownership/capability semantics and CI conformance are complete.

See [ports and adapters](ports-and-adapters.md), [external adapters](external-adapters.md) and [builder composition](builder-composition.md) across companion PRs. These documents remain in separate PRs until integrated.
