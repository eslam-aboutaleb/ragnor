# SDK pull-request / acceptance audit (2026-10-23)

This audit compares the open SDK/clean-architecture PRs #28–#41 to the issue
requirements (#20, #21, #23, #24, #26 and #27), the repo's
`.github/pull_request_template.md`, and GitHub Actions evidence. It records
**observed scope and limitations**; it is not a production-readiness signoff.

## Duplication and merge plan

**Do not merge all 14 PRs independently.**

| Review branch | Scope | Merge handling |
|---|---|---|
| #28 -> #31 -> #34 | Pure SQL ports, SDK wheel smoke and real PostgreSQL SQL contracts | Merge in this dependency order only if choosing granular PRs |
| #29 -> #33 | Pluggable builder/security checks, async ownership/lifecycle | Merge in this order if choosing granular PRs |
| #30 -> #32 | Adapter protocols/contract suites | Merge in this order if choosing granular PRs |
| #35 | Combined qualification of #28–#34 | **Verification-only; do not merge with component stacks** |
| #36 | New public import contract | Feature-specific but currently includes #35 ancestry in its `main` diff |
| #39 | PEP 561 public API typing | Depends on #36; package metadata overlaps #37 |
| #37 | Exact-token chunker/evidence policy | Feature-specific but includes #35 ancestry |
| #38 | PostgreSQL outbox/lease claims/recovery | Feature-specific but includes #35 ancestry |
| #40 | Production startup/dimension validation | Builds on #29/#33; shared `builder.py` requires reconciliation |
| #41 | Superset integrated qualification of #28–#40 | **Alternate single merge candidate. Never merge alongside those PRs or #35** |

**Recommended merge decision:** If no separate feature-level approval/merge requirement
exists, merge **only #41** after independent security/concurrency review and
remaining release-critical acceptance work. Close/mark superseded the embedded
component PRs after that merge. Alternatively, merge the original component
stacks in dependency order and close #35/#41 without merging them.
Do not rebase a component onto `main` and merge it again if the resulting
content has already landed through #41.

The head of #41 passed [full combined Actions CI (27/27)](https://github.com/eslam-aboutaleb/ragnor/actions/runs/39932871791)
and [SonarCloud](https://github.com/eslam-aboutaleb/ragnor/actions/runs/39932871184).
A clean GitHub Actions run is **not** a substitute for independent review or
a test run on the resulting final `main` commit.

## Requirement-to-test traceability and gaps

| Issue | Implemented and verified in current PRs | Test evidence | Remaining / merge blockers |
|---|---|---|---|
| #20 / SDK-01 | Lazy `ragnor` aliases, import isolation, old import identity | `tests/test_sdk_public_api.py`, isolated SDK consumer on 3.11/3.12 wheel/sdist | Public signature compatibility snapshot and broader downstream adapters; docs/release verification |
| #21 / SDK-02 | Explicit production dependencies, fail-closed capability and authorizer checks, owned async close, dimension validation | `tests/test_production_capability_security.py`, `tests/test_production_authorizer.py`, `tests/test_builder_resource_lifecycle.py`, `tests/test_sdk_production_startup.py` | **P0:** real injected vector+SQL-backed lifecycle/rollback/authorization test, not only test doubles; direct low-level resource access outside facade close tracking |
| #23 / SDK-04 | `py.typed`, sdist/wheel marker checks, installed-wheel strict mypy and no new core dependencies | `tests/test_typed_package.py`, `tests/typing/sdk_consumer.py`, CI typing/consumer matrix | Optional extras install/feature matrix, release metadata, publishing/licensing issue #19, broad dynamic adapter typing |
| #24 / architecture | SQL port separated from SQLAlchemy, selective AST checks, host adapter substitution, ownership/injection | `tests/test_architecture_contracts.py`, `tests/test_external_adapter_conformance.py`, `tests/test_extended_adapter_conformance.py`, clean wheel consumer | **P0:** audit ALL production port injection/capabilities and real tenant/security enforcement; selected AST paths and fake adapters cannot prove universal pluggability |
| #26 / SDK-06 | PostgreSQL atomic `SKIP LOCKED`, generation/lease-fenced acks, tenant scoped enqueue, real independent-process race and kill/reclaim tests | `tests/integration/test_postgres_job_outbox.py`, `tests/integration/test_postgres_process_outbox.py`, `tests/test_job_heartbeat.py`, `tests/test_job_error_sanitization.py` | **P0:** idempotent staged external vector write/active-generation switch, replay after provider-upsert/before-ack, forced DB timeout/disconnect, migration, performance baseline; exactly-once remote effects NOT proven |
| #27 / SDK-07 | Lossless injected token chunker, strict mode, tiktoken extra, approximation warning, snapshot fingerprint | `tests/test_token_aware_chunker.py`, `tests/integration/test_tiktoken_adapter.py`, `tests/test_evidence_exact_policy.py` | **P1:** property-based fuzz (malformed/unusual Unicode, long input), full real reindex/retire integration, tokenizer accuracy/performance corpus, whole serialized prompt accounting |

## Regression safeguards actually exercised

- Full inherited Python test suite on Python 3.11 and 3.12: combined #41 CI.
- Existing local word chunker and basic `RagnorBuilder.local()`: regression tests plus
  installed-wheel/sdist consumer, preserving the zero-dependency core.
- Host-owned resources: disposal and cancellation tests; real network clients remain untested.
- Real PostgreSQL: separate connection contention, independent process claims, rollback,
  stale owner detection and kill/reclaim. These are DB guarantees **only**.
- Optional dependency isolation: root API, optional `tiktoken`, and the SQL port.
- Security: explicit authorization tests plus CodeQL; **independent human review pending**.
- Installed downstream type consumer: strict mypy on supported Python versions.

## Release gates and definition of done

- [x] PR scopes and overlaps mapped; individual descriptions aligned to PR template.
- [x] Combined #41 source tree passed CI and SonarCloud on its tested SHA.
- [ ] Select exactly one merge path; do not merge overlapping aggregate+component PRs.
- [ ] Obtain independent review for authorization, lifetime, worker fencing and
  transactional consistency.
- [ ] Close the P0 production-security and external-vector stale-generation risks,
  with real-provider failure/replay tests attached to the corresponding feature PR.
- [ ] Complete remaining P1 requirements or explicitly split/approve them into
  non-blocking follow-up issues; do **not** close original issues prematurely.
- [ ] Rebase chosen merge target onto latest `main`, validate full installed package,
  type, optional adapter, PostgreSQL and security matrix on that exact tree.
- [ ] Run required checks again on the final merge SHA on `main`.
- [ ] Confirm legal license #19 and release metadata/publish mechanism before PyPI release.

A green integration PR ensures the **tested contracts** held in that build.
It cannot establish an absence of undiscovered bugs or all provider-specific
production behaviors.
