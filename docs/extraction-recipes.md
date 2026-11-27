# Document-family extraction recipes (EX-09)

Recipes let repeated PDF layouts (one vendor's invoices, a lab-report template)
skip full layout detection. They are an **optimisation behind verification**:
the fast path never publishes output that did not pass count, coverage and
shape checks. Implement them only after benchmarks show recurring families.

Module: `ragnor.ingestion.recipes`. No third-party dependencies; persistence is
injected by the host.

## Pieces

| Piece | Purpose |
| --- | --- |
| `ExtractionRecipe` | Typed, versioned, tenant-scoped plan: family criteria, saved regions (normalized bbox, provider, DPI, margin), serializer, expected counts, quality thresholds, artifact policy, provenance. JSON round-trip is strict; corrupt data raises `RecipeError("corrupt_recipe")`. |
| `RecipeValidity` | Extractor, extractor version, model, layout schema, options hash, rotation, template version. Any change gives a new `extraction_fingerprint`. |
| `RecipeStore` | Host port: `get`, `put_if_absent`, `compare_and_set`, `deactivate`, `list_revisions`. Revisions are immutable and append-only (audit trail). |
| `InMemoryRecipeStore`, `JsonFileRecipeStore`, `SQLiteRecipeStore` | Reference stores. The file store publishes revisions with an atomic hard link, so concurrent writers get a real CAS. The SQLite store uses `(tenant, family, revision)` as primary key and is the schema template for host SQL. |
| `FamilyClassifier` | Strong-match classifier: exact metadata, first-page anchors, page count, rotation. A weak match or two near-equal matches returns no family. |
| `RecipeFastPath` | classify, look up, apply regions, **verify**, publish. Otherwise exactly one full-detection run. |
| `CountCoverageVerifier` | Default `RecipeVerifier`: every region present, counts in range, table shape, non-empty text. A verifier is a required constructor argument. |
| `RecipeLearner` | Stores a recipe only from pages the host `PageVerifier` accepted, after sample confirmation and (by default) explicit host approval. |
| `RecipeTelemetry` | Per-tenant counters and rates: match precision, false-hit rate, fallback rate, cache hit rate, calls saved. No document content. |

## Fast-path rules

1. No strong family match: normal flow (`no_family`, `ambiguous_match`).
2. No active recipe, or recipe quarantined or deactivated: normal flow.
3. Recipe validity differs from the running engine: recipe is deactivated
   (`stale_validity`), normal flow. Compare `extraction_fingerprint` with
   stored index metadata to schedule re-indexing.
4. Recipe applied, then verified. A failure (or any provider error, reported
   without its message) counts as a false hit, quarantines the recipe after
   `quarantine_after` consecutive failures (default 1), and falls back once.
   There is no retry loop.
5. Cancellation raises `ExtractionCancelled` and never quarantines.

The failure counter used by `quarantine_after` lives in the process; hosts that
need it shared across workers should keep `quarantine_after=1`.

## Learning

`RecipeLearner.learn(draft, verified_pages)` refuses with a fixed
`RecipeLearningRefused.code` (`unverified_run`, `sample_not_confirmed`,
`not_approved`, `concurrent_update`) and stores nothing unless every page is
complete and accepted by the host verifier, the host confirmed a sample and,
when `approval_required`, the host approver returned True. Ragnor never claims
human review on its own.

## Benchmark

`python -m benchmarks.recipe_family` runs synthetic families, including
template-drift documents, through the full path and the recipe path and prints
accuracy, provider calls, p50/p95 latency and throughput. Provider latency is
simulated and the full-path call count is an explicit assumption (detection plus
one extraction and one reconciliation call per region). It proves the guard and
call-count behaviour; real cost and accuracy on production families must be
measured by the host.
