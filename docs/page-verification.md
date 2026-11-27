# Page verification and bounded region repair (EX-08)

`ragnor.ingestion.verification` makes extraction checks observable and lets a
host retry only failing regions with fixed, bounded actions. The source PDF
stays the source of truth. A repair changes how a region is parsed, never a
value. The package has no network or model dependency.

## Verification

`PageVerificationPipeline.verify(outcome, expected=..., alternate_tables=...,
relations=...)` returns a versioned `VerificationReport` (`verified`,
`failed` or `needs_review`) of typed `Finding`s
(`page, region_id, check_id, severity, observed, expected, source, disposition`).

Checks run on every page they have input for:

- `completeness`: reuses the EX-01 page status. An incomplete page never verifies.
- `count`: expected inventory (`ExpectedPage.regions`) versus extracted tables,
  figures, headings and text.
- `coverage`: each expected foreground content box must be covered by extracted
  boxes. Raw page area is never evidence, so whitespace cannot cause a failure.
- `shape`: table grid and merged spans against an independent parser, when given.
- `arithmetic`: decimal-safe column sums on relationships the host declares
  (`ArithmeticRelation`). A total row with no declared relationship, or an
  amount that cannot be parsed, is a warning. Nothing is ever corrected.
- `render_back` (optional): an expensive image comparison port, called only for
  regions that already have a repairable error, capped per page.

A check named in `VerifierSettings.mandatory` that cannot run, or any check that
raises, yields a `needs_review` finding. It is never skipped silently.
`PipelinePageVerifier` plugs the pipeline into the existing `PageVerifier` gate
used by the PDF extractors; a pipeline crash surfaces as `verification_failed`.

## Repair

`RepairStateMachine.run(tenant_id, run_id, regions, cancel)` repairs each
failing region with at most 3 attempts (configurable lower, never higher),
chosen from fixed allowlisted actions per check: adjust crop, re-segment, raise
resolution, change serializer, tile dense table, change extractor. `RepairAction`
accepts only numeric or short-token parameters (no values or text). The same
action and parameters are never tried twice, DPI and pixels are capped, and total
attempts, cost and time are limited.

Each attempt is written to the host's `AttemptStore` before it runs. A replay
after a crash or cancellation counts earlier attempts, skips actions already
tried and reuses an earlier promotion, so it cannot reset a budget. A region is
promoted only when its re-extracted artifact passes verification and its box
still overlaps the original; otherwise it is `needs_review`.

Audit records (`AttemptRecord`) hold IDs, geometry, check IDs, action,
duration, outcome and an opaque artifact reference only. They are validated to
reject free text, URLs and path traversal. Storage is tenant-scoped and
`AuditReader` enforces per-principal tenant grants. The host owns persistence.

## Publication

`publish_verified` publishes a whole revision or nothing. Under the strict
policy a revision replaces the active one only when every page is `verified`,
reports exist for every page, all used one verifier settings fingerprint and
every required check ran. Commit failures leave the active revision unchanged.

## Evaluation

`python -m benchmarks.pdf_extraction.verification_eval` runs the verifiers over
the EX-02 corpus annotations (as a perfect extraction, plus one injected omitted
region per page). It measures the verifiers, not any extractor.
