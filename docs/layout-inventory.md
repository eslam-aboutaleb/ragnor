# Confidence-triggered page inventory and layout reconciliation (EX-07)

`ragnor.ingestion.layout` lets a host ask a vision model *where* things are on
a page, only when the parser's layout looks wrong, and turns the answer into a
deterministic `LayoutPlan`. It is off by default. The base install has no
model, network or imaging dependency (enforced by a subprocess import test and
the inner-layer architecture test).

The model never produces table or cell values. A plan holds geometry, region
IDs and reason codes only. Re-extraction of changed regions is done by the
host's existing extractor through `apply_plan`.

## Flow

1. `proposed_from_outcome(page_outcome)` (or a host-built `ProposedLayout`)
   describes the baseline layout in normalized, upright page coordinates.
2. `assess(layout)` computes suspicion signals: `low_confidence`,
   `overlapping_tables`, `adjacent_tables`, `orphan_caption`,
   `reading_order_conflict`, `unclaimed_ink`, `count_mismatch`,
   `parser_error`. A region with no reported confidence is never treated as
   high confidence; it is reported but only triggers an inventory when
   `SuspicionPolicy(trigger_on_unreported_confidence=True)`.
3. `LayoutInventoryPlanner.plan_page` makes no call for a page without
   signals. Otherwise it renders at bounded DPI/pixels and makes one
   provider call (more only for bounded retries after a timeout or provider
   error).
4. The provider returns JSON that `PageInventory.from_json` validates strictly
   (schema version, fingerprint, image size, box range, object count, output
   size, duplicate or conflicting regions, unknown keys). Boxes from a rotated
   render are mapped back to the upright page.
5. `reconcile` applies fixed rules: split a table box only when two disjoint
   inventory tables each lie inside it; merge stacked pieces only when they
   lie inside one inventory table, are horizontally aligned, nearly touching
   and have the same column count (if known); add objects the baseline
   missed; tag repeated headers as `repeated_header` artifacts instead of
   dropping them; keep unconfirmed regions as `uncertain`. Nothing is deleted.
6. `LayoutPlan` lists each region with its action, reason codes, source
   region IDs and boxes, inventory region IDs and a crop box (box plus margin,
   clamped to the page). `changed_region_ids` names the regions to re-extract.
   `LayoutPlan.validate_invariants()` checks unique IDs, crop containment,
   provenance and that no source region disappeared.

Region IDs are hashes of page, kind, geometry and parent IDs, so they do not
depend on provider ordering. The plan for shuffled inventories is identical.

## Opting in

```python
from ragnor.ingestion.layout import (
    InventoryBudget, LayoutInventoryPlanner, LayoutInventoryPolicy,
)

policy = LayoutInventoryPolicy(
    enabled=True,
    allow_remote_upload=False,  # a provider with remote=True is refused unless True
    on_failure="baseline",      # or "needs_review"
    max_concurrency=2,
    budget=InventoryBudget(max_calls=5, max_cost=1.0, timeout_seconds=30,
                           max_retries=1, max_dpi=150),
)
planner = LayoutInventoryPlanner(provider, image_source, policy)
results = planner.plan_pages(layouts)   # shared per-document budget ledger
```

`provider` implements `PageInventoryProvider` (`id`, `version`, `remote`,
`cost_per_call`, `inventory(page_image, proposed_layout, budget)`).
`image_source` implements `PageImageSource.render(page, max_dpi, max_pixels)`.
Ragnor ships neither a VLM provider nor a renderer for this port.

Failures are typed and content-free (`external_upload_denied`,
`budget_exceeded`, `timeout`, `cancelled`, `provider_failed`, `image_limit`,
`invalid_schema`, `output_limit`, `object_limit`, `fingerprint_mismatch`,
`image_mismatch`, `conflicting_regions`). Logs carry page numbers, status and
codes only. A timed-out provider thread cannot be killed from Python; the
result is discarded and the budget stays charged.

## Metrics

`ragnor.ingestion.layout.metrics.layout_disagreement(plans)` reports pages
disagreed, splits, merges, added, tagged, uncertain and fallback counts.
`segmentation_f1(truth, predicted)` scores box agreement for before/after
comparisons.
