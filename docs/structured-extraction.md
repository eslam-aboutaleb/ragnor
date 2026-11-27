# Structured extraction contracts (EX-03)

`ragnor.extraction` is a dependency-free, provider-neutral model for OCR,
layout, tables and VLM output. Core never imports an engine, never discovers
providers on its own and never sends content to a remote service. Hosts and
third-party packages plug in through public ports.

## Model (`ragnor.extraction.types`)

| Type | Purpose |
| --- | --- |
| `BoundingBox` | Box plus page width/height, rotation, unit and coordinate space. |
| `Region` | Typed region (`text`, `heading`, `table`, `figure`, `formula`, `caption`, `header`, `footer`, `footnote`, `sidebar`) with explicit `order`, page, `section_path`, caption/continuation links, assets, method and optional confidence. |
| `Table` / `Cell` | `rowspan`, `colspan`, `header_scope`, exact `text` (`None` = missing), optional cell `bbox` and `source_span`. |
| `Page`, `DocumentProfile` | Page frame and status; cheap source facts. |
| `ExtractionAttempt`, `ExtractionResult` | Per-provider attempts (codes only) and the document result. |

Invariants (enforced in constructors, covered by tests):

* The canonical space is `pdf-points-top-left`: origin top-left of the
  unrotated page, y downwards. Bottom-left input converts with
  `BoundingBox.to_top_left()`; `to_display()` gives the rotated view readers see.
* Every region's box frame must equal its page frame; region ids and orders
  are unique per page; region ids are unique per document.
* Confidence is `None` unless a provider supplied one. Nothing is invented.
* Cell values are stored exactly as parsed. Missing cells stay `None`, and grid
  positions no cell covers are reported by `Table.uncovered_positions()`.
* Model-derived text (`DerivedContent`) is never verified source text, and a
  chart estimate can never be marked verified.
* `ExtractionResult.to_dict()` / `to_json()` / `from_json()` round-trip
  losslessly with a schema version; unknown versions fail closed.

## Ports and registry

`ports.py` defines async protocols: `PageProfiler`, `PageRenderer`,
`LayoutDetector`, `OCREngine`, `RegionExtractor`, `DocumentExtractor`,
`Verifier`, `ArtifactStore`. Providers own heavy imports, credentials, network
and lifecycle. Cancellation/timeout contract: methods are cancellation-safe,
Ragnor wraps each call in `asyncio.wait_for`, and `aclose()` is awaited once
per instance after success, timeout, error or cancellation.

```python
registry = ProviderRegistry()
registry.register("acme", factory, formats=["application/pdf"],
                  capabilities=[Capability.TEXT, Capability.STRUCTURE], version="1")
result = await registry.extract(data, revision=sha, name="a.pdf", mode="structured")
```

* Modes: `text`, `structured`, `auto` (structured providers first, then text).
* Selection order is `(priority, provider_id)`, deterministic, and fails closed:
  `unsupported_format`, `incompatible_capabilities`, `remote_not_enabled`.
* `remote=True` providers are skipped until the host calls `enable_remote(id)`.
* Provider failures surface as fixed codes, never provider messages.
  If every provider fails, `ExtractionError("fallback_exhausted")`; partial
  output raises `partial_only` unless `accept_partial=True`.
* `load_entry_points()` (group `ragnor.extraction_providers`) is explicit
  opt-in discovery of installed third-party packages. See
  `examples/custom_extraction_provider.py`.

## Serializers and legacy compatibility

* `to_json`/`from_json`: lossless. `table_to_html`: faithful, escaped, spans
  and missing cells marked. `table_to_markdown`: only when lossless, else `None`.
  `table_to_text`/`region_text`/`result_to_text`: lossy retrieval projections.
  All table projections are bounded (`ExtractionLimitError`, never silent truncation).
* `to_raw_documents` projects a result onto `RawDocument` (per region or per
  page) with page, region id, bbox and revision in `metadata`/`location`, so
  the existing `IngestionPipeline` and chunkers keep working unchanged.
* `from_page_outcomes` lifts legacy `PageExtractionOutcome` evidence.
* `Extractor.extract`, `extract_pages` and `PDFExtractor` are untouched.
  Structured mode is opt-in through `StructuredExtractor`.

## Fingerprints and reindexing

`StructuredExtractor.options` carries the mode and, for each provider, its
registered id, version and *safe* options (no prompts, credentials, URL
userinfo, long free text or bytes). It feeds `build_snapshot`: a changed
provider version, option, recipe/model value or mode changes the snapshot and
forces a re-index; an unchanged configuration compares equal.

## Page profiling and routing

`ragnor.extraction.routing` (zero dependencies) decides, per page, which
capability reads it. `PageProfile` holds cheap facts: kind (`digital`, `scan`,
`mixed`, `blank`, `unknown`), `/Rotate`, dominant text direction, text length,
image/drawing/annotation flags and a text-column hint. `route_page` maps a
profile plus `Available` capabilities (built from registered `ProviderSpec`s)
onto a `RouteDecision`: `native`, `layout`, `ocr`, `skip_blank` or
`unroutable`.

* Pure and deterministic: same profile, capabilities and policy give the same
  decision. No model or network call. A remote provider is used only with
  `RouteOptions(allow_remote=True)`; a local one always wins.
* Cost order is text, layout, OCR. A scan needs OCR. A mixed page (text plus
  images, drawings or annotations) prefers layout, then OCR, and uses native
  text only when nothing else is configured.
* Nothing is dropped. With no provider able to read a page the decision is
  `unroutable` and names the missing capability; only a verified blank page is
  `skip_blank`, and it is still reported. `plan_document` returns exactly one
  decision per page.
* Sideways or upside-down text is recorded (`visual_orientation`) and flagged
  `bbox_requires_rotation_transform` when boxes need the rotated frame.

`ragnor.ingestion.pdf_profiler.PdfPageProfiler` (`ragnor[pdf]`) builds
profiles from the PDF structure and text-positioning operators with pypdf. It
renders nothing, implements the `PageProfiler` port and also exposes
`profile_pages`. Limitations: the column hint counts left-edge clusters of
upright text lines, so it is a hint, not layout analysis; it does not decode
rotated images inside scans.

## Layout, tables and citations (EX-05)

Everything below is pure Python on top of the EX-03 model. No layout engine,
VLM or network call ships in core.

* **Regions** (`layout.RegionNormalizer`, `normalize_docling`): provider items
  (label, box, optional grid/latex/text) become typed regions. Labels map to
  `RegionKind`; unknown labels fall back to `text` unless `strict_labels` is set.
  Bottom-left boxes are converted to the canonical top-left space. Region ids
  come from `deterministic_region_id(revision, page, key)`, so they are stable
  for one document revision and change with the revision.
* **Reading order** (`reading_order.reconcile_reading_order`): a constrained
  XY-cut with RTL and rotation support. Headers, footers, footnotes and sidebars
  are tagged and kept, never dropped. When the cut is not clear (overlap, or a
  gap smaller than the threshold) every affected region is marked
  `order_state=ambiguous` instead of guessed.
* **Tables** (`tables`): `table_from_grid` (rule-based output, ragged rows
  rejected, `None` stays missing), `table_from_cell_dicts` (spans, header flags,
  Docling offsets), `split_table_at`, `stack_tables`. Detected boxes are merged
  only when the provider declares the group (`merge_declared_table_boxes`);
  similarity is never used. `link_table_continuations` links a table across
  pages when the next page's first table repeats the header or the provider
  declared it; `logical_table` stacks the chain as a read-only view.
* **Captions** (`figures.link_captions`): nearest clear figure or table, above
  or below. Ties and orphans are reported as ambiguous and left unlinked.
* **Citations** (`citation`): `to_raw_documents` stamps revision, page, region id
  and bbox on every chunk. `resolve_citation` maps a retrieval hit back to the
  region, and to one cell when `cell_text` matches exactly one cell. A revision
  mismatch fails closed.
* **Routes** (`routes.apply_routes`): per-kind provider routes through the
  `RegionExtractor` port. Unknown providers and remote providers that were not
  enabled fail before any data is sent; a failing route keeps the base region.
* **Fidelity** (`fidelity`): `teds_like` (a sequence approximation, not the
  published TEDS), cell accuracy, exact-cell accuracy, order accuracy, region recall.

Limits: Docling input is tested against a docling-core-shaped dict, not a live
Docling run. Fidelity tests use EX-02 annotation-derived inputs, not a live
layout model's output.
