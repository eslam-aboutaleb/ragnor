# Offline PDF extraction qualification

This source-checkout benchmark measures text, omission, reading-order and table
fidelity separately. It adds no SDK dependencies and does not register a
production OCR or layout adapter by itself. The SDK now provides separately
configured [runtime adapters](pdf-extractors.md); benchmark adapters remain
measurement tools. Acquisition, rendering and provider execution
happen only through explicit commands.

## Reproduce

```sh
python -m pip install -e '.[pdf]'
python -m pip install -r benchmarks/requirements.txt
python -m benchmarks.quality_corpus --output benchmarks/data/dpbench-full
python -m benchmarks.run_quality \
  --manifest benchmarks/data/dpbench-full/quality-manifest.json \
  --provider pypdf --output benchmarks/data/dpbench-full/pypdf-quality.json
python -m benchmarks.run_quality \
  --manifest benchmarks/data/dpbench-full/quality-manifest.json \
  --provider pdfplumber --output benchmarks/data/dpbench-full/pdfplumber-quality.json
# Install the host Tesseract executable and English language data for this run.
python -m benchmarks.run_quality \
  --manifest benchmarks/data/dpbench-full/quality-manifest.json \
  --provider tesseract --cohort common \
  --output benchmarks/data/dpbench-full/tesseract-quality.json
```

Run providers sequentially when comparing latency. The benchmark tools support
Linux/macOS. Tesseract is optional and is invoked locally without a shell; these
commands send no PDFs to cloud providers. Python package versions are pinned in
`benchmarks/requirements.txt`; reports also record actual provider/executable
versions, options, Python/platform, corpus identity and deadline. Native executable
versions and hardware still need to match for performance comparisons.

## Corpus and lineage

Acquisition uses the entire pinned 200-page [DP-Bench](https://huggingface.co/datasets/upstage/dp-bench)
source corpus, rather than selecting apparently successful pages. The reference
JSON digest and upstream Git LFS SHA-256 hashes validate downloaded PDFs, including
cached copies. A local manifest records every input hash, source URL, reference
page/region IDs, element sequence, normalized boxes, table HTML and source text.
The dataset card declares MIT; source data retains its upstream terms.

The 200 pages contain 55 table regions, 67 charts, 58 equations and 804 paragraphs,
plus headings, figures, lists, captions, headers, footnotes and indexes. Ten source
pages with the highest table cell/span complexity generate four variants each:
200-DPI raster scans, 100-DPI grayscale/JPEG-quality-35 scans, 90-degree PDF
rotation, and a two-page digital-plus-scan document. This gives 240 documents /
250 page instances, with 200 independent source pages. Derived pages are correlated
and are reported separately by family. Their text/structure references come from
the original annotations, never from the provider being evaluated. They are
controlled degradation tests, not a representative real-scan distribution.

The `common` cohort contains those ten original pages and all forty variants:
50 documents / 60 page instances. The native provider reports can be filtered to
exactly this cohort before comparing with the OCR run. Never compare the full
public native aggregate with the smaller OCR aggregate as if they were the same
workload.

Source polygons use the MediaBox rendered at 200 DPI, converted to normalized
coordinates with top-left origin. All boxes are checked against page bounds;
four rendered overlay spot checks covered tables, charts, equations and two-column
prose. Rotation transforms boxes; mixed documents retain page order. This is not
independent manual approval of every annotation or transcription.

## Metrics and failure accounting

- Page CER/WER after whitespace normalization, multiset token recall and strict
  numeric-token recall. Numeric metrics compare lexical values, not units or
  semantic equivalence. Page CER also reflects serialization order.
- Region token recall and empty reference text regions, using predicted word
  centers inside reference boxes. Repeated tokens consume their occurrence counts
  within each region. Charts/formulas without text labels are not certified.
- Accuracy of upstream element-order pairs, accompanied by pair coverage. Detecting
  two correctly ordered regions while losing the remaining column cannot report
  full order coverage.
- Table recall/precision at declared IoU 0.5, with deterministic greedy one-to-one
  matching. Unmatched reference tables contribute zero to structural/cell scores.
- Canonical table TEDS and structure-only TEDS using APTED, plus row/column-aligned
  exact cell values, span accuracy and numeric-cell accuracy. `th`/`td` and table
  section wrappers are canonicalized and inline formatting is reduced to text.
  These are explicitly normalized TEDS variants, not official leaderboard parity.
  The metric follows the [TEDS tree-edit definition](https://arxiv.org/abs/1911.10683).

Failed pages remain in capable providers' denominators. Invalid predictions get
zero; invalid references are counted separately. Unsupported layout/table ports
produce null scores with explicit capability counts. Thus a text-only parser
cannot claim table accuracy, and dropping a hard page cannot improve its score.

Each provider runs in a spawned process with a configurable per-document deadline
(default 60 seconds). POSIX timeout cleanup terminates its process group, including
OCR children. Raw child stdout/stderr and logs are suppressed; reports contain IDs,
fixed failure codes and scores, not document text. Latency includes process startup
and IPC, excludes metric scoring, and is reported as document p50/p95. Python
provider RSS and OCR subprocess RSS are recorded separately, not summed or called
whole-system memory. Cost accounting is described below.

## Regression checks and remaining acceptance

```sh
python -m benchmarks.compare_quality \
  benchmarks/data/baseline.json benchmarks/data/candidate.json \
  --output benchmarks/data/comparison.json --tolerance 0
python -m pytest -q tests/test_pdf_quality_metrics.py tests/test_pdf_benchmark.py
```

Comparison requires the same manifest/cohort/family coverage and fails when
measured baseline capabilities disappear, a family regresses, or provider failures
increase. Tolerance is explicit and defaults to exact non-regression. This is a
baseline-relative tool, not an independently approved production accuracy bar.
Tests use hand-computed wrong-value, row swap, lost merge, omitted column, missing
table, provider timeout and corrupted-manifest cases. Small regression PDFs are
generated in temporary directories; CI never fetches the public corpus.

Issue #60 remains open for independent annotation/family review, measured chart/formula
provider semantics, natural-question RAG validation, supplied host prices and approved
release thresholds. V3 adds Arabic scans/handwriting and SDK retrieval/citation contracts
as described below. Results are in [the aggregate qualification summary](qualification/pdf-quality-results.md).

## No benchmark data in Git

PDFs, images, ground-truth annotations, manifests and raw JSON reports stay under
ignored `benchmarks/data/`. Historical local fixtures and raw qualification JSON
are also ignored. Only acquisition/scoring code, generated test recipes and
aggregate Markdown results are committed. `scripts/check_benchmark_payloads.py`
checks the Git index and CI rejects committed data payloads.

## Extended qualification paths (schema v3)

The extended importer adds 20 Arabic document images, ten physical Arabic scans,
and nine unambiguous Arabic manuscript image/transcript pairs. It excludes the
handwriting dataset's shared A/B transcripts where page boundaries cannot be
established. Sources, licenses, pinned revisions and archive/model hashes are
recorded in the local manifest. Original references are retained locally.
These imported transcriptions do **not** supply layout boxes: those pages set
`geometry_ground_truth=false` and layout scores remain unmeasured. See
[source decisions](qualification/public-pdf-sources.md).

```sh
python -m benchmarks.extended_corpus \
  --output benchmarks/data/extended/human-corpus \
  --base-manifest benchmarks/data/dpbench-full/quality-manifest.json \
  --combined-output benchmarks/data/extended/comparison
# This combines the 60-page structural cohort with 39 Arabic pages: 99 pages.
# Repeat this command sequentially with pypdf, pdfplumber, and tesseract.
python -m benchmarks.run_quality \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --provider tesseract --tesseract-language ara+eng \
  --tessdata-dir benchmarks/data/extended/human-corpus/tessdata \
  --timeout 90 --output benchmarks/data/extended/comparison/tesseract-v3.json
python -m benchmarks.qualification packet \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --output benchmarks/data/extended/review-packet --count 50
```

All 50 review records start **pending**. `index.html` presents source renderings,
transcriptions, region overlays, reading order and critical fields. A reviewer
must check alignment, omissions, spans, numeric values, exclusions, ambiguities
and family labels; upstream review provenance is separate from local approval.
PDF image conversion uses a fixed metadata epoch so repeated acquisition produces
stable input hashes. Physical scans are embedded as image PDFs, not mislabeled
as original born-digital PDFs.

V3 adds the following measurements and contracts:

- Class-agnostic text-block precision/recall at IoU 0.5 and omitted region IDs.
  These measure block detection separately from word recall and order.
- Foreground ink coverage uses Otsu thresholding and predicted word boxes.
  Background page area is excluded. Graphics can contribute foreground pixels;
  coverage is a diagnostic proxy and cannot certify transcription correctness.
- Critical-field text exact match, normalized Decimal numeric accuracy with
  explicit absolute tolerances, chart-value scoring, and restricted rational
  formula equivalence. Formula parsing uses a bounded AST without `eval`, supports
  at most two symbols and degree 12, and compares on the common rational domain.
  It does not support arbitrary LaTeX or prove equality at excluded singularities.
- Upstream table cells project deterministically into explicit grid coordinates,
  spans and numeric reference fields. Predicted values are read from the matched
  table at those coordinates. Covered grid positions in merged cells are not
  independent annotations. Missing and wrong values both fail; answer coverage
  is reported separately. Generic chart/formula fields require explicit references
  and provider outputs; the current providers do not implement those ports.
- Observable unsafe completeness claims and their rate among positive claims.
  This is a lower-bound diagnostic: unsupported geometry/semantics and missing
  reference labels cannot reveal every false negative.
- Real SDK chunking, lexical retrieval, evaluation, evidence assembly and source
  lookup with BM25Plus. Recall@K/MRR and answer-source hits use actual expected
  chunk IDs; omitted answers remain misses. Reference-derived anchor queries are
  explicitly contract probes, not human-authored natural-question QA. Optional
  `qa_probes` supply unique queries, reference answers and source page IDs.
- Citation resolution requires exact immutable source spans. Separate probes
  deny another tenant, missing revisions and revoked sources, and verify that a
  retained old revision still resolves after a new revision appears. These SDK
  contract checks do not qualify a production backend's authorization policy.
- Provider CPU time includes completed OCR subprocesses; normalized CPU hours
  and observed external API charges are reported separately. CPU accounting
  excludes process startup and scoring. `--cpu-usd-per-hour` requires
  `--pricing-source`; total USD remains null without a supplied price. No cloud
  calls are made by the bundled providers. Idle/capital costs remain excluded.

`--predictions-out` explicitly enables local raw prediction export; the default
report contains scores and IDs. `--no-rag` disables SDK probes when measuring
extraction alone. P95 uses the nearest-rank definition. Schema v2/v3 results
cannot be compared as if they used the same scoring protocol. Comparison also
rejects changing page identities, losing field metrics, changing retrieval
queries, degrading retrieval/citations, or failing revision/authorization probes.

The packet also writes an unapproved `policy-template.json`. Maintainers must
review its scope, fill its allowlists and baseline hash, and save their approved
policy as `policy.json`; generating the template grants no approval.
The qualification checker consumes these host-owned policy and review files:

```sh
python -m benchmarks.qualification check \
  --report benchmarks/data/extended/comparison/tesseract-v3.json \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --policy benchmarks/data/extended/review-packet/policy.json \
  --reviews benchmarks/data/extended/review-packet/reviews.json \
  --baseline benchmarks/data/extended/comparison/tesseract-v3.json \
  --output benchmarks/data/extended/review-packet/qualification.json
```

Policy requires `approved_by`, `approved_at`, `allowed_policy_approvers`,
`allowed_reviewers`, `minimum_reviewed_pages` (at least 30), `required_families`,
`required_summary_metrics`, `quality_tolerance`, and `baseline_sha256`. The baseline
hash is SHA-256 of `json.dumps(report, sort_keys=True).encode()`. Review records
bind corpus and per-page reference hashes, must be distinct, and require allowlisted
reviewer identities and dates. These are externally supplied attestations in
host-controlled files, not an identity provider or automatically generated approval.
An empty metric policy, absent price, unresolved citations, pending reviews or an
unapproved baseline fails qualification. Baseline tolerance defaults to zero;
production tolerances require maintainers' approval.

## Benchmarking the SDK adapters offline

`--provider sdk-native`, `sdk-ocr` and `sdk-auto` invoke the installed Ragnor SDK
adapters. The benchmark uses the SDK's own bounded subprocess transport rather
than wrapping it in another process timeout. Existing standalone provider names
remain available for comparison. No dataset or model fetching occurs in this
command; OCR requires an already installed executable and local traineddata.

```sh
python -m benchmarks.run_quality \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --provider sdk-auto --cohort common \
  --tesseract-language eng+ara --tessdata-dir benchmarks/data/extended/tessdata \
  --output benchmarks/data/extended/comparison/sdk-auto-v6.json \
  --predictions-out benchmarks/data/extended/comparison/sdk-auto-v6-predictions.json
python -m benchmarks.qualification packet \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --predictions benchmarks/data/extended/comparison/sdk-auto-v6-predictions.json \
  --output benchmarks/data/extended/sdk-review-v6 --count 50
```

The review packet shows SDK text and tables beside source previews and reference
annotations. Each pending review binds the reference and exact SDK evidence hash.
Reference `status` and `extraction_review_status` are separate pending decisions;
approving the reference does not approve SDK output. It does not approve the
reference, extraction or release policy automatically.
After actual review, the host may supply approved extraction evidence hashes to
`EvidenceAllowlistVerifier`; authentication and storage remain host responsibilities.
Re-extraction after changes to source, models, adapter version or evidence requires
fresh approval. The release qualification command still requires an explicitly
approved policy and human reviews; extraction success alone does not satisfy it.

SDK text includes native table HTML. CER/WER score that actual returned text,
including markup, while table structure/value metrics score table HTML separately.
Those text error rates cannot be interpreted as a pure visible-transcription
comparison with standalone providers returning plain text. SDK CPU usage includes
reaped worker descendants; RSS is left unmeasured rather than reporting the
benchmark parent's memory as the worker's. Runs sharing a host with other work
are unsuitable for isolated throughput or latency claims.
