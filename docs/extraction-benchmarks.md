# Offline PDF extraction qualification

This source-checkout benchmark measures text, omission, reading-order and table
fidelity separately. It adds no SDK dependencies and does not register a
production OCR or layout adapter. Acquisition, rendering and provider execution
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
whole-system memory. Cloud cost remains unmeasured.

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

Issue #60 remains open for independent annotation review, Arabic/RTL, handwriting,
real scan families, full chart/formula semantics, downstream retrieval and
permission-aware citation checks, measured provider costs and reviewed release
thresholds. Results are in [the aggregate qualification summary](qualification/pdf-quality-results.md).

## No benchmark data in Git

PDFs, images, ground-truth annotations, manifests and raw JSON reports stay under
ignored `benchmarks/data/`. Historical local fixtures and raw qualification JSON
are also ignored. Only acquisition/scoring code, generated test recipes and
aggregate Markdown results are committed. `scripts/check_benchmark_payloads.py`
checks the Git index and CI rejects committed data payloads.
