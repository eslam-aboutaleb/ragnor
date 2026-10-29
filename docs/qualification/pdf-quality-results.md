# PDF qualification results — 29 October 2026

Actual local runs on Apple M4 / macOS arm64 / Python 3.12.12. Providers ran
sequentially, with a 60-second deadline per document. No provider failures,
missing outcome pages, invalid reference tables or invalid table predictions
were recorded. These observations describe the configurations below; they do
not establish production accuracy or independent annotation approval.

## Corpus and measurements

The pinned full DP-Bench corpus supplies 200 original pages and 55 table regions.
Ten source pages with the highest table cell/span complexity produce 40 derived
documents: clean scans, degraded scans, rotation and mixed digital/scanned PDFs.
The complete run has 240 documents / 250 page instances. Derived cases are
correlated with their originals and are not counted as independent documents.
The identical three-provider comparison uses 50 documents / 60 page instances:
ten original controls plus all forty variants.

### All 200 original public pages

| Metric | pypdf | pdfplumber |
| --- | ---: | ---: |
| Mean page CER ↓ | 19.65% | 15.51% |
| Mean page WER ↓ | 27.64% | 26.70% |
| Numeric-token recall ↑ | 84.27% | 84.35% |
| Region-token recall ↑ | — | 82.48% |
| Order-pair accuracy ↑ | — | 82.86% |
| Order-pair coverage ↑ | — | 85.43% |
| Table recall at IoU 0.5 ↑ | — | 52.73% |
| Table precision at IoU 0.5 ↑ | — | 30.85% |
| Normalized TEDS ↑ | — | 40.84% |
| Structure-only TEDS ↑ | — | 42.55% |
| Exact grid-cell accuracy ↑ | — | 34.92% |
| Span accuracy ↑ | — | 48.80% |
| Numeric-cell accuracy ↑ | — | 36.34% |

A dash means an unsupported/unmeasured capability, not zero error. pypdf emits
no table model or word boxes. Table scores include missed reference tables as
zero; they are not restricted to successfully detected tables. TEDS is the
canonicalized variant defined in the benchmark guide, not an official leaderboard score.
Text scores evaluate available extracted text independently of the SDK publication
policy; a `partial` status is not an independently verified extraction failure.

### Same source/variant cohort: mean page CER

| Family | Pages | pypdf | pdfplumber | Tesseract |
| --- | ---: | ---: | ---: | ---: |
| Original controls | 10 | 7.45% | 9.60% | 13.01% |
| 200-DPI scans | 10 | 100.00% | 100.00% | 19.06% |
| 100-DPI JPEG-35 scans | 10 | 100.00% | 100.00% | 66.20% |
| 90° rotation | 10 | 7.45% | 57.95% | 13.43% |
| Mixed two-page PDFs | 20 | 53.73% | 54.80% | 16.03% |

### Same 60 page instances: omission and numeric recovery

| Metric | pypdf | pdfplumber | Tesseract |
| --- | ---: | ---: | ---: |
| Token recall ↑ | 45.59% | 27.35% | 91.29% |
| Numeric-token recall ↑ | 43.94% | 32.06% | 80.73% |
| Region-token recall ↑ | — | 25.98% | 88.99% |
| Order-pair coverage ↑ | — | 42.83% | 90.93% |

Tesseract has no structured table output in this adapter, so no table accuracy
is claimed for its OCR text. Its numeric-token recall on degraded scans is only
43.88%; OCR success status alone would therefore be unsafe as a numeric-quality gate.
On those pages its order-pair accuracy is 100%, but pair coverage is only 76.57%. Both metrics are needed to expose omitted regions.

### Same 50 documents: extraction latency and process memory

| Provider | Document p50 | Document p95 | Provider Python peak RSS | OCR subprocess peak RSS |
| --- | ---: | ---: | ---: | ---: |
| pypdf | 0.222s | 0.254s | 47.3 MiB | — |
| pdfplumber | 0.225s | 0.305s | 56.2 MiB | — |
| tesseract | 1.896s | 4.183s | 126.2 MiB | 113.5 MiB |

Latency includes a fresh provider process, IPC and cleanup per document; it
excludes metric scoring. RSS columns are separate maxima, not simultaneous
whole-system totals. These are local observations, not capacity guarantees.

## Reproducibility and regression evidence

- Dataset revision: `24702c61a2fb13325534be664653bc6e60250d13`.
- Reference SHA-256: `116a42c284890772048c5d21486882f2632da4947c3dd90434efef4f49b7d9fd`.
- Original PDF bytes verified against the pinned upstream Git LFS SHA-256 index.
- pypdf 6.21.2, pdfplumber 0.11.10 / pdfminer.six 20260107, Tesseract 5.5.3.
- Tesseract: English, PSM 3, 200-DPI rendering. pdfplumber: text flow, line-based tables.
- Rendering: pypdfium2 5.14.0 / Pillow 13.0.0; fonts: fonttools 4.68.0.
- Scoring: APTED 1.0.3 / RapidFuzz 3.14.6; full environment pinned in requirements.
- Four rendered annotation overlays were inspected: tables, charts, equations and
  two-column prose. All source region boxes passed normalized bounds checks.
- Hand-computed tests reject wrong numbers, swapped cells, lost merged cells,
  missed columns/tables, unsupported capabilities and provider failures.
- The zero-tolerance pypdf→pdfplumber comparison **rejects** the candidate despite
  better aggregate public CER: its rotation and mixed-page families regress.
  This demonstrates why a single global text score is not a release gate.

## Extended v3 comparison: Arabic scans, handwriting and SDK contracts

An additional fixed comparison uses **89 documents / 99 page instances**: the
60-page structural cohort plus 20 Misraj document images, ten physical Yarmouk
scans and nine unambiguous HATHRD manuscript image/transcript pairs. Sources are
pinned and described in [the source decisions](public-pdf-sources.md). These are
upstream transcriptions; they do not constitute independent local approval.

All providers completed this cohort with **zero failures or missing pages**.
The deadline was 90 seconds per document. Tesseract 5.5.3 used Arabic + English
fast models pinned to tag 4.1.0, PSM 3 and 200 DPI. This differs from the earlier
English-only v2 run above; those OCR numbers are separate configurations.
Pdfplumber retains its original text-flow extraction; an independent analysis
pass measures blocks without reordering the extraction stream.

| Family | Pages | pypdf CER | pdfplumber CER | Bilingual OCR CER |
| --- | ---: | ---: | ---: | ---: |
| arabic_handwriting | 9 | 100.00% | 100.00% | 72.92% |
| arabic_mixed_origin | 20 | 100.00% | 100.00% | 46.46% |
| arabic_real_scan | 10 | 100.00% | 100.00% | 28.88% |
| mixed_two_page | 20 | 53.73% | 54.80% | 32.43% |
| public | 10 | 7.45% | 9.60% | 30.93% |
| rotated_90 | 10 | 7.45% | 57.95% | 34.41% |
| scan_100dpi_jpeg35 | 10 | 100.00% | 100.00% | 42.73% |
| scan_200dpi | 10 | 100.00% | 100.00% | 33.93% |

CER can exceed 100% when insertions exceed reference length. The native parsers
are negative controls on image-only PDFs; their 100% errors do not imply missing
outcome pages. Arabic transcription-only sources have no asserted geometric
ground truth, so their region/order accuracy remains unmeasured.

| Contract / resource metric | pypdf | pdfplumber | Bilingual OCR |
| --- | ---: | ---: | ---: |
| Recall@5 | 15.18% | 11.26% | 4.76% |
| MRR | 0.1667 | 0.1429 | 0.0476 |
| Answer-source hit rate | 23.81% | 14.29% | 4.76% |
| Citation resolution | 100.00% | 100.00% | 100.00% |
| Mean numeric-field task accuracy | 0.00% | 8.48% | 0.00% |
| Positive completeness claims | 0 | 0 | 0 |
| Document p50 | 0.224s | 0.186s | 2.133s |
| Document p95 | 0.270s | 0.574s | 6.984s |
| Provider peak RSS | 48.0 MiB | 156.8 MiB | 176.0 MiB |
| OCR subprocess peak RSS | — | — | 183.5 MiB |
| CPU hours / 1,000 pages | 0.0187 | 0.0364 | 0.6459 |

Pdfplumber text-block precision/recall are 7.09% / 25.00%
on pages with layout ground truth. OCR foreground-ink coverage averages 71.93%;
coverage is a diagnostic, not proof of correct words or values. There are 3,030
numeric grid-position references projected from upstream table annotations,
including covered positions in merged cells and repeated stress variants.
The field score is a macro average over pages with reference fields. OCR emits
no structured table values in this adapter, so those field tasks score zero.

The 21 unique queries are reference-derived lexical anchors, deduplicated from
40 balanced probes, with fixed source/page targets. Expected answer-bearing
chunks are located in actual parser output; omitted answers remain misses.
They test the SDK chunker, BM25 retrieval, evaluation, evidence and source lookup,
not natural-question or semantic-model QA. Citation resolution tests exact spans
of whatever was retrieved; it must not be interpreted as correct-answer accuracy.
All providers passed the four source authorization/revision checks and the
cross-tenant retrieval denial probe. Production backend authorization remains
a separate qualification concern.

No provider asserted verified completeness on this cohort. Observable verifier
false-negative rates therefore remain **unmeasured**, rather than being called
zero. Injected false completeness claims are caught by unit tests. General
chart/formula scoring is implemented with bounded semantics, but the corpus has
no independently validated canonical field labels or provider semantic outputs
for those tasks; no measured chart/formula accuracy is claimed.

External API charges were zero for these local providers. Compute USD and total
USD / 1,000 pages remain null because no host price was supplied. CPU measurement
excludes startup/scoring and does not price idle or capital costs.

The local 50-page review packet contains pending decisions and an unapproved
policy template. The qualification command correctly returns **not qualified**
with six failed gates: policy approval, human review, reviewed family coverage,
table-value measurement, span measurement and a priced compute scope for the
OCR baseline. No human approval or production tolerance was synthesized.
A second pypdf replay passed all **89** zero-tolerance quality, retrieval and
citation comparisons on the same manifest. Latency and usage remain observations.

## SDK integration smoke verification

The optional SDK adapters now run independently of benchmark code. An
`AutoPDFExtractor` smoke check exercised six existing corpus PDFs: a native
public page, degraded scan, rotated page, physical Arabic scan, manuscript and
mixed-origin Arabic document. It returned 2,305 words and three native table
models across those six pages, with no worker failures. All six remained partial
because no host verification policy was supplied. This verifies runtime wiring,
not extraction accuracy or publication approval. The historical provider accuracy
measurements above are not a complete requalification of the SDK adapters.
See [SDK usage and verification](../pdf-extractors.md).

## Remaining limits

Issue #60 remains open for independent annotation and full-family review,
natural-question QA, measured chart/formula provider semantics, host prices and
approved release policy. Contracts/invoices, cross-page table stitching and
arbitrary capture/noise distributions are not certified by these selections.
The public transcriptions and controlled variants do not replace manual review.

All PDFs, images, ground-truth text/HTML, manifests, predictions, raw JSON reports
and review packets remain local and ignored. This document contains aggregate
results only. See [metric definitions and commands](../extraction-benchmarks.md).
