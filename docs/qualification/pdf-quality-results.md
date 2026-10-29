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

## Remaining limits

Issue #60 remains open. Full independent annotation review, Arabic/RTL, handwriting,
representative real-world scans, semantic chart/formula evaluation, downstream RAG
retrieval, permission-aware citations, provider cost and approved production
limits are not established. The reference table models are upstream labels, not
new human annotations created by this run. The controlled degradations do not
certify arbitrary camera captures, skew, real noise or cross-page table stitching.

All PDFs, images, ground-truth text/HTML, manifests, predictions and raw JSON
reports remain local and ignored. This document contains aggregate results only.
See [metric definitions and commands](../extraction-benchmarks.md).
