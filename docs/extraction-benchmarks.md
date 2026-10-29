# Offline PDF extraction benchmarks

Use these source-checkout tools to measure the installed SDK adapters. They are
not runtime SDK dependencies. See [PDF extractors](pdf-extractors.md) for the
public API, verification policies and deployment limits.

## Run against an existing local corpus

The commands below do not fetch documents or OCR models. Install benchmark Python
dependencies from `benchmarks/requirements.txt` in a separate environment if
needed. Supply an already installed Tesseract executable and local language data.

```sh
python -m benchmarks.run_quality \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --provider sdk-auto --cohort common \
  --tesseract-language ara+eng --tessdata-dir benchmarks/data/extended/tessdata \
  --output benchmarks/data/extended/comparison/sdk-auto.json \
  --predictions-out benchmarks/data/extended/comparison/sdk-auto-predictions.json
python -m benchmarks.qualification packet \
  --manifest benchmarks/data/extended/comparison/quality-manifest.json \
  --predictions benchmarks/data/extended/comparison/sdk-auto-predictions.json \
  --output benchmarks/data/extended/sdk-review --count 50
```

Other providers are `sdk-native`, `sdk-ocr` and the reference `pypdf`,
`pdfplumber` and `tesseract` adapters. SDK runs use the SDK worker deadline,
without an outer timeout process that could abandon worker descendants.

Review packets bind source/reference fingerprints and exact SDK evidence hashes.
Reference `status` and `extraction_review_status` are separate pending decisions.
The host owns reviewer authentication, approval storage and any approved hashes
supplied to `EvidenceAllowlistVerifier`. Approval is never synthesized.

## Measurement and release gates

Text CER/WER and token recall, word/region geometry, reading order, table
structure/cells/spans, numeric/critical fields and lexical retrieval/citations
are scored separately. Missing or failed predictions remain in denominators;
unmeasured capabilities are reported as null. SDK text can contain table HTML,
so raw text error rates also count markup. Table metrics score structure and
values independently. Retrieval probes use reference-derived lexical anchors,
not human-authored natural questions or a semantic answer model.

`python -m benchmarks.compare_quality BASELINE CANDIDATE --output REPORT`
checks baseline-relative regressions. Reports must have matching corpus hashes,
cohort labels, page identities and retrieval queries. A passing comparison is
not independent production qualification. `python -m benchmarks.qualification
check --help` lists the report, manifest, approved policy, reviews and baseline
required for the separate release gate. Human approvals and acceptance
thresholds must be supplied by maintainers.

The smoke corpus verifies contracts; it does not establish extraction accuracy.
CI uses generated fixtures and metric tests and does not acquire the public
corpus. Dataset importers remain explicit opt-in tools for rebuilding a corpus;
they are not part of an offline replay or runtime extraction.

## Corpus provenance

Source PDFs, images, annotations, manifests, traineddata, predictions, raw reports
and review packets remain local and ignored. Importers verify pinned checksums;
reports record source/configuration hashes, package versions and actual runtime
model/executable identities. No corpus content is committed.

- [Upstage DP-Bench](https://huggingface.co/datasets/upstage/dp-bench): MIT,
  revision `24702c61a2fb13325534be664653bc6e60250d13`; 200 original pages plus
  deterministic scan, rotation and mixed-page variants.
- [Misraj-DocOCR](https://huggingface.co/datasets/Misraj/Misraj-DocOCR): Apache-2.0,
  revision `7177bf70f77ce259890d6af0bef53f18faf26ec9`; fixed rows 0–19 with
  upstream transcriptions. Mixed-origin pages are not asserted to be physical scans.
- [HATHRD](https://data.mendeley.com/datasets/xz6f8bw3w8/1): CC-BY-4.0;
  nine exact image/transcript pairs. Shared transcripts without a proven page
  partition are excluded.
- [Noisy OCR Dataset](https://zenodo.org/records/5068735): CC-BY-4.0;
  ten single-image articles matched to upstream transcripts by article ID.
- Arabic/English models use pinned
  [tessdata_fast 4.1.0](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0).
  Transcription-only sources do not acquire invented geometric ground truth.

## Latest SDK results — 2026-10-29

Adapter version 2 was measured directly through installed SDK APIs using only the existing
local corpus, executable and traineddata. No documents or OCR models were downloaded. The
native adapter covered 240 documents / 250 pages; the matched native, OCR and automatic
comparison covered 89 documents / 99 pages. All four runs returned every expected page, with no
document-level or page-level failures. A page needing OCR is an incomplete outcome, not
successful extraction.

The existing extended report labels its cohort `all`; the new runs label it `common`. Both
select exactly the same 89 documents and 99 page identities from the same manifest hash. Page-
set identity was checked explicitly. These labels are recorded unchanged; the strict comparison
CLI rejects mismatched cohort labels.

| Provider | Mean token recall | Mean region recall | Mean exact table-cell accuracy |
| --- | ---: | ---: | ---: |
| Standalone pdfplumber | 16.58% | 25.00% | 5.66% |
| SDK native | 18.80% | 25.00% | 5.66% |
| Standalone Tesseract | 73.12% | 78.52% | Unmeasured |
| SDK OCR | 73.12% | 78.52% | Unmeasured |
| SDK automatic | 65.56% | 66.63% | 5.66% |

SDK OCR exactly matches every shared aggregate quality metric from the frozen standalone
Tesseract report, including CER (39.83%), WER, word/number recovery, geometry/order and
critical-field scores. Both use `ara+eng`, 200 DPI, PSM 3 and the same traineddata hashes. Ink
coverage was not projected into this SDK benchmark report, so it is excluded from this equality
assertion. Runtime package, executable and traineddata identities are recorded in the local
reports.

SDK automatic routing is **not accuracy-equivalent to OCR-only** on this selection. Native
text/table retention changes recovery and representation. SDK text includes table HTML; its raw
CER (77.25%, versus OCR 39.83%) also includes markup insertions and is not a pure visible-
transcription comparison. The native full 250-page run achieved 71.97% token recall and 18.30%
exact table-cell accuracy. These are observations on these selections, not production
thresholds.

On the matched cohort SDK native returned 30 partial pages and 69 needing OCR. SDK OCR and
automatic each returned 98 partial pages and one needing OCR. No host verifier was supplied, so
strict connector ingestion would withhold these documents. Raw-outcome quality measurements
must not be confused with approved publication.

The lexical retrieval contract remains weak: SDK OCR and automatic source/answer recall@5 is
4.76%; resolvable citations are 100%, and authorization/revision checks pass. This demonstrates
citation safety, not satisfactory natural-question retrieval or answer quality.

The local 50-page SDK review packet now includes source previews, reference text, actual SDK
text/tables and exact evidence hashes. Reference approval and extraction approval are separate
pending decisions. No reviewer or policy approval was generated. OCR-only matching does not
close the independent-review, complex-table, handwriting, chart/formula, natural-QA or release-
policy gaps.

Worker RSS remains unmeasured; CPU accounting includes worker descendants. Runs shared the host
with checks, so timings are not isolated production throughput measurements. Raw reports,
predictions, previews and review decisions remain local and ignored.
