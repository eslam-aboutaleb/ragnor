# Offline PDF quality benchmarks (source checkout)

Use these source-checkout tools to measure the installed SDK adapters. They are
not runtime SDK dependencies. The fixture-based gate is described in
[Extraction-quality benchmarks](extraction-benchmarks.md). See [PDF extractors](pdf-extractors.md) for the
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
