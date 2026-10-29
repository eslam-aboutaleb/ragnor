# Public PDF benchmark source decisions

The benchmark acquires all 200 original pages from
[Upstage DP-Bench](https://huggingface.co/datasets/upstage/dp-bench), pinned to
revision `24702c61a2fb13325534be664653bc6e60250d13`. Its reference annotations contain
text, polygons, ordered elements and table HTML. The dataset card declares MIT.
All source PDFs are verified against the pinned upstream LFS hashes; the reference
JSON SHA-256 is checked independently. No source documents or annotations are
included in this repository's commits.

[DocLayNet](https://github.com/DS4SD/DocLayNet) remains a candidate for more human
layout annotations; layout labels and PDF text cells do not establish independently
validated transcription or table-cell accuracy.
[KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench) remains a candidate for
Arabic/RTL and handwriting after component-specific data terms are verified.
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) states noncommercial
research restrictions for its dataset, so it is not acquired or vendored as the
default corpus.

The earlier 40-page selection is superseded by the full corpus, structural metrics,
multiple providers and controlled stress variants. The independent full-annotation
review gate is still open. See [reproduction and metric definitions](../extraction-benchmarks.md)
and [actual aggregate results](pdf-quality-results.md).

## Added Arabic sources

- [Misraj-DocOCR](https://huggingface.co/datasets/Misraj/Misraj-DocOCR), Apache-2.0,
  pinned revision `7177bf70f77ce259890d6af0bef53f18faf26ec9`. The card describes
  expert-verified transcriptions for real and synthetic document images. The
  importer selects fixed rows 0–19 and preserves Markdown references locally.
  Its family is `arabic_mixed_origin`; individual pages are not asserted to be
  physical scans. Both Parquet shards are verified against upstream LFS SHA-256.
- [HATHRD](https://data.mendeley.com/datasets/xz6f8bw3w8/1),
  DOI `10.17632/xz6f8bw3w8.1`, CC-BY-4.0. Nine exact image/DOCX stem pairs are
  imported from the expert-transcribed historical handwriting collection.
  Shared A/B transcripts without a proven page partition are excluded.
- [Noisy OCR Dataset](https://zenodo.org/records/5068735),
  DOI `10.5281/zenodo.5068735`, CC-BY-4.0. Ten single-image articles from the
  clean Yarmouk colour scans are matched by article ID to upstream transcripts.
  The importer excludes multi-image articles to avoid scoring one page against
  an entire article. Raw image/transcript archives are SHA-256 pinned.
- [Tesseract fast language data](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0),
  tag 4.1.0, supplies pinned Arabic and English traineddata. Reports record their
  SHA-256 hashes and the host executable version.

The image sources are converted to fixed-metadata, 200-DPI image PDFs for this
extractor comparison. Text references are upstream labels, not outputs from the
extractor under evaluation. No geometric layout or caption associations are
invented for transcription-only sources. Independent local review and full
coverage of every issue #60 family remain pending; a 50-page review packet is
created locally with every decision initialized to pending. Acquisition and raw
review evidence remain excluded from Git.
