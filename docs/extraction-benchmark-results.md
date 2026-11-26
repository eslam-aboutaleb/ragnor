# Extraction benchmark results (EX-02 fixture corpus)

Measured on the in-repo synthetic CC0 corpus (`benchmarks/pdf_extraction/fixtures`: 18 documents, 30 pages, 18 families, 2 of them corrupt/oversized inputs expected to error). Real extractors only: `pypdf`, `sdk-native` (pdfplumber), `sdk-ocr` (real Tesseract 5), `sdk-auto`. No mocks, no network, no paid calls (cost is 0 because nothing hosted ran). Python 3.12, one sandbox CPU, single run each: timings are indicative, not a performance claim.

Reproduce: `python -m benchmarks.pdf_extraction run --provider <id> --out run.json` and `python -m benchmarks.pdf_extraction probes --provider <id> --out probes.json`.

## Result

| Provider | CER | WER | Region recall | Cell-value acc. | Field acc. | Silent omissions | Probe Recall@5 / MRR | p50 ms/page | p95 ms/page | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| pypdf | 0.144 | 0.145 | 0.848 | 1.000 | 1.000 | 1 | 1.000 / 1.000 | 11 | 511 | 22 |
| sdk-native | 0.143 | 0.143 | 0.857 | 0.889 | 1.000 | 0 | 1.000 / 1.000 | 159 | 266 | 33 |
| sdk-ocr | 0.168 | 0.207 | 0.791 | 0.622 | 0.833 | 11 | 0.375 / 0.375 | 1118 | 1586 | 33 |
| sdk-auto | 0.000 | 0.000 | 1.000 | 0.889 | 1.000 | 0 | 1.000 / 1.000 | 160 | 1160 | 33 |

CER/WER of 0.14 for pypdf and sdk-native come from the scan and handwriting pages, which have no text layer (nothing to extract without OCR). `sdk-auto` routes those pages to OCR.

## Where each one loses

* **pypdf**: no text on scans or handwriting (CER 1.0 on those families), one silent omission (multilingual page), and Arabic-Indic digits come out reversed (`٥٧٫٠٨٤٢` for 2480.75). It is the fastest by 14x.
* **sdk-native**: no OCR, so scans and handwriting are empty. Cell-value accuracy 0.889 (below pypdf's 1.000): the ruled-invoice family scores 0.67. I have not yet diagnosed which cells are wrong. Slower than pypdf (159 vs 11 ms/page p50) for the layout and word boxes.
* **sdk-ocr alone**: worst on digital pages. Cell-value accuracy is 0.622 (split/merged tables 0.20, ruled invoice 0.68). Arabic CER is 0.88 with the English model, because only `eng` was installed. Silent omissions: 11 pages across the Arabic, borderless, formula, multilingual, ruled and split-table families. 100x slower than pypdf. Only use it through `sdk-auto`, which sends just the pages without native text to OCR.
* **sdk-auto**: best on this corpus, with 1.000 region recall and 0.000 CER. The corpus is small and synthetic, so that is the ceiling of a controlled test, not a production accuracy claim. Its p95 (1160 ms/page) is the OCR pages.

## Defects this benchmark found and fixed

1. **Right-to-left text came back reversed from the native engine.** Before: Arabic family CER 0.80, region recall 0.00, 2 silent-omission pages for `sdk-native` and `sdk-auto`; probe Recall@5 0.625. Cause: pdfminer returns glyphs in drawn (visual) order. Fix: `to_logical_order` in `ragnor/ingestion/_pdf_native.py` reverses RTL lines and words while keeping LTR runs. The Arabic fixture now equals pypdf's logical text word for word.
2. **The text metrics counted table markup as errors.** Native and auto engines embed tables as HTML in `text`, so CER on the ruled-invoice and split-table families was 2.99 and 3.84 (insertions). The benchmark provider now scores the plain-text view; tables are scored from the cell data, so nothing is hidden. This is a measurement fix, not an accuracy gain: the HTML stays in the extractor output for chunkers.

| Provider | Before CER / region recall / probe R@5 | After |
|---|---|---|
| sdk-native | 0.705 / 0.705 / 0.625 | 0.143 / 0.857 / 1.000 |
| sdk-auto | 0.562 / 0.848 / 0.625 | 0.000 / 1.000 / 1.000 |

## What this does not show

* Only 8 retrieval probes, with the benchmark's own deterministic hashing embedder and chunker. Recall@5 of 1.000 means the corpus is easy, not that retrieval is solved. A real embedding model and a larger human-annotated corpus are needed to rank providers for retrieval.
* The corpus is synthetic. Real contracts, invoices and scans (the corpus fetched by `benchmarks/fetch_public_pdf_corpus.py`) are not part of this run.
* No VLM, hosted OCR or Docling provider was run. Running them needs: a configured provider (credentials and model id) through the provider ports, or `pip install docling` plus model weights for Docling. Their numbers are therefore not reported here.
* Single run per provider on a shared sandbox CPU: no confidence intervals.
