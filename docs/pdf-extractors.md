# SDK PDF extractors

Optional adapters that produce typed `PageExtractionOutcome` evidence for the
fail-closed ingestion policy described in
[Extraction And Provenance](connectors.md#extraction-and-provenance). The
default `PDFExtractor` (pypdf) stays the lightweight path. These adapters are
for hosts that need OCR or layout extraction; every one of them emits
**unverified** evidence that a host verifier must accept before publication.

| Adapter | Extra | Runs | Output |
| --- | --- | --- | --- |
| `PdfPlumberExtractor` | `pdf-layout` | Sandboxed SDK worker; no OCR needed | Native words, regions, table cells with row/column spans |
| `AutoPDFExtractor` | `pdf-auto` | Native first; OCR only for uncertain pages | Native text and tables, OCR text for image pages |
| `TesseractPDFExtractor` / `TesseractOCRProvider` | `pdf-ocr` | Sandboxed SDK worker; host-installed Tesseract | Word boxes, confidence, text |
| `LiteLLMOCRProvider` | `pdf-llm` | Hosted multimodal model via LiteLLM | Text/Markdown only, no boxes |
| `DoclingPDFExtractor` | `pdf-docling` | Sandboxed SDK worker; host-provisioned Docling models | Text and table cells/spans |

The core library keeps **zero runtime dependencies**; importing these adapters
imports no vendor package. Local vendors run in a bounded subprocess (quotas,
deadlines, process-group cleanup, parent-death supervision); hosted models are
called through host-owned transports. OCR or layout success alone never
certifies a page: output stays `partial` until a `PageVerifier` accepts the
exact evidence hash, which binds source bytes, provider, model, prompt and
rendered image identities. `EvidenceAllowlistVerifier` accepts only explicit
SHA-256 evidence digests supplied by the host.

## Native layout and automatic routing

```sh
pip install 'ragnor[pdf-auto]'   # pdf-layout (pdfplumber) plus pdf-ocr
```

`PdfPlumberExtractor` reads machine-generated PDFs without OCR. It returns
normalized word and region boxes (top-left origin on the full MediaBox) and table
cells with row/column spans. `include_tables=True` replaces table words with
escaped table HTML at the table's first word; `False` keeps plain word flow while
still returning structured tables. One ambiguous table grid is rejected on its
own: the page keeps its words, becomes `partial` and records `rejected_tables`.

`AutoPDFExtractor` keeps complete native pages and sends only pages with no usable
native text, or with image content, to OCR. Native tables are kept next to OCR
text and the words inside those table boxes are replaced by the native table HTML
so values are not duplicated. If OCR returns nothing for a page that had native
text, the native text is kept. Configure the verifier on the router, not on its
child adapters. Shared quotas are the strictest of the router and both children.
`ocr=` accepts any `OCRProvider`, including `LiteLLMOCRProvider`.

```python
from ragnor.ingestion import AutoPDFExtractor, TesseractPDFExtractor

extractor = AutoPDFExtractor(
    ocr=TesseractPDFExtractor(language="ara+eng", tessdata_dir="/path/to/tessdata"),
)
pages = extractor.extract_outcomes(pdf_bytes)
```

As with every adapter here, OCR and uncertain visual pages stay `partial` until a
host `PageVerifier` accepts the exact evidence hash.

For corpus-level quality measurement of these adapters (CER, token recall, region
and table scores, retrieval probes, review packets) see
[PDF quality benchmarks](pdf-quality-benchmarks.md).

## Limits

All adapters accept `PDFExtractionLimits` for byte, page, pixel, word, cell,
text and output quotas plus a document deadline. Exceeding a quota raises
`PDFExtractionError` with a fixed, content-free code instead of truncating.

## Local Tesseract OCR (`pdf-ocr`)

```python
from ragnor.ingestion import TesseractPDFExtractor

extractor = TesseractPDFExtractor(language="ara+eng", dpi=200)
pages = extractor.extract_outcomes(data)  # pages stay partial until verified
```

The host installs the `tesseract` executable and its language models; the
worker hashes the executable and every selected `*.traineddata` into page
evidence, so a model change invalidates prior approvals. The process-tree
deadline requires a POSIX host. Rendering uses pypdfium2 at a configurable DPI
under a pixel quota; an ink-coverage diagnostic compares OCR word boxes against
rendered foreground pixels.

## Hosted multimodal OCR (`pdf-llm`)

```python
from ragnor.ingestion import LiteLLMOCRProvider

provider = LiteLLMOCRProvider(model="openai/gpt-5", max_ocr_pages=20)
pages = provider.extract_outcomes(data, page_numbers=frozenset({3}))
```

Any LiteLLM vision-capable model identifier works; the requested and returned
model identities, prompt hash and rendered-image hash are bound into evidence.
Credentials and injected callables stay in memory and never enter snapshot
options or diagnostics. Truncated, refused or empty responses become `failed`
pages that a verifier cannot promote. Sync and async calls share page, request
and output quotas; retries are disabled by default.

## Docling layout and tables (`pdf-docling`)

```python
from ragnor.ingestion import DoclingPDFExtractor

extractor = DoclingPDFExtractor(artifacts_path="/opt/docling-models")
pages = extractor.extract_outcomes(data)
```

Docling is MIT licensed. The default `docling` install pulls PyTorch and
layout models, so it stays an opt-in extra; the worker never downloads models
implicitly (Hub offline mode), so hosts must provision them ahead of time and
pass `artifacts_path`. Converted pages become `partial` evidence with text and
anchored table cells (row/column spans). Geometry is coarse: tables carry
table-level boxes and there are no word boxes. Pages Docling leaves empty keep
their pypdf baseline status, so unknown pages still fail closed as
`needs_ocr`.

## Verification gate

```python
from ragnor.ingestion import EvidenceAllowlistVerifier, TesseractPDFExtractor

verifier = EvidenceAllowlistVerifier(approved_hashes)  # host-reviewed digests
extractor = TesseractPDFExtractor(verifier=verifier)
```

A verifier is deterministic host code with `id`, `version` and `options`; its
identity joins the extractor fingerprint, so a policy change forces
reindexing. Non-boolean verdicts fail closed. Direct `extract()` and
`extract_pages()` calls follow the same policy: under `strict` or `review` they
raise `unverified_content` unless every page is verified, verified blank or
policy excluded; only `allow_partial` returns unverified text. Under the default `strict`
policy a single unverified page rejects the whole PDF revision and preserves
previously indexed content; `allow_partial` publishes available text with
provenance but never becomes authoritative.
