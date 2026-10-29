# SDK PDF layout and OCR extractors

The SDK ships optional native-layout and local OCR adapters. They run from an
installed Ragnor package and do not import `benchmarks/` or download models/data.
The default `PDFExtractor` remains the lightweight pypdf adapter.

```sh
# From this source checkout:
pip install -e '.[pdf-auto,directory]'
# For a release containing these adapters:
pip install 'ragnor[pdf-auto,directory]'
# Install Tesseract and the language models on the host separately.
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara
# macOS: install Tesseract using your host package manager; provide Arabic data
# in an explicitly selected tessdata directory when needed.
```

| Adapter | Python extra | Output |
| --- | --- | --- |
| `PdfPlumberExtractor` | `pdf-layout` | Native text, word/region boxes, table cells and row/column spans |
| `TesseractPDFExtractor` | `pdf-ocr` | OCR text, word/region boxes, word confidence and ink-coverage diagnostics |
| `AutoPDFExtractor` | `pdf-auto` | Complete native pages; OCR fallback for uncertain pages, retaining native tables |

## Inspect extracted data

```python
from pathlib import Path
from ragnor.ingestion import AutoPDFExtractor, TesseractPDFExtractor

extractor = AutoPDFExtractor(
    ocr=TesseractPDFExtractor(
        language="ara+eng",
        tessdata_dir="/path/to/tessdata",  # contains ara.traineddata and eng.traineddata
    ),
)
data = Path("report.pdf").read_bytes()
pages = extractor.extract_outcomes(data)
for page in pages:
    # page.text, page.words, page.regions, page.tables are actual extracted data.
    # page.tables[*].cells contain row, col, rowspan, colspan, text and bbox.
    # page.tables[*].to_html() preserves spans and escapes cell values.
    print(page.page, page.status, len(page.words), len(page.tables))
```

All boxes use normalized `[x0, y0, x1, y1]` coordinates, with a top-left origin
on the full MediaBox after PDF page rotation. OCR expands a CropBox to MediaBox
before rendering so its boxes align with native table evidence. This includes
content outside a narrower CropBox. Page numbers are one-based. Native word flow
is preserved; region detection uses a separate layout-analysis pass.

`include_tables=True` on `PdfPlumberExtractor` replaces native table words with
escaped HTML at the table's first word, retaining merged-cell structure in the
page's retrieval text. Set it to `False` for plain native word-flow text while
keeping structured tables. When routing through OCR, native table HTML is
appended to OCR text if this option is enabled; the native table provider is
recorded separately. OCR does not infer structured tables from raster images.

The legacy `extract(data)` and `extract_pages(data)` views return available text,
including unverified text. They are useful for inspection, and do not authorize
publication. Use `extract_outcomes()` and connectors for quality-aware ingestion.

## Verify before publishing uncertain pages

A readable native text-only page can be `extracted`. OCR text and native pages
with unverified visual content remain `partial` by default. OCR finding words
is not evidence that it recovered every field, chart or column. A white rendered
page with no OCR words can be `verified_blank`; a nonblank image with no words
stays `needs_ocr`. Provider/page failures remain `failed`.

A host can supply a `PageVerifier` with `id`, `version`, JSON `options`, and a
synchronous `verify(page) -> bool`. Ragnor invokes it for recovered text pages,
including otherwise complete native pages; `False` makes the page partial.
Failed/empty pages cannot be promoted by this hook. Exceptions, nonboolean
results or policy identity changes fail the extraction. Versions/options must
describe the host's acceptance rules, including any approved thresholds.
Confidence and ink coverage are diagnostics, not built-in acceptance thresholds.

For exact reviewed evidence, the SDK supplies `EvidenceAllowlistVerifier`:

```python
from ragnor.ingestion import (
    AutoPDFExtractor, EvidenceAllowlistVerifier, TesseractPDFExtractor,
)

# Load these hashes from a host-owned review store AFTER reviewing the source,
# recovered text, table values/spans, omissions and exclusions.
# Ragnor never creates approvals from an extraction success or benchmark score.
def approved_extractor(approved_hashes: list[str]) -> AutoPDFExtractor:
    return AutoPDFExtractor(
        ocr=TesseractPDFExtractor(language="ara+eng", tessdata_dir="/path/to/tessdata"),
        verifier=EvidenceAllowlistVerifier(approved_hashes),
    )
```

Each page's `evidence_sha256` binds source PDF bytes, actual provider/model
identities, text, geometry, table cells and diagnostics. The subsequent acceptance
decision does not change this hash. A changed model or source requires fresh
approval. An empty allowlist accepts no recovered text pages. This is a host
attestation mechanism; it does not authenticate a human reviewer.

## Use through ingestion connectors

```python
from ragnor.ingestion import DirectoryConnector

# extractor is an AutoPDFExtractor configured with your host's acceptance policy.
connector = DirectoryConnector(
    "./documents",
    options={"extensions": ["pdf"], "extractor": extractor},
)
# In an async function:
# result = await connector.load()
# Check result.authoritative before treating it as a complete revision.
# Pass connector/result through your existing IngestionPipeline or sync host.
```

Directory, HTTP and object-storage connectors share the same extraction seam.
Page documents retain `location['page']` and structured evidence in
`metadata['page_extraction']`. Actual package versions, executable/model hashes
and configured policy feed extraction metadata and source/snapshot hashes.
Runtime identities belong to each result rather than mutable shared adapter
state, so concurrent documents retain their own provenance.

Under strict/review policy, one unverified page withholds every page in that
PDF revision. The existing publication path preserves the active version and
refuses empty-load purges from an incomplete result. Explicit `allow_partial`
exposes recovered documents with incomplete quality; it does not make the result
authoritative or bypass pipeline safeguards.

## Resource and execution boundaries

`PDFExtractionLimits` configures input bytes, page count, rendered pixels per
page, cumulative words/cells/text, output bytes and a document deadline. Defaults
are 50 MB input, 1,000 pages, 25 million rendered pixels/page, 200,000 words,
50,000 cells, 2 million text characters, 32 MB worker output and 60 seconds.
Each Tesseract call also has `page_timeout` (default 30 seconds). Limits reject
rather than silently truncating. The router uses the strictest quotas from its
own configuration and its two child adapters.

Optional vendor code runs in a fresh installed SDK subprocess, including PDFium
rendering; SDK callers can use threads without concurrently calling PDFium in
the host process. No multiprocessing main-module guard is required. Provider
stdout/stderr are suppressed, and exceptions expose fixed error codes. On
Linux/macOS the document deadline kills the worker process group, including OCR
descendants. Cleanup allows up to one second to reap children before force-killing
a worker. Tesseract OCR currently requires a POSIX host for that cleanup guarantee;
native layout extraction also works without the OCR executable.

These quotas do not constitute a hard RSS cap on native parser allocations.
No cloud service, model download, chart/formula interpretation, or handwriting
accuracy guarantee is included. The benchmark's poor handwriting/financial
recovery remains a reason to review evidence, not to certify completeness.

Underlying interfaces: [pdfplumber](https://github.com/jsvine/pdfplumber),
[Tesseract](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html), and
[PDFium's threading requirements](https://pypdfium2.readthedocs.io/en/stable/python_api.html).

## Routing and evidence storage

Adapter version 2 keeps native text and tables when a partial page has no image
content. Drawing rules alone no longer trigger OCR. Pages with images or without
usable native text still receive OCR. These routing decisions do not approve a
page: uncertain visual content remains partial. Failed OCR preserves available
native text. When native tables accompany OCR, words inside those table boxes
are replaced with the native table HTML to avoid duplicate table values.
Ambiguous table grids are rejected individually; the page retains its words and
text and remains partial, with a `rejected_tables` diagnostic.

Connector `RawDocument` metadata retains full page words, regions and tables.
Vector-store chunks carry their counts and `evidence_sha256` instead of copying
that geometry into every chunk. Hosts needing later geometry lookup must retain
the original page evidence; Ragnor does not introduce an evidence database.
Review allowlist configuration contains a digest and count of the approval set,
so large host-reviewed allowlists do not exceed extractor configuration bounds.

At most four optional PDF workers run concurrently per host Python process;
waiting for a worker consumes the document deadline. On POSIX, worker and OCR
output files have an OS-enforced file-size ceiling, and a watchdog terminates
the worker process group if its owning host disappears. An optional
`PDFExtractionLimits(max_memory_bytes=...)` applies a Linux `RLIMIT_AS` address
space ceiling, inherited by OCR children. It is disabled by default; requesting
it on another platform fails closed. This is not an RSS measurement or a total
memory budget across concurrent workers. Container limits remain appropriate
for deployment-wide resource control.

## Hosted multimodal OCR through LiteLLM

`OCRProvider` is a PDF-level port with synchronous `extract_outcomes` and
asynchronous `aextract_outcomes` methods, optional page selection and shared
resource limits. `TesseractOCRProvider` implements it using the existing local
Tesseract worker; `TesseractPDFExtractor` remains compatible. The hosted
`LiteLLMOCRProvider` uses the same outcome types and verification policy.

```sh
pip install 'ragnor[pdf-llm,pdf-layout]'
```

```python
from pathlib import Path
from ragnor.ingestion import AutoPDFExtractor, LiteLLMOCRProvider

ocr = LiteLLMOCRProvider(model="openai/gpt-5")
# Or: LiteLLMOCRProvider(model="gemini/gemini-2.5-pro")
extractor = AutoPDFExtractor(ocr=ocr)
pages = extractor.extract_outcomes(Path("document.pdf").read_bytes())
for page in pages:
    print(page.page, page.status, page.text)

# In an async application:
# pages = await extractor.aextract_outcomes(pdf_bytes)
```

Choose an available vision-capable model identifier supported by your LiteLLM
installation and account. The backend forwards the identifier without a hardcoded
model list. LiteLLM handles provider-specific routing for the standard
[text plus image input](https://docs.litellm.ai/docs/completion/vision).
GPT-5 and Gemini request contracts are tested with fake transports; no live model
accuracy or availability qualification is claimed.

Credentials can come from LiteLLM's environment configuration or the provider's
`api_key` argument. `api_base` supports a host-selected gateway endpoint.
Host-owned LiteLLM Router methods can be injected as `completion=router.completion`
and `acompletion=router.acompletion`; the SDK does not close host clients.
Keys and callables are excluded from options, provenance and representations.

Selecting hosted OCR explicitly authorizes sending selected page images to the
configured model service. There is no automatic switch from local OCR to cloud.
The auto router sends only uncertain pages that lack usable native text or contain
images. PDF rendering stays inside the bounded local worker and does not require
Tesseract or traineddata. Standalone hosted OCR renders every requested page.

Defaults are 150 DPI, at most 20 OCR pages per document, 4,096 completion tokens
per page, a 30-second request timeout and two concurrent async requests per document. Document,
pixel and output quotas still apply. Internal LiteLLM retries are disabled; a
host-owned Router may implement its own policy. `generation_options` permits
explicit `temperature`, `top_p`, `seed` or `reasoning_effort` settings; none are forced
onto models that may not support them. Limits bound requests/tokens, not an exact
USD price. Usage counters are recorded when returned by the provider.

Hosted output supplies text/Markdown without fabricated word coordinates or table
cells. Native table evidence may be retained separately by the auto router.
Successful responses remain partial until host verification. Truncated, refused,
empty or otherwise incomplete responses cannot be promoted by a verifier. Model,
prompt, source and rendered-image identities are bound to the review evidence;
changing them requires new approval. Hosted output can vary between runs; evidence
allowlists approve exact outcomes, not future calls.

Use async methods for cancellable model requests and bounded concurrency. Sync
methods use the synchronous LiteLLM transport and its HTTP timeout; injected
transports must honor that timeout. Existing connector extraction remains
synchronous. Worker memory limits cover local rendering, not the hosted client.
Cancellation during local rendering leaves the worker governed by
its existing deadline/cleanup bounds. The SDK disables LiteLLM request logging
for these calls; hosts still own gateway logging, callbacks and data retention.
