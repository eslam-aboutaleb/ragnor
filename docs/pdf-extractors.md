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
a worker. OCR currently requires a POSIX host for that cleanup guarantee;
native layout extraction also works without the OCR executable.

These quotas do not constitute a hard RSS cap on native parser allocations.
No cloud service, model download, chart/formula interpretation, or handwriting
accuracy guarantee is included. The benchmark's poor handwriting/financial
recovery remains a reason to review evidence, not to certify completeness.

Underlying interfaces: [pdfplumber](https://github.com/jsvine/pdfplumber),
[Tesseract](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html), and
[PDFium's threading requirements](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
