# Source Connectors

Connectors separate discovery from extraction and preserve the existing
`DocumentSource.load()` API. Imports do not require cloud or extraction SDKs.
Install only the integrations you use:

```sh
pip install 'ragnor[directory,pdf,web,object-storage]'
# Optional native filesystem notifications:
pip install 'ragnor[watch]'
```

HTML extraction uses the Python standard library; `pdf` installs `pypdf`,
`directory` installs `pathspec` for ignore rules, `web` installs `httpx`, and
`object-storage` installs `boto3`. Injecting an object-storage client avoids the
SDK dependency. Core text extraction and connector protocols have no required
third-party dependencies.

## Discovery And Loading

`SourceConnector` provides asynchronous `discover(cursor=None, limit=None)`,
`load_page(cursor=None, limit=None)`, and `load()` methods. `DiscoveryPage` and
`LoadResult` are list-compatible records with `status`, `errors`, `skipped`,
`has_more`, `next_cursor`, and `complete`. `LoadResult.documents` returns the
loaded documents. `load()` and `load_all()` collect a snapshot from its beginning;
`load_page()` reads only one bounded page.

```python
from ragnor.ingestion import DirectoryConnector

source = DirectoryConnector(
    "./knowledge",
    include=["**/*.md", "**/*.txt", "**/*.html", "**/*.pdf"],
    exclude=["private/**"],
    extensions=["md", "txt", "html", "pdf"],
    max_bytes=10 * 1024 * 1024,
    page_size=100,
)
result = await source.load_all(max_pages=50)
for skipped in result.skipped:
    print(skipped.path, skipped.reason)
if result.authoritative:
    # Pass the connector/result-preserving adapter to the ingestion pipeline.
    # A production sync must never infer deletion from an incomplete result.
    pass
```

`limit` bounds the discovery work represented by a page, not necessarily the
number of output documents: a PDF can produce multiple page documents and a
policy exclusion produces a skip record instead. Objects/responses have separate
byte limits, and PDFs have an extraction page limit. A full snapshot necessarily
collects its output documents in memory; cap pages and bytes for your workload.

## Completeness And Deletion

Only `result.authoritative` permits full-snapshot replacement. It requires a
complete successful traversal, no errors, and a non-incremental result.
Intermediate pages, interrupted runs, failed reads, malformed or repeated
continuation tokens, rate limits, and extraction failures cannot authorize
deletion. The ingestion pipeline checks this for **both empty and nonempty**
results before changing versions or the search index.

Successful empty scans use `SourceStatus.EMPTY`. Missing sources use `NOT_FOUND`,
permission/authentication failures use `INACCESSIBLE`, and temporary service or
transport failures use `TRANSIENT`. These are not interchangeable with emptiness.
`skipped` records represent intentional exclusions such as ignored, binary,
unsupported content, size-policy exclusions for non-PDF files, or legacy
custom-extractor no-text output. Recognized PDFs exceeding a connector size
limit fail the scan; increasing file size must not authorize deletion of a
previous revision. `errors` represent inability to
establish an authoritative scan. A failed file is not evidence that it was deleted.

An authoritative empty result is a no-op by default. The existing
`purge_on_empty=True` pipeline option explicitly enables retirement for a
successful empty snapshot. Nonempty authoritative snapshots replace the previous
version. The host chooses a stable source scope/version store for each snapshot;
never mix unrelated connector scopes in one version store.

## Connector Semantics

| Connector | Pagination And Completeness | Failure And Rate Limits |
| --- | --- | --- |
| Directory | Deterministic depth-first traversal with sorted directory entries and relative POSIX names. Opaque cursors belong to one traversal; starting a new traversal invalidates prior cursors. Reconciliation starts fresh. | Missing/unreadable paths and extraction failures invalidate the snapshot. Policy skips are reported separately. |
| Object storage | Opaque provider listing tokens; all pages and selected objects must succeed. Missing/repeated tokens are incomplete. Listing is not a point-in-time snapshot during concurrent writes. ETag preconditions protect discovered object versions. | Client credentials and retries are injected. S3 `SlowDown` and temporary SDK errors are transient, not empty results. The SDK/host owns retry/backoff. |
| HTTP/API | Real bounded discovery/load pages. Callback-based APIs must supply a valid continuation when more data exists. A callback's `has_more` is authoritative only when pagination exhausts without errors. | 404 is not-found, 401/403 inaccessible, 429 and temporary server/transport failures transient. Rate-limit metadata is exposed without credentials; the host schedules backoff rather than treating failure as exhaustion. |
| Website | Explicit allowed domains; independent URL/request, successful-page, depth, response-byte, and timeout bounds. Limit exhaustion is incomplete. Relative links resolve against the final response URL. | Every redirect is revalidated. Private-network destinations are denied by default. Failed pages do not turn into a successful empty website. |

No connector infers a deletion from absence in a single page. Page cursors are
not incremental checkpoints and should not be persisted as completed snapshots.
Resume only within the connector's documented cursor lifetime. Periodically
restart reconciliation from the first page to repair concurrent changes.

## Directory Policy

Include/exclude patterns and extension/media-type allowlists operate on relative
paths. Ignore files default to `.gitignore`, `.ragignore`, and `.ignore`, with
scoped git-style patterns and negation. Ignored directories are pruned; as with
Git, negation cannot re-include a descendant of a pruned parent unless that parent
is itself re-included. Explicit exclusions/allowlists remain policy boundaries.

The default does not follow symlinks. When enabled, symlinks remain constrained to
the configured root and directory cycles are detected. Secure descriptor-based
opens prevent a discovered pathname from being swapped for an external symlink
before loading. These filesystem safeguards target POSIX systems; do not silently
fall back to unsafe pathname reads on unsupported platforms.

Each scan is a live traversal, not a transactional filesystem snapshot. A host
requiring a point-in-time view must scan a filesystem snapshot or hold an
appropriate host-level lock. Periodic reconciliation converges ordinary edits.

## Extraction And Provenance

Plain text is decoded with the configured encoding. The effective encoding is
part of the extractor options fingerprint. HTML removes obvious navigation,
script/style, and landmark boilerplate conservatively, preserving article
headings, visible link text, and link targets. It does not execute JavaScript.

PDF extraction uses real PDF page-tree order. Each extracted page retains its
1-based `location['page']`. Built-in PDF extraction now defaults to **strict**:
only a page without content operations or annotations can be certified blank.
Textless images, drawings, forms, annotations and unknown content need OCR or
verification; a page combining text and visual content is partial. This is
deliberately conservative: even a logo or decorative rule may require a custom
verified extractor. The pypdf adapter cannot certify table fidelity, reading order,
or the semantic completeness of a text layer. Optional SDK OCR/layout adapters
are available through explicit configuration; see [SDK PDF extractors](pdf-extractors.md).

`PDFExtractor.extract_outcomes(data)` returns typed `PageExtractionOutcome`
records with `extracted`, `verified_blank`, `needs_ocr`, `partial`, `failed`, or
host-defined `policy_excluded` statuses. Connector results expose a bounded
`metadata['extraction_quality']` summary (complete, page count, status counts),
separate from traversal. The collector aggregates these summaries across files
and pages. A successful traversal cannot override failed extraction quality.
Only verified blanks or explicit exclusions produce intentional skip records.

Under strict policy a single unverified page rejects **all pages of that PDF
revision** with an incomplete, non-authoritative result. This prevents sync from
replacing a prior complete revision and prevents `purge_on_empty=True` from
deleting it. Other successfully extracted files in a non-authoritative sync can
still be updated; absence never authorizes deletion. Malformed/encrypted/zero-page
PDFs and page/byte-limit overflow also fail closed. Defaults are 1,000 pages and
50,000,000 source bytes; configure `options['max_pages']` / `options['max_bytes']`.
These bounds do not sandbox PDF decompression; isolate untrusted parsing in a
host-managed worker with memory and execution limits.

`PDFExtractor(options={'policy': 'review'})` withholds unverified revisions for
host review. Explicit `allow_partial` returns available text with page provenance
and incomplete quality; it never becomes authoritative and the legacy ingestion
pipeline refuses it. A sync host opting into this policy may publish the partial
document without deleting unseen documents. Choose this only when that tradeoff
is intentional. There is no implicit OCR or review service.

Migration: direct `extract()` and `extract_pages()` retain their string and
`(page, text)` return types, including empty pages. They are legacy text views,
not completeness certificates. Connector integrations use outcomes instead;
custom extractors without outcomes retain their existing caller-owned contract.
PDF extractor version 3 and effective policy/limits change fingerprints and force
reindexing. Non-PDF records keep their existing fingerprint behavior.

Diagnostics contain reason codes, page numbers, and a SHA-256 source identity,
not document text, provider messages or signed URLs. Keep the original bytes in
the host's private artifact store for retries; pass an opaque artifact reference
through source metadata/provenance if needed. Ragnor does not retain PDF bytes in
vector metadata or choose the host's artifact retention policy.

Actual extractor identity, version, and effective options feed ingestion
fingerprints. Original source provenance, source metadata, and location survive
chunking. Extraction changes force re-ingestion even when resulting text happens
to be unchanged. Reserved ingestion/version metadata remains pipeline-owned.

## Credentials And Network Boundaries

Inject credentials using `Credential`, headers, or an object-storage client;
never put credentials in source URLs or extractor options. Credential-bearing
URL userinfo and recognized token query parameters are rejected. Persisted HTTP
URL identities/options redact query values, and connector error reports avoid raw
exception messages. Redirects must not forward authentication to another origin
or downgrade HTTPS. Credentials are not part of ingestion fingerprints or logs.

The default HTTP transport validates resolved addresses and connects to a
validated public IP while preserving the original Host and TLS server name.
This avoids a DNS-check/connection-rebinding gap. Environment proxy settings are
not trusted implicitly. An explicitly injected transport is trusted application
code and must enforce its own equivalent guarantees. Retain deployment-level
egress firewall/proxy rules as defense in depth, especially in environments with
special network routes, service meshes, or private-network access policies.

## Legacy Sources, Incremental Loads, And Events

`DocumentSourceConnector` adapts an existing source to discovery and bounded
document pages without changing that source's `load()`. The legacy source may
still materialize its entire dataset; adaptation cannot create provider-side
streaming. `ConnectorDocumentSource` preserves the rich result when passing a
connector to the pipeline. Do not turn a partial result into a plain list before
ingestion, because that discards completeness evidence.

`FileDocumentSource.load()` still returns `[]` and logs a warning for a missing
file. This historical ambiguity remains only for direct legacy usage; the file
connector adapter reports missing separately. Arbitrary legacy sources must
honor their existing raise-on-failure contract.

`IncrementalSourceConnector.load_incremental(checkpoint=...)` is optional and
returns `incremental=True`. Persist checkpoints only after the host successfully
applies the batch. Absence from a delta never authorizes deletion; the snapshot
pipeline refuses to replace a full dataset with incremental results. Providers
may expose explicit tombstones/change events, applied idempotently by host sync
logic. Built-in connectors (`DirectoryConnector`, `ObjectStorageConnector`,
`HttpApiConnector`) implement `load_incremental(checkpoint=...)` to load
deltas modified after a checkpoint without mutating un-updated items.

`ChangeEventSource` is an optional provider event contract. Filesystem watcher
events are instead **sync requests**, not authoritative upserts/deletions.
`DirectoryConnector.changes(checkpoint=...)` and `FilesystemWatcher.changes(checkpoint=...)`
satisfy `ChangeEventSource`. `FilesystemWatcher.watch()` yields change or periodic requests;
without the optional native watcher it still yields periodic requests. The host consumes
these requests and starts reconciliation. `PeriodicReconciler.run(stop_event)`
performs periodic fresh scans; `run_once()` returns the result for a host-managed
schedule, and `sync(pipeline, version_store)` coordinates reconciliation through the
ingestion pipeline. Run reconciliation independently of event delivery to repair dropped
events. The host remains responsible for applying results through its pipeline.

Native SaaS integrations are intentionally not bundled into this contract. Ship
them as separate optional packages/extras after provider semantics stabilize.
