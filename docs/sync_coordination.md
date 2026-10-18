# Sync Coordination, Consistency Guarantees, and Deletion Privacy

This document defines the architecture, coordination algorithm, backend consistency levels, and privacy guarantees of `ragnor`'s sync subsystem (`ragnor.sync`).

---

## 1. Core Protocols and Storage Seams

- **`ManifestStore`**: Tracks document-level metadata, observed revision hints, content hashes, extractor/chunker/embedding fingerprints, active and staged generations, chunk IDs, and lifecycle status (`ACTIVE`, `STAGED`, `TOMBSTONED`, `FAILED`).
- **`SyncStateStore`**: Coordinates distributed workers using scoped leases, monotonically increasing fencing epochs (`FenceToken`), collection-level generations, and sync observation checkpoints.
- **`SyncCoordinator`**: Executes the 7-step manifest-driven synchronization algorithm.
- **`SyncRetriever`**: Provides defense-in-depth retrieval filtering enforcing mandatory scope, active generation, and immediate tombstone protection.

---

## 2. The 7-Step Synchronization Algorithm

1. **Scoped Lease & Fencing Token Acquisition**:
   - The coordinator acquires a scoped lease for the namespace/collection/source (`state_store.acquire_lease`).
   - Obtains a monotonic fencing token (`epoch`). If a lease cannot be acquired or has not expired, execution aborts to prevent concurrent split-brain writes.
   - Captures current generation and configuration fingerprints (`extractor`, `chunker`, `embedding`).

2. **Bounded Scan & Checkpoint Observations**:
   - The connector scans pages incrementally or via bounded discovery (`discover` / `load_page` / `load`).
   - Checkpoints cursors and observed item counts without deleting unseen documents.
   - Preserves whether the traversal achieved complete, authoritative coverage (`page.authoritative`).

3. **Differential Plan Computation**:
   - Compares observed items with the manifest store:
     - **`ADDED`**: New document ID or document currently marked `TOMBSTONED`.
     - **`CHANGED`**: Content hash mismatch, or configuration fingerprint change.
     - **`METADATA_ONLY`**: Same content hash (or unchanged opaque revision hint), but document metadata differs.
     - **`UNCHANGED`**: Same content hash and metadata.
     - **`REMOVED`**: Documents in manifest missing from the scan (*only evaluated if the scan is fully authoritative*).
   - **Opaque Revision Hints**: ETags, mtime, or version strings are treated strictly as revision hints, never universal content hashes. If an ETag is identical, content hashing is avoided; if an ETag changes, content hash is verified before triggering re-embedding.

4. **Bounded Batch Embedding & Staging**:
   - Chunks and embeds added or changed documents in bounded batches (`batch_size`).
   - Assigns unique generation-scoped IDs (`{scope}:{doc_id}:g{staged_gen}:{chunk_index}`).
   - Upserts chunks to the vector store with `active=False` (staging).
   - Chunk IDs also include the lease epoch and a unique attempt identifier. A delayed write from an expired worker cannot overwrite another attempt's vectors.
   - Leaves the committed manifest unchanged while preparing a generation. Staged chunks remain invisible to readers; an update failure preserves the old active content and fingerprints.
   - Metadata-only updates copy stored vectors into a new generation without invoking the embedding provider. Text and chunk locations are retained, and removed metadata keys disappear from the new payloads.
   - Enqueues optional `pending` outbox reconciliation jobs with epoch-specific idempotency keys and exact chunk IDs. The host's `JobProcessor` must check manifest membership before acting on a job: abandoned attempts must never be activated by replaying a stale job.

5. **Fencing Verification & Activation**:
   - Renews the lease during preparation and enters `state_store.fenced(token)` before publishing visibility. This context rejects stale workers and excludes lease takeover for the duration of the commit; a separate token check followed by an unguarded write is insufficient.
   - **Document-Level Activation (Default)**:
     - Publishes successfully prepared vectors with `active=True`, then commits their exact IDs in the manifest using Compare-And-Swap (CAS). Readers reject prepublished vectors whose IDs are absent from the committed manifest.
     - One failing or broken document does not block unrelated document updates.
   - **Collection-Level Activation (Opt-In)**:
     - For policy releases needing all-or-nothing visibility.
     - Publishes all changed vectors, then atomically switches all affected manifest records with `compare_and_swap_many`. Readers use one manifest snapshot per search. The collection generation counter is advanced within the same fenced commit context.
     - If preparation fails, neither committed manifests nor the collection generation change. Retrying compares against the last committed fingerprints and prepares the update again.

6. **Authoritative Tombstone Enforcement**:
   - Evaluated **only** when scan coverage is authoritative (`authoritative=True`, no errors, no pagination interruptions).
   - Partial scans, connector errors, and incremental listings **never** authorize deletions.
   - Authoritatively deleted documents are transitioned to `status=TOMBSTONED` and `active_generation=0`.

7. **Checkpoints, Async Garbage Collection & Lease Release**:
   - Saves scan authority, errors, observation time and the last successful sync time with summary statistics. An interrupted or failed scan is not reported as an identical successful scan.
   - Cleans up retired and abandoned attempt IDs (`vector_store.delete(ids=retired_chunk_ids)`). Physical cleanup failures can be retried; manifest checks continue to enforce visibility.
   - Releases the scoped lease.

---

## 3. Backend Consistency Levels (No Cross-Database Atomicity)

`ragnor` explicitly does **not** promise cross-database atomicity between independent vector stores and relational/manifest databases. Two-phase commit (2PC) is not implemented. Instead, consistency and convergence are guaranteed through generation fencing, CAS activation, and zero-trust retrieval filtering.

| Backend | Transactionality | Write Visibility & Consistency | Fencing / Concurrency Support |
| :--- | :--- | :--- | :--- |
| **In-Memory Store** | Process-level locks | Immediate read-after-write within process memory. | Asyncio locks and atomic dictionary CAS. |
| **PostgreSQL / pgvector** | Full ACID (per-session) | Read-committed or serializable within database transactions. | Native advisory locks (`pg_advisory_lock`) and row-level fencing. |
| **Qdrant** | Write-Ahead Log (WAL) | Configurable write acknowledgement (`wait=true`). Eventual replica consistency unless read consistency is enforced. | Manifest-driven generation gating prevents reading uncommitted vectors. |
| **Chroma** | SQLite / DuckDB or HTTP | Local ACID on SQLite; HTTP mode depends on server consistency. | External synchronization via `SyncStateStore` leases. |
| **FAISS** | Non-transactional | In-memory index. Index mutations require rebuilding or cell addition. | External mutual exclusion via fencing tokens mandatory. |

---

## 4. Deletion Privacy & Retrieval Filtering Policy

- **Ordinary Ingestion Failure**:
  - If a document update fails embedding or staging, the manifest retains the last known good active generation. Previous permitted content remains intact and queryable.
- **Authoritative Deletion & Access Revocation**:
  - Authoritative deletion or explicit access revocation (`coordinator.revoke_access(scope, doc_id)`) immediately updates the manifest record to `status=TOMBSTONED` and `active_generation=0`.
  - `SyncRetriever` applies mandatory runtime filtering on every retrieval candidate:
    1. Query filter: `scope == scope` AND `active == True`.
    2. Defense-in-depth manifest check: Candidates whose document is `TOMBSTONED`, `FAILED`, or whose generation does not match `rec.active_generation` are **immediately dropped**.
  - **Zero-Trust Privacy Guarantee**: Even if physical deletion in the underlying vector store is asynchronous, delayed, or retrying after network errors, revoked or deleted data is **never** served to readers.

## Host adapter requirements

Custom `SyncStateStore` adapters must implement `fenced(token)` using a durable transaction or lock shared by every lease-acquisition path. The context must prevent a new epoch from acquiring ownership while a manifest commit is in progress. `ManifestStore.compare_and_swap_many(updates)` must validate all expected record versions and publish all records atomically, or publish none. The coordinator fails before mutation when these capabilities are absent.

Vector stores used for metadata-only synchronization must implement `get_embeddings(ids, session=None) -> dict[str, list[float]]`. All built-in stores provide it. A failed or incomplete vector fetch leaves the committed document untouched. Durable manifest adapters should return independent record values rather than mutable references to stored state.

The coordinator accepts legacy `DocumentSource` objects, connectors, document sequences and `LoadResult` values. Complete pagination from the beginning can establish authority; a resumed, interrupted, failed or incremental observation cannot authorize absence-based deletion, even when the caller passes `is_authoritative=True`. Multiple artifacts with the same parent ID must carry distinct locations, such as PDF page numbers.
