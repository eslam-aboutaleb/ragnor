# Sync Coordination, Consistency Guarantees, and Deletion Privacy

This document defines the architecture, coordination algorithm, backend consistency levels, and privacy guarantees of `ragit`'s sync subsystem (`ragit.sync`).

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
   - Marks manifest record status as `STAGED`. Staged chunks remain completely invisible to readers.
   - Re-uses the outbox pattern (`EmbeddingJobStore`) with idempotency keys (`{scope}:{doc_id}:{staged_gen}:embed`) for worker convergence.

5. **Fencing Verification & Activation**:
   - Re-verifies the fencing token (`state_store.validate_token`). If the lease expired or another worker acquired a higher epoch, the worker aborts (`StaleWorkerError`), preventing stale workers from activating obsolete revisions.
   - **Document-Level Activation (Default)**:
     - Activates each successfully staged document independently using Compare-And-Swap (CAS) on `ManifestRecord.fence_token`.
     - Flips vector store chunks to `active=True`.
     - One failing or broken document does not block unrelated document updates.
   - **Collection-Level Activation (Opt-In)**:
     - For policy releases needing all-or-nothing visibility.
     - Advances collection generation atomically (`state_store.advance_generation`).
     - If any document fails embedding or validation, collection generation is not advanced and staging data remains unactivated.

6. **Authoritative Tombstone Enforcement**:
   - Evaluated **only** when scan coverage is authoritative (`authoritative=True`, no errors, no pagination interruptions).
   - Partial scans, connector errors, and incremental listings **never** authorize deletions.
   - Authoritatively deleted documents are transitioned to `status=TOMBSTONED` and `active_generation=0`.

7. **Checkpoints, Async Garbage Collection & Lease Release**:
   - Saves final sync checkpoint with summary statistics.
   - Asynchronously triggers garbage collection (`vector_store.delete(ids=retired_chunk_ids)`).
   - Releases the scoped lease.

---

## 3. Backend Consistency Levels (No Cross-Database Atomicity)

`ragit` explicitly does **not** promise cross-database atomicity between independent vector stores and relational/manifest databases. Two-phase commit (2PC) is not implemented. Instead, consistency and convergence are guaranteed through generation fencing, CAS activation, and zero-trust retrieval filtering.

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
