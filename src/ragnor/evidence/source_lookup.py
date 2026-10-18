"""Source document lookup with scope authorization and immutable revision checks.

Enforces:
1. Scope authorization: caller scope must match snapshot scope.
2. Immutable historical snapshots: if a cited revision is unavailable,
   returns ``revision_unavailable`` (or raises :class:`RevisionUnavailableError`).
   Never silently substitutes current or newer content for an old citation.
3. Bounded authorized spans: returns only bounded text spans.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ragnor.evidence.types import (
    EvidenceStatus,
    RevisionUnavailableError,
    ScopeAuthorizationError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotRecord:
    """An immutable retained snapshot of a document at a specific revision.

    Attributes:
        source_id: Source repository or system identifier.
        document_id: Document primary identifier.
        revision: Stated revision or version hash.
        content: Immutable raw text content of the document at this revision.
        isolation_scope: Mandatory tenant or collection scope required to access.
        metadata: Associated document metadata.
        created_at: ISO timestamp when snapshot was captured.
    """

    source_id: str
    document_id: str
    revision: str
    content: str
    isolation_scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@runtime_checkable
class SnapshotStore(Protocol):
    """Protocol for storing and retrieving immutable document revision snapshots."""

    def put(self, snapshot: SnapshotRecord) -> None:
        """Store an immutable snapshot record."""
        ...

    def get(self, source_id: str, document_id: str, revision: str) -> SnapshotRecord | None:
        """Retrieve an immutable snapshot record by source, document, and revision."""
        ...

    def has_revision(self, source_id: str, document_id: str, revision: str) -> bool:
        """Return whether an exact revision snapshot is retained."""
        ...


class MemorySnapshotStore:
    """In-memory implementation of :class:`SnapshotStore`."""

    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str, str, str | None], SnapshotRecord] = {}
        self._revoked: set[tuple[str, str, str | None]] = set()

    def put(self, snapshot: SnapshotRecord) -> None:
        """Save a snapshot record."""
        key = (
            snapshot.source_id,
            snapshot.document_id,
            snapshot.revision,
            snapshot.isolation_scope,
        )
        existing = self._snapshots.get(key)
        if existing is not None:
            if existing.content != snapshot.content or existing.metadata != snapshot.metadata:
                raise ValueError("Cannot replace an immutable document revision")
            return
        self._snapshots[key] = deepcopy(snapshot)

    def get(self, source_id: str, document_id: str, revision: str) -> SnapshotRecord | None:
        """Get snapshot by source, document, and revision."""
        matches = [
            record
            for key, record in self._snapshots.items()
            if key[:3] == (source_id, document_id, revision)
        ]
        if len(matches) > 1:
            raise ScopeAuthorizationError("Snapshot lookup requires an isolation scope")
        return deepcopy(matches[0]) if matches else None

    def get_scoped(
        self, source_id: str, document_id: str, revision: str, scope: str | None
    ) -> SnapshotRecord | None:
        """Select an exact scoped revision, or an explicitly public revision."""
        for candidate_scope in (scope, None):
            key = (source_id, document_id, revision, candidate_scope)
            if key in self._snapshots:
                return deepcopy(self._snapshots[key])
        return None

    def revoke(self, source_id: str, document_id: str, scope: str | None) -> None:
        """Retain historical bytes while denying access to every revision."""
        self._revoked.add((source_id, document_id, scope))

    def is_revoked(self, source_id: str, document_id: str, scope: str | None) -> bool:
        return (source_id, document_id, scope) in self._revoked or (
            source_id,
            document_id,
            None,
        ) in self._revoked

    def has_revision(self, source_id: str, document_id: str, revision: str) -> bool:
        """Check if revision exists."""
        return any(key[:3] == (source_id, document_id, revision) for key in self._snapshots)


@dataclass
class BoundedSpanResult:
    """Result of an authorized bounded span lookup.

    Attributes:
        source_id: Source system identifier.
        document_id: Document identifier.
        revision: Resolved revision identifier.
        text: Bounded text span.
        start_offset: Start offset of the returned span.
        end_offset: End offset of the returned span.
        total_document_length: Total character length of the document revision.
        is_truncated: Whether the requested span was clamped to maximum allowed length.
        status: EvidenceStatus indicating success or failure.
    """

    source_id: str
    document_id: str
    revision: str
    text: str
    start_offset: int
    end_offset: int
    total_document_length: int
    is_truncated: bool
    status: EvidenceStatus

    def to_dict(self) -> dict[str, Any]:
        """Convert bounded span result to a structured dictionary representation."""
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "revision": self.revision,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "total_document_length": self.total_document_length,
            "is_truncated": self.is_truncated,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
        }


class SourceLookup:
    """Authorized source document lookup engine.

    Requires the same authorization scope as search and returns only
    bounded authorized spans from immutable retained snapshots.
    """

    def __init__(
        self,
        snapshot_store: SnapshotStore,
        default_max_span_length: int = 4000,
    ) -> None:
        """Initialize SourceLookup.

        Args:
            snapshot_store: Store holding immutable revision snapshots.
            default_max_span_length: Maximum allowed character length for returned spans.
        """
        self._store = snapshot_store
        self._validate_span_limit(default_max_span_length)
        self.default_max_span_length = default_max_span_length

    @staticmethod
    def _validate_span_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_span_length must be a positive integer")

    def revoke(self, source_id: str, document_id: str, scope: str | None) -> None:
        """Persist revocation through a store supporting document access control."""
        if not all(hasattr(self._store, method) for method in ("revoke", "is_revoked")):
            raise TypeError("Snapshot store must implement revoke and is_revoked for management")
        self._store.revoke(source_id, document_id, scope)

    def lookup_span(  # noqa: PLR0913, PLR0917
        self,
        source_id: str,
        document_id: str,
        revision: str,
        authorization_scope: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        max_span_length: int | None = None,
        strict: bool = False,
    ) -> BoundedSpanResult:
        """Retrieve a bounded span from an immutable revision snapshot.

        Args:
            source_id: Source repository identifier.
            document_id: Document identifier.
            revision: Stated revision identifier.
            authorization_scope: Caller's isolation scope (e.g. tenant_id).
            start_offset: Optional character start offset.
            end_offset: Optional character end offset.
            max_span_length: Maximum allowed span length in characters.
            strict: If True, raises exceptions instead of returning error statuses.

        Returns:
            :class:`BoundedSpanResult` with the slice of text and provenance.

        Raises:
            RevisionUnavailableError: If exact revision is not retained and strict=True.
            ScopeAuthorizationError: If scope check fails and strict=True.
        """
        span_limit = self.default_max_span_length if max_span_length is None else max_span_length
        self._validate_span_limit(span_limit)
        span_limit = min(span_limit, self.default_max_span_length)
        try:
            if hasattr(self._store, "is_revoked") and self._store.is_revoked(
                source_id, document_id, authorization_scope
            ):
                raise ScopeAuthorizationError("Document access has been revoked")
            if hasattr(self._store, "get_scoped"):
                snapshot = self._store.get_scoped(
                    source_id, document_id, revision, authorization_scope
                )
                if snapshot is None and self._store.has_revision(source_id, document_id, revision):
                    raise ScopeAuthorizationError("Document revision belongs to a different scope")
            else:
                snapshot = self._store.get(source_id, document_id, revision)
        except ScopeAuthorizationError:
            if strict:
                raise
            return BoundedSpanResult(
                source_id,
                document_id,
                revision,
                "",
                0,
                0,
                0,
                False,
                EvidenceStatus.BACKEND_FAILURE,
            )

        # Historical revision availability check
        if snapshot is None:
            msg = (
                f"Historical revision {revision!r} is unavailable for document {document_id!r} "
                f"in source {source_id!r}. Never substituting newer content for old citations."
            )
            logger.warning(msg)
            if strict:
                raise RevisionUnavailableError(msg)
            return BoundedSpanResult(
                source_id=source_id,
                document_id=document_id,
                revision=revision,
                text="",
                start_offset=start_offset or 0,
                end_offset=end_offset or 0,
                total_document_length=0,
                is_truncated=False,
                status=EvidenceStatus.REVISION_UNAVAILABLE,
            )

        # Scope authorization check
        if snapshot.isolation_scope is not None and snapshot.isolation_scope != authorization_scope:
            msg = (
                f"Scope authorization failure for {document_id!r}: document scope is "
                f"{snapshot.isolation_scope!r}, caller presented {authorization_scope!r}."
            )
            logger.warning(msg)
            if strict:
                raise ScopeAuthorizationError(msg)
            return BoundedSpanResult(
                source_id=source_id,
                document_id=document_id,
                revision=revision,
                text="",
                start_offset=start_offset or 0,
                end_offset=end_offset or 0,
                total_document_length=len(snapshot.content),
                is_truncated=False,
                status=EvidenceStatus.BACKEND_FAILURE,
            )

        # Extract bounded span
        content = snapshot.content
        doc_len = len(content)

        start = max(0, start_offset) if start_offset is not None else 0
        end = min(doc_len, end_offset) if end_offset is not None else doc_len

        start = min(start, doc_len)
        end = max(end, start)

        is_truncated = False
        if end - start > span_limit:
            end = start + span_limit
            is_truncated = True

        span_text = content[start:end]

        return BoundedSpanResult(
            source_id=source_id,
            document_id=document_id,
            revision=revision,
            text=span_text,
            start_offset=start,
            end_offset=end,
            total_document_length=doc_len,
            is_truncated=is_truncated,
            status=EvidenceStatus.SUCCESS,
        )
