"""Tests for SourceLookup, scope authorization, and historical revision checks."""

from __future__ import annotations

import pytest

from ragit.evidence.source_lookup import (
    MemorySnapshotStore,
    SnapshotRecord,
    SourceLookup,
)
from ragit.evidence.types import (
    EvidenceStatus,
    RevisionUnavailableError,
    ScopeAuthorizationError,
)


def test_citations_resolve_to_stated_revision() -> None:
    store = MemorySnapshotStore()
    lookup = SourceLookup(store)

    # Store revision 1 and revision 2 of doc1
    store.put(
        SnapshotRecord(
            source_id="repoA",
            document_id="doc1",
            revision="rev1",
            content="Historical text from revision 1 that was cited in the past.",
            isolation_scope="tenant_omega",
        )
    )
    store.put(
        SnapshotRecord(
            source_id="repoA",
            document_id="doc1",
            revision="rev2",
            content="Completely rewritten current text from revision 2.",
            isolation_scope="tenant_omega",
        )
    )

    # Query specifically for old revision rev1
    res = lookup.lookup_span(
        source_id="repoA",
        document_id="doc1",
        revision="rev1",
        authorization_scope="tenant_omega",
    )

    assert res.status == EvidenceStatus.SUCCESS
    assert res.revision == "rev1"
    assert "revision 1" in res.text
    assert "revision 2" not in res.text


def test_missing_revision_returns_revision_unavailable_never_substituting_current() -> None:
    store = MemorySnapshotStore()
    lookup = SourceLookup(store)

    # Only current revision 3 is in store
    store.put(
        SnapshotRecord(
            source_id="repoA",
            document_id="doc1",
            revision="rev3",
            content="Current revision 3 content.",
            isolation_scope="tenant_omega",
        )
    )

    # Looking up un-retained historical rev1 must NOT return rev3!
    res = lookup.lookup_span(
        source_id="repoA",
        document_id="doc1",
        revision="rev1",
        authorization_scope="tenant_omega",
        strict=False,
    )
    assert res.status == EvidenceStatus.REVISION_UNAVAILABLE
    assert res.text == ""

    # In strict mode, raises RevisionUnavailableError
    with pytest.raises(RevisionUnavailableError) as exc_info:
        lookup.lookup_span(
            source_id="repoA",
            document_id="doc1",
            revision="rev1",
            authorization_scope="tenant_omega",
            strict=True,
        )
    assert "rev1" in str(exc_info.value)
    assert "Never substituting newer content" in str(exc_info.value)


def test_scope_authorization_enforced() -> None:
    store = MemorySnapshotStore()
    lookup = SourceLookup(store)

    store.put(
        SnapshotRecord(
            source_id="repo_secret",
            document_id="payroll",
            revision="rev1",
            content="Confidential compensation data.",
            isolation_scope="finance_dept",
        )
    )

    # Mismatched scope fails authorization
    res = lookup.lookup_span(
        source_id="repo_secret",
        document_id="payroll",
        revision="rev1",
        authorization_scope="marketing_dept",
        strict=False,
    )
    assert res.status == EvidenceStatus.BACKEND_FAILURE
    assert res.text == ""

    # In strict mode, raises ScopeAuthorizationError
    with pytest.raises(ScopeAuthorizationError) as exc_info:
        lookup.lookup_span(
            source_id="repo_secret",
            document_id="payroll",
            revision="rev1",
            authorization_scope="marketing_dept",
            strict=True,
        )
    assert "different scope" in str(exc_info.value)


def test_bounded_span_offsets_and_max_length() -> None:
    store = MemorySnapshotStore()
    lookup = SourceLookup(store, default_max_span_length=50)

    full_text = "0123456789" * 20  # 200 chars
    store.put(
        SnapshotRecord(
            source_id="src1",
            document_id="doc1",
            revision="v1",
            content=full_text,
        )
    )

    # Slice from offset 20 to 40
    res = lookup.lookup_span("src1", "doc1", "v1", start_offset=20, end_offset=40)
    assert res.status == EvidenceStatus.SUCCESS
    assert len(res.text) == 20
    assert res.is_truncated is False

    # Oversized span clamped to default_max_span_length (50)
    res_large = lookup.lookup_span("src1", "doc1", "v1", start_offset=0, end_offset=150)
    assert len(res_large.text) == 50
    assert res_large.is_truncated is True
