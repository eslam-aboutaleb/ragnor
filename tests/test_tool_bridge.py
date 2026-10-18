"""Unit tests for the framework-neutral KnowledgeBridge."""

from __future__ import annotations

import pytest

from ragit.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragit.evidence.types import ContextBudget, EvidenceStatus, ScopeAuthorizationError
from ragit.tools.bridge import KnowledgeBridge
from ragit.tools.security import RateLimitExceededError, RateLimiter, ScopeAuthorizer
from ragit.tools.validation import SecurityValidationError


class MockRetriever:
    """Simple deterministic mock retriever."""

    def __init__(self, items: list[dict] | None = None) -> None:
        self.items = items or []

    async def retrieve(self, query: str, n_results: int = 5, **filters) -> list[dict]:
        results = []
        for item in self.items:
            # Check filters
            match = True
            for k, v in filters.items():
                if k == "isolation_scope":
                    if item.get("isolation_scope") != v:
                        match = False
                        break
                elif item.get(k) != v and item.get("metadata", {}).get(k) != v:
                    match = False
                    break
            if match:
                results.append(item)
        return results[:n_results]


@pytest.fixture
def snapshot_store() -> MemorySnapshotStore:
    store = MemorySnapshotStore()
    store.put(
        SnapshotRecord(
            source_id="repo1",
            document_id="doc1",
            revision="v1",
            content="Line 1: System requirements.\nLine 2: Minimum RAM is 16GB.\nLine 3: OS is Linux.",
            isolation_scope="tenant_a",
            metadata={"title": "System Manual"},
        )
    )
    return store


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever(
        [
            {
                "id": "c1",
                "document": "Minimum RAM is 16GB for production deployments.",
                "isolation_scope": "tenant_a",
                "source_id": "repo1",
                "document_id": "doc1",
                "revision": "v1",
                "page": 1,
                "title": "System Manual",
            },
            {
                "id": "c2",
                "document": "Testing requires 8GB of memory.",
                "isolation_scope": "tenant_b",
                "source_id": "repo1",
                "document_id": "doc2",
                "revision": "v1",
                "page": 2,
                "title": "Test Manual",
            },
        ]
    )


@pytest.fixture
def bridge(mock_retriever: MockRetriever, snapshot_store: MemorySnapshotStore) -> KnowledgeBridge:
    authorizer = ScopeAuthorizer(
        scope_mapping={
            "client_a": {"tenant_a"},
            "client_b": {"tenant_b"},
            "admin": {"*"},
        }
    )
    return KnowledgeBridge(
        retriever=mock_retriever,
        source_lookup=SourceLookup(snapshot_store),
        authorizer=authorizer,
        allow_management=False,
    )


async def test_search_knowledge_authorized_scope(bridge: KnowledgeBridge) -> None:
    """Authorized tenant searches successfully and hits match scope."""
    bundle = await bridge.search_knowledge(
        query="minimum ram",
        scope="tenant_a",
        client_id="client_a",
    )
    assert bundle.status == EvidenceStatus.SUCCESS
    assert len(bundle.items) == 1
    assert bundle.items[0].document_id == "doc1"
    assert "16GB" in bundle.items[0].text


async def test_search_knowledge_unauthorized_scope_rejected(bridge: KnowledgeBridge) -> None:
    """Client attempting to search unauthorized scope raises ScopeAuthorizationError."""
    with pytest.raises(ScopeAuthorizationError):
        await bridge.search_knowledge(
            query="minimum ram",
            scope="tenant_b",
            client_id="client_a",  # client_a is only authorized for tenant_a
        )


async def test_search_knowledge_rejects_unsafe_inputs(bridge: KnowledgeBridge) -> None:
    """Input sanitization rejects credentials, path traversal, and SQL injection."""
    # Path traversal in query
    with pytest.raises(SecurityValidationError, match="path or traversal"):
        await bridge.search_knowledge(query="read ../../../etc/passwd")

    # Credential pattern in query
    with pytest.raises(SecurityValidationError, match="credential or secret"):
        await bridge.search_knowledge(query="token is Bearer a1b2c3d4e5f6g7h8i9j0")

    # SQL injection in filter values
    with pytest.raises(SecurityValidationError, match="SQL injection"):
        await bridge.search_knowledge(
            query="valid search",
            filters={"category": "admin' OR 1=1; --"},
        )

    # Unsafe collection identifier
    with pytest.raises(SecurityValidationError):
        await bridge.search_knowledge(
            query="valid",
            collection="bad/collection",
        )


async def test_search_knowledge_budget_enforcement(bridge: KnowledgeBridge) -> None:
    """ContextBudget limits the number of assembled evidence items."""
    budget = ContextBudget(total_budget=250, max_evidence_items=1)
    bundle = await bridge.search_knowledge(
        query="ram memory",
        scope="tenant_a",
        client_id="admin",
        budget_limit=budget,
    )
    assert bundle.status == EvidenceStatus.SUCCESS
    assert len(bundle.items) <= 1


async def test_get_source_authorized_span(bridge: KnowledgeBridge) -> None:
    """get_source returns authorized bounded span for valid revision."""
    result = await bridge.get_source(
        source_ref="repo1",
        document_id="doc1",
        revision="v1",
        scope="tenant_a",
        client_id="client_a",
        start_offset=0,
        end_offset=30,
    )
    assert result.status == EvidenceStatus.SUCCESS
    assert "System requirements" in result.text
    assert result.revision == "v1"


async def test_get_source_scope_mismatch_fails_safely(bridge: KnowledgeBridge) -> None:
    """get_source with wrong tenant scope returns failure without leaking text."""
    result = await bridge.get_source(
        source_ref="repo1",
        document_id="doc1",
        revision="v1",
        scope="tenant_b",
        client_id="client_b",  # client_b asks for tenant_b, but doc1 belongs to tenant_a
    )
    assert result.status != EvidenceStatus.SUCCESS
    assert result.text == ""


async def test_get_source_rejects_unsafe_source_ref(bridge: KnowledgeBridge) -> None:
    """get_source rejects file paths and SSRF URLs."""
    with pytest.raises(SecurityValidationError):
        await bridge.get_source(source_ref="/etc/passwd")

    with pytest.raises(SecurityValidationError):
        await bridge.get_source(source_ref="file:///etc/hosts")


async def test_knowledge_status_sanitized(bridge: KnowledgeBridge) -> None:
    """knowledge_status returns sanitized summary without leaking internals."""
    status = await bridge.knowledge_status(
        collection="default",
        scope="tenant_a",
        client_id="admin",
    )
    assert status["is_healthy"] is True
    assert status["collection"] == "default"
    assert "password" not in status


async def test_management_tools_guarded_by_permissions(bridge: KnowledgeBridge) -> None:
    """Management operations raise PermissionError when allow_management=False."""
    with pytest.raises(PermissionError, match="disabled by host security rules"):
        await bridge.sync_collection(
            collection="default",
            source_id="repo1",
            documents=[{"id": "doc10", "text": "New doc"}],
        )

    with pytest.raises(PermissionError, match="disabled by host security rules"):
        await bridge.revoke_document(
            collection="default",
            document_id="doc1",
            source_id="repo1",
        )


async def test_management_tools_allowed_when_enabled(
    mock_retriever: MockRetriever, snapshot_store: MemorySnapshotStore
) -> None:
    """Management operations succeed when allow_management=True."""
    bridge = KnowledgeBridge(
        retriever=mock_retriever,
        source_lookup=SourceLookup(snapshot_store),
        allow_management=True,
    )
    res = await bridge.sync_collection(
        collection="default",
        source_id="repo1",
        documents=[{"id": "doc10", "text": "New document content", "revision": "v1"}],
    )
    assert res["status"] == "success"

    # Verify snapshot was registered in source lookup
    snapshot = snapshot_store.get("repo1", "doc10", "v1")
    assert snapshot is not None
    assert snapshot.content == "New document content"


async def test_rate_limiter() -> None:
    """RateLimiter enforces max requests per minute and concurrency."""
    limiter = RateLimiter(requests_per_minute=2, max_concurrent=2)
    await limiter.acquire("test_client")
    await limiter.acquire("test_client")

    # Exceed requests per minute
    with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
        await limiter.acquire("test_client")

    await limiter.release("test_client")
    await limiter.release("test_client")
