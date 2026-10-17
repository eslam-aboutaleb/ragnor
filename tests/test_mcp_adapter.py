"""Unit tests for the MCP server adapter and stdio transport."""

from __future__ import annotations

import json
import pytest

from ragit.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragit.mcp.server import create_mcp_server
from ragit.stores.memory import InMemoryVectorStore
from ragit.tools.bridge import KnowledgeBridge


class MockEmbedder:
    """Mock embedder returning fixed dimension vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def __call__(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, list):
            return [self.embed(t) for t in texts]
        return self.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        return self.embed(query)


@pytest.fixture
def test_bridge() -> KnowledgeBridge:
    embedder = MockEmbedder(dim=4)
    store = InMemoryVectorStore(embedding_dim=4, embedding_fn=embedder)
    snapshot_store = MemorySnapshotStore()
    source_lookup = SourceLookup(snapshot_store)

    snapshot_store.put(
        SnapshotRecord(
            source_id="mcp_source",
            document_id="doc_mcp_1",
            revision="v1",
            content="Chapter 1: MCP Server Adapter integration guide.",
            isolation_scope="default",
            metadata={"title": "MCP Guide"},
        )
    )

    return KnowledgeBridge(
        retriever=store,
        source_lookup=source_lookup,
        allow_management=True,
    )


async def test_mcp_server_registers_read_tools_by_default(test_bridge: KnowledgeBridge) -> None:
    """MCP server by default only exposes read tools, not management tools."""
    server = create_mcp_server(test_bridge, name="test-ragit", enable_management_tools=False)
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert "search_knowledge" in tool_names
    assert "get_source" in tool_names
    assert "knowledge_status" in tool_names
    assert "sync_collection" not in tool_names
    assert "revoke_document" not in tool_names


async def test_mcp_server_registers_management_tools_when_enabled(
    test_bridge: KnowledgeBridge,
) -> None:
    """When enable_management_tools=True, sync and deletion tools are registered."""
    server = create_mcp_server(test_bridge, name="test-ragit", enable_management_tools=True)
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert "search_knowledge" in tool_names
    assert "get_source" in tool_names
    assert "knowledge_status" in tool_names
    assert "sync_collection" in tool_names
    assert "revoke_document" in tool_names


async def test_mcp_call_knowledge_status(test_bridge: KnowledgeBridge) -> None:
    """Calling knowledge_status through MCP returns sanitized status."""
    server = create_mcp_server(test_bridge, name="test-ragit")
    res = await server.call_tool("knowledge_status", {"collection": "default"})
    assert not res.is_error
    text_content = res.content[0].text
    status_data = json.loads(text_content)
    assert status_data["collection"] == "default"
    assert status_data["is_healthy"] is True


async def test_mcp_call_get_source(test_bridge: KnowledgeBridge) -> None:
    """Calling get_source through MCP returns authorized span."""
    server = create_mcp_server(test_bridge, name="test-ragit")
    res = await server.call_tool(
        "get_source",
        {
            "source_ref": "mcp_source",
            "document_id": "doc_mcp_1",
            "revision": "v1",
            "start_offset": 0,
            "end_offset": 20,
        },
    )
    assert not res.is_error
    span_data = json.loads(res.content[0].text)
    assert span_data["status"] == "success"
    assert "Chapter 1: MCP" in span_data["text"]


async def test_mcp_call_search_knowledge(test_bridge: KnowledgeBridge) -> None:
    """Calling search_knowledge through MCP returns structured EvidenceBundle."""
    # Seed a chunk
    await test_bridge.sync_collection(
        collection="default",
        source_id="mcp_source",
        documents=[
            {
                "id": "c_mcp",
                "text": "MCP Server Adapter integration guide.",
                "embedding": [0.1, 0.1, 0.1, 0.1],
                "revision": "v1",
            }
        ],
    )

    server = create_mcp_server(test_bridge, name="test-ragit")
    res = await server.call_tool(
        "search_knowledge",
        {"query": "MCP Server Adapter", "candidate_k": 3},
    )
    assert not res.is_error
    bundle_data = json.loads(res.content[0].text)
    assert bundle_data["status"] == "success"
    assert len(bundle_data["items"]) >= 1
    assert "MCP Server Adapter" in bundle_data["items"][0]["text"]
