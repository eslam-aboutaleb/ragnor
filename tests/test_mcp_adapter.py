"""Unit tests for the MCP server adapter and stdio transport."""

from __future__ import annotations

import json
from typing import Any
import pytest

from ragit.evidence.source_lookup import MemorySnapshotStore, SnapshotRecord, SourceLookup
from ragit.mcp.server import _MCPServerClass, create_mcp_server
from ragit.stores.memory import InMemoryVectorStore
from ragit.tools.bridge import KnowledgeBridge


class MockTool:
    """Mock representation of a registered MCP tool."""

    def __init__(self, name: str, fn: Any, description: str = "") -> None:
        self.name = name
        self.fn = fn
        self.description = description


class MockMCPServer:
    """Lightweight duck-typed mock for MCPServer / FastMCP."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, MockTool] = {}

    def tool(self, name: str, description: str = "") -> Any:
        def decorator(fn: Any) -> Any:
            self._tools[name] = MockTool(name=name, fn=fn, description=description)
            return fn

        return decorator

    async def list_tools(self) -> list[MockTool]:
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools[name]
        return await tool.fn(**arguments)


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


def _tool_payload(result: Any) -> dict[str, Any]:
    """Normalize the public call_tool response from either supported SDK API."""
    if hasattr(result, "content"):
        assert not result.is_error
        content = result.content
    elif isinstance(result, tuple):
        content, _ = result
    else:
        content = result
    return json.loads(content[0].text)


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


def test_create_mcp_server_raises_when_no_sdk(
    test_bridge: KnowledgeBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_mcp_server raises friendly ImportError when mcp extra is not available."""
    monkeypatch.setattr("ragit.mcp.server._MCPServerClass", None)
    with pytest.raises(ImportError, match="requires the 'mcp' package"):
        create_mcp_server(test_bridge, server_class=None)


async def test_mock_mcp_server_tool_registration(test_bridge: KnowledgeBridge) -> None:
    """Mock MCP server registers read tools by default, and management tools when enabled."""
    server_read = create_mcp_server(
        test_bridge,
        name="test-ragit",
        enable_management_tools=False,
        server_class=MockMCPServer,
    )
    tools = await server_read.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_knowledge" in tool_names
    assert "get_source" in tool_names
    assert "knowledge_status" in tool_names
    assert "sync_collection" not in tool_names
    assert "revoke_document" not in tool_names

    server_mgmt = create_mcp_server(
        test_bridge,
        name="test-ragit",
        enable_management_tools=True,
        server_class=MockMCPServer,
    )
    tools_mgmt = await server_mgmt.list_tools()
    tool_names_mgmt = [t.name for t in tools_mgmt]
    assert "sync_collection" in tool_names_mgmt
    assert "revoke_document" in tool_names_mgmt


async def test_mock_mcp_server_call_tools(test_bridge: KnowledgeBridge) -> None:
    """Mock MCP server executes search, get_source, and knowledge_status calls."""
    server = create_mcp_server(
        test_bridge,
        name="test-ragit",
        server_class=MockMCPServer,
    )

    # knowledge_status
    status = await server.call_tool("knowledge_status", {"collection": "default"})
    assert status["is_healthy"] is True

    # get_source
    span = await server.call_tool(
        "get_source",
        {
            "source_ref": "mcp_source",
            "document_id": "doc_mcp_1",
            "revision": "v1",
            "start_offset": 0,
            "end_offset": 20,
        },
    )
    assert span["status"] == "success"
    assert "Chapter 1: MCP" in span["text"]

    # search_knowledge
    bundle = await server.call_tool(
        "search_knowledge",
        {"query": "MCP Guide", "candidate_k": 3},
    )
    assert bundle["status"] in ("success", "empty_search")


@pytest.mark.skipif(_MCPServerClass is None, reason="Requires 'mcp' extra")
async def test_real_mcp_server_registers_read_tools_by_default(
    test_bridge: KnowledgeBridge,
) -> None:
    """Official MCP server by default only exposes read tools, not management tools."""
    server = create_mcp_server(test_bridge, name="test-ragit", enable_management_tools=False)
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    assert "search_knowledge" in tool_names
    assert "get_source" in tool_names
    assert "knowledge_status" in tool_names
    assert "sync_collection" not in tool_names
    assert "revoke_document" not in tool_names


@pytest.mark.skipif(_MCPServerClass is None, reason="Requires 'mcp' extra")
async def test_real_mcp_server_registers_management_tools_when_enabled(
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


@pytest.mark.skipif(_MCPServerClass is None, reason="Requires 'mcp' extra")
async def test_real_mcp_call_knowledge_status(test_bridge: KnowledgeBridge) -> None:
    """Calling knowledge_status through real MCP returns sanitized status."""
    server = create_mcp_server(test_bridge, name="test-ragit")
    res = await server.call_tool("knowledge_status", {"collection": "default"})
    status_data = _tool_payload(res)
    assert status_data["collection"] == "default"
    assert status_data["is_healthy"] is True


@pytest.mark.skipif(_MCPServerClass is None, reason="Requires 'mcp' extra")
async def test_real_mcp_call_get_source(test_bridge: KnowledgeBridge) -> None:
    """Calling get_source through real MCP returns authorized span."""
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
    span_data = _tool_payload(res)
    assert span_data["status"] == "success"
    assert "Chapter 1: MCP" in span_data["text"]


@pytest.mark.skipif(_MCPServerClass is None, reason="Requires 'mcp' extra")
async def test_real_mcp_call_search_knowledge(test_bridge: KnowledgeBridge) -> None:
    """Calling search_knowledge through real MCP returns structured EvidenceBundle."""
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
    bundle_data = _tool_payload(res)
    assert bundle_data["status"] == "success"
    assert len(bundle_data["items"]) >= 1
    assert "MCP Server Adapter" in bundle_data["items"][0]["text"]
