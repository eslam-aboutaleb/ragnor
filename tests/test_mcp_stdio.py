"""Unit tests for MCP stdio transport runner and JSON-RPC protocol loop."""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragit.mcp.stdio import _run_builtin_stdio_loop, run_stdio_server
from ragit.tools.bridge import KnowledgeBridge


@pytest.fixture
def mock_bridge() -> MagicMock:
    bridge = MagicMock(spec=KnowledgeBridge)
    bridge.default_collection = "default"
    mock_bundle = MagicMock()
    mock_bundle.to_dict.return_value = {
        "items": [
            {
                "text": "sample text",
                "source_id": "src-1",
                "citation_label": "[1]",
                "score": 0.9,
            }
        ],
        "status": "ok",
    }
    bridge.search_knowledge = AsyncMock(return_value=mock_bundle)

    mock_span = MagicMock()
    mock_span.to_dict.return_value = {
        "source_id": "src-1",
        "document_id": "doc-1",
        "revision": "v1",
        "text": "full source content",
    }
    bridge.get_source = AsyncMock(return_value=mock_span)

    bridge.knowledge_status = AsyncMock(
        return_value={
            "collection": "default",
            "status": "ready",
            "document_count": 10,
        }
    )
    return bridge


@pytest.mark.asyncio
async def test_run_stdio_server_with_mcp_sdk(mock_bridge: MagicMock) -> None:
    mock_server = MagicMock()
    mock_server.run_stdio_async = AsyncMock()

    with (
        patch("ragit.mcp.stdio._MCPServerClass", object()),
        patch("ragit.mcp.stdio.create_mcp_server", return_value=mock_server),
    ):
        await run_stdio_server(mock_bridge, name="test-server")
        mock_server.run_stdio_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_stdio_server_fallback_to_builtin(mock_bridge: MagicMock) -> None:
    with (
        patch("ragit.mcp.stdio._MCPServerClass", None),
        patch("ragit.mcp.stdio._run_builtin_stdio_loop", new_callable=AsyncMock) as mock_loop,
    ):
        await run_stdio_server(mock_bridge, name="test-server", enable_management_tools=True)
        mock_loop.assert_awaited_once_with(
            mock_bridge,
            name="test-server",
            enable_management=True,
            client_id=None,
            client_id_provider=None,
        )


@pytest.mark.asyncio
async def test_builtin_stdio_loop_methods(mock_bridge: MagicMock) -> None:
    requests = [
        # Empty line
        "",
        # Initialize
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        # Tools list
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        # Search knowledge
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search_knowledge", "arguments": {"query": "sample"}},
            }
        ),
        # Get source
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "get_source", "arguments": {"source_ref": "src-1"}},
            }
        ),
        # Knowledge status
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "knowledge_status", "arguments": {"collection": "default"}},
            }
        ),
        # Unknown tool
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "unknown_tool", "arguments": {}},
            }
        ),
        # Unknown method
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "unsupported_method"}),
        # Malformed JSON
        "{not valid json",
    ]

    input_data = ("\n".join(requests) + "\n").encode("utf-8")

    reader = asyncio.StreamReader()
    reader.feed_data(input_data)
    reader.feed_eof()

    stdout_capture = io.StringIO()

    with (
        patch("asyncio.StreamReader", return_value=reader),
        patch("asyncio.get_running_loop") as mock_get_loop,
        patch("sys.stdout.write", side_effect=stdout_capture.write),
        patch("sys.stdout.flush"),
    ):
        mock_loop = MagicMock()
        mock_loop.connect_read_pipe = AsyncMock()
        mock_get_loop.return_value = mock_loop

        await _run_builtin_stdio_loop(mock_bridge, name="test-server", enable_management=False)

    output = stdout_capture.getvalue()
    lines = [json.loads(line) for line in output.strip().split("\n") if line.strip()]

    # Verify initialize response
    init_res = next(entry for entry in lines if entry.get("id") == 1)
    assert init_res["result"]["serverInfo"]["name"] == "test-server"

    # Verify tools/list response
    tools_res = next(entry for entry in lines if entry.get("id") == 2)
    tool_names = [t["name"] for t in tools_res["result"]["tools"]]
    assert "search_knowledge" in tool_names
    assert "get_source" in tool_names
    assert "knowledge_status" in tool_names

    # Verify search_knowledge response
    search_res = next(entry for entry in lines if entry.get("id") == 3)
    assert search_res["result"]["isError"] is False

    # Verify get_source response
    get_res = next(entry for entry in lines if entry.get("id") == 4)
    assert get_res["result"]["isError"] is False

    # Verify status response
    status_res = next(entry for entry in lines if entry.get("id") == 5)
    assert status_res["result"]["isError"] is False

    # Verify unknown tool response error
    unknown_tool_res = next(entry for entry in lines if entry.get("id") == 6)
    assert unknown_tool_res["error"]["code"] == -32601

    # Verify unknown method response error
    unknown_method_res = next(entry for entry in lines if entry.get("id") == 7)
    assert unknown_method_res["error"]["code"] == -32601

    # Verify invalid JSON error response
    json_err_res = [entry for entry in lines if entry.get("id") is None]
    assert len(json_err_res) > 0
    assert json_err_res[0]["error"]["code"] == -32603
