"""Stdio transport runner for MCP server.

Provides local stdio communication for integration with MCP host agents
(e.g. Claude Desktop, IDE extensions, or custom CLI hosts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable

from ragit.mcp.server import _MCPServerClass, create_mcp_server
from ragit.tools.bridge import KnowledgeBridge

logger = logging.getLogger(__name__)


async def run_stdio_server(
    bridge: KnowledgeBridge,
    *,
    name: str = "ragit-knowledge",
    enable_management_tools: bool = False,
    client_id: str | None = None,
    client_id_provider: Callable[[], str | None] | None = None,
) -> None:
    """Run MCP server over standard input and standard output.

    Uses the official MCP SDK if available, or falls back to a built-in
    zero-dependency JSON-RPC 2.0 stdio server for read tools.

    Args:
        bridge: The KnowledgeBridge providing retrieval and source verification.
        name: MCP server name.
        enable_management_tools: Whether to enable management tools.
    """
    if _MCPServerClass is not None:
        server = create_mcp_server(
            bridge=bridge,
            name=name,
            enable_management_tools=enable_management_tools,
            client_id=client_id,
            client_id_provider=client_id_provider,
        )
        if hasattr(server, "run_stdio_async"):
            await server.run_stdio_async()
            return

    # Lightweight built-in JSON-RPC stdio loop (zero external dependencies)
    await _run_builtin_stdio_loop(
        bridge,
        name=name,
        enable_management=enable_management_tools,
        client_id=client_id,
        client_id_provider=client_id_provider,
    )


async def _run_builtin_stdio_loop(
    bridge: KnowledgeBridge,
    name: str,
    enable_management: bool,
    client_id: str | None = None,
    client_id_provider: Callable[[], str | None] | None = None,
) -> None:
    """Zero-dependency JSON-RPC 2.0 stdio protocol handler."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    tool_definitions = [
        {
            "name": "search_knowledge",
            "description": "Execute authorized hybrid search across indexed knowledge collections.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string", "default": bridge.default_collection},
                    "filters": {"type": "object"},
                    "candidate_k": {"type": "integer", "default": 20},
                    "final_k": {"type": "integer", "default": 5},
                    "budget_limit": {"type": "integer"},
                    "scope": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_source",
            "description": "Fetch authorized, bounded text span and revision metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_ref": {"type": "string"},
                    "document_id": {"type": "string"},
                    "revision": {"type": "string", "default": "v1"},
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "collection": {"type": "string", "default": bridge.default_collection},
                    "scope": {"type": "string"},
                },
                "required": ["source_ref"],
            },
        },
        {
            "name": "knowledge_status",
            "description": "Inspect sanitized health, document count, and freshness summary.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "default": bridge.default_collection},
                    "scope": {"type": "string"},
                },
            },
        },
    ]

    while not reader.at_eof():
        line = await reader.readline()
        if not line:
            break

        line_str = line.decode("utf-8").strip()
        if not line_str:
            continue

        req_id = None
        try:
            request = json.loads(line_str)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": name, "version": "0.4.0"},
                    },
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": tool_definitions},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                # Override any untrusted identity in raw JSON-RPC arguments.
                arguments = {
                    **arguments,
                    "client_id": (
                        client_id_provider() if client_id_provider is not None else client_id
                    ),
                }

                if tool_name == "search_knowledge":
                    res = await bridge.search_knowledge(**arguments)
                    out = res.to_dict()
                elif tool_name == "get_source":
                    res = await bridge.get_source(**arguments)
                    out = res.to_dict()
                elif tool_name == "knowledge_status":
                    out = await bridge.knowledge_status(**arguments)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method {tool_name!r} not found"},
                    }
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                    continue

                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(out, indent=2)}],
                        "isError": False,
                    },
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method {method!r}"},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            err_response = {
                "jsonrpc": "2.0",
                "id": req_id if "req_id" in locals() else None,
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
            }
            sys.stdout.write(json.dumps(err_response) + "\n")
            sys.stdout.flush()
