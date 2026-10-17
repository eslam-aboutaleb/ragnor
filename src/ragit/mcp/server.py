"""MCP (Model Context Protocol) server adapter for ragit.

Exposes KnowledgeBridge retrieval, source verification, and status tools
over MCP stdio or remote transports.
"""

from __future__ import annotations

import logging
from typing import Any

from ragit.tools.bridge import KnowledgeBridge

logger = logging.getLogger(__name__)

# Detect available MCP SDK versions (mcp 2.x MCPServer vs mcp 1.x FastMCP)
_MCPServerClass: Any = None
try:
    from mcp.server.mcpserver import MCPServer

    _MCPServerClass = MCPServer
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

        _MCPServerClass = FastMCP
    except (ImportError, ModuleNotFoundError):
        _MCPServerClass = None


def create_mcp_server(
    bridge: KnowledgeBridge,
    name: str = "ragit-knowledge",
    *,
    enable_management_tools: bool = False,
    server_class: Any | None = None,
) -> Any:
    """Create and configure an MCP server instance wrapping a KnowledgeBridge.

    By default, only read tools (search_knowledge, get_source, knowledge_status)
    are registered. Ingestion/sync and revocation are separate opt-in management
    tools enabled only when :paramref:`enable_management_tools` is True and the
    bridge's allow_management flag is set.

    Args:
        bridge: The configured :class:`KnowledgeBridge`.
        name: MCP server name reported during initialization.
        enable_management_tools: If True, registers sync and deletion tools.
        server_class: Optional override class for MCPServer / FastMCP.

    Returns:
        Configured MCPServer or FastMCP instance.

    Raises:
        ImportError: If the official 'mcp' SDK is not installed and no server_class is given.
    """
    cls = server_class or _MCPServerClass
    if cls is None:
        raise ImportError(
            "The MCP server adapter requires the 'mcp' package. "
            "Install it with: pip install 'ragit-toolkit[mcp]' or pip install 'mcp>=1.0.0'"
        )

    server = cls(name)

    # 1. search_knowledge tool
    @server.tool(
        name="search_knowledge",
        description=(
            "Execute authorized hybrid search across indexed knowledge collections. "
            "Returns a structured EvidenceBundle containing deduplicated evidence items, "
            "stable source/revision citations, and token budget accounting."
        ),
    )
    async def search_knowledge(  # noqa: PLR0913, PLR0917
        query: str,
        collection: str = "default",
        filters: dict[str, Any] | None = None,
        candidate_k: int = 20,
        final_k: int = 5,
        budget_limit: int | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Search knowledge collection with budget limits and tenant isolation."""
        bundle = await bridge.search_knowledge(
            query=query,
            collection=collection,
            filters=filters,
            candidate_k=candidate_k,
            final_k=final_k,
            budget_limit=budget_limit,
            scope=scope,
        )
        return bundle.to_dict()

    # 2. get_source tool
    @server.tool(
        name="get_source",
        description=(
            "Fetch an authorized, bounded text span and revision metadata from an immutable "
            "document snapshot. Prevents substituting newer content for old citations."
        ),
    )
    async def get_source(  # noqa: PLR0913, PLR0917
        source_ref: str,
        document_id: str | None = None,
        revision: str = "v1",
        start_offset: int | None = None,
        end_offset: int | None = None,
        max_span_length: int | None = None,
        collection: str = "default",
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Fetch bounded source span for citation verification."""
        span_result = await bridge.get_source(
            source_ref=source_ref,
            document_id=document_id,
            revision=revision,
            start_offset=start_offset,
            end_offset=end_offset,
            max_span_length=max_span_length,
            collection=collection,
            scope=scope,
        )
        return span_result.to_dict()

    # 3. knowledge_status tool
    @server.tool(
        name="knowledge_status",
        description=(
            "Inspect sanitized health, document count, and freshness summary for an "
            "authorized knowledge collection. Does not expose internal connection credentials."
        ),
    )
    async def knowledge_status(
        collection: str = "default",
        scope: str | None = None,
        include_expensive: bool = False,
    ) -> dict[str, Any]:
        """Retrieve sanitized status summary for a collection."""
        return await bridge.knowledge_status(
            collection=collection,
            scope=scope,
            include_expensive=include_expensive,
        )

    # 4. Opt-in management tools
    if enable_management_tools:
        if not bridge.allow_management:
            logger.warning(
                "enable_management_tools=True was requested on MCP server, but "
                "bridge.allow_management is False. Management calls will be rejected."
            )

        @server.tool(
            name="sync_collection",
            description=(
                "Ingest and sync document batches into a knowledge collection. "
                "Guarded by host management permissions."
            ),
        )
        async def sync_collection(
            collection: str,
            source_id: str,
            documents: list[dict[str, Any]],
            scope: str | None = None,
        ) -> dict[str, Any]:
            """Ingest document batches into collection."""
            return await bridge.sync_collection(
                collection=collection,
                source_id=source_id,
                documents=documents,
                scope=scope,
            )

        @server.tool(
            name="revoke_document",
            description=(
                "Authoritatively remove/revoke a document and its active chunks from search. "
                "Guarded by host management permissions."
            ),
        )
        async def revoke_document(
            collection: str,
            document_id: str,
            source_id: str = "default_source",
            scope: str | None = None,
        ) -> dict[str, Any]:
            """Revoke document from active search visibility."""
            return await bridge.revoke_document(
                collection=collection,
                document_id=document_id,
                source_id=source_id,
                scope=scope,
            )

    return server
