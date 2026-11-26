"""A ragnor ``VectorStore`` as a LlamaIndex vector store."""

from __future__ import annotations

import math
from typing import Any

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict
from pydantic import PrivateAttr

from ragnor.core.validation import validate_identifier
from ragnor.integrations.llamaindex._async import run_sync
from ragnor.retrieval.types import ScopeViolationError

_NODE_CONTENT = "node_content"
_NODE_TYPE = "node_type"
_REF_DOC = "ref_doc_id"
_RESERVED = {"id", "text", "embedding", _NODE_CONTENT, _NODE_TYPE, _REF_DOC}
_SCALARS = (str, int, float, bool)


class RagnorVectorStore(BasePydanticVectorStore):
    """Use a ragnor store (memory, qdrant, pgvector...) from LlamaIndex.

    Tenant/owner isolation lives here, not in the caller:

    * ``isolation_scope`` is required at construction.
    * Every node written is stamped with the scope; a node whose own
      metadata sets the scope field to a different value is rejected.
    * Every query, get, delete and clear is restricted to the scope.
      Caller filters naming the scope field are rejected.
    * Query results are re-checked; a row from another scope raises
      :class:`~ragnor.retrieval.types.ScopeViolationError`.

    Supported filters: ``EQ`` and ``IN`` combined with ``AND``. Anything else
    raises ``NotImplementedError`` rather than being silently dropped.
    Modes: ``DEFAULT`` (dense), ``HYBRID`` (dense + keyword, RRF) and
    ``TEXT_SEARCH`` (keyword). Nodes keep their full LlamaIndex content, so
    any node type round-trips. Only scalar metadata becomes a filterable field
    (keys must be safe identifiers).
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    _store: Any = PrivateAttr()
    _scope: str = PrivateAttr()
    _scope_field: str = PrivateAttr()
    _threshold: float = PrivateAttr()

    def __init__(
        self,
        store: Any,
        *,
        isolation_scope: str,
        scope_field: str = "tenant_id",
        distance_threshold: float | None = None,
    ) -> None:
        if not isolation_scope:
            raise ValueError("isolation_scope is required and must be non-empty")
        validate_identifier(scope_field, "scope_field")
        super().__init__()
        self._store = store
        self._scope = isolation_scope
        self._scope_field = scope_field
        self._threshold = math.inf if distance_threshold is None else distance_threshold

    @property
    def client(self) -> Any:
        """The wrapped ragnor store."""
        return self._store

    # -- helpers -----------------------------------------------------------

    def _metadata_fields(self) -> list[str]:
        return [_NODE_CONTENT, _NODE_TYPE, _REF_DOC, self._scope_field]

    def _filters(self, filters: MetadataFilters | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if filters is not None:
            if filters.condition not in (FilterCondition.AND, None):
                raise NotImplementedError("only AND metadata filters are supported")
            for item in filters.filters:
                if not isinstance(item, MetadataFilter):
                    raise NotImplementedError("nested metadata filters are not supported")
                if item.key == self._scope_field:
                    raise ValueError(
                        f"filters may not set the isolation scope field {self._scope_field!r}"
                    )
                validate_identifier(item.key, "filter key")
                if item.operator == FilterOperator.EQ:
                    out[item.key] = item.value
                elif item.operator == FilterOperator.IN:
                    out[item.key] = list(item.value)  # type: ignore[arg-type]
                else:
                    raise NotImplementedError(
                        f"filter operator {item.operator.name} is not supported"
                    )
        out[self._scope_field] = self._scope
        return out

    def _document(self, node: BaseNode) -> dict[str, Any]:
        embedding = node.get_embedding()
        meta = node_to_metadata_dict(node, remove_text=False, flat_metadata=False)
        flat: dict[str, Any] = {}
        for key, value in (node.metadata or {}).items():
            if key == self._scope_field:
                if value != self._scope:
                    raise ValueError(
                        f"node {node.node_id!r} sets {key!r} to a different scope than the store"
                    )
                continue
            if key in _RESERVED or not isinstance(value, _SCALARS):
                continue
            try:
                validate_identifier(key, "metadata key")
            except ValueError:
                continue
            flat[key] = value
        flat[_NODE_CONTENT] = meta["_node_content"]
        flat[_NODE_TYPE] = meta["_node_type"]
        flat[_REF_DOC] = node.ref_doc_id or ""
        return {
            "id": node.node_id,
            "text": node.get_content(),
            "metadata": flat,
            "embedding": embedding,
        }

    def _node(self, row: dict[str, Any]) -> BaseNode:
        meta = row.get("metadata") or {}
        found = meta.get(self._scope_field)
        if found != self._scope:
            raise ScopeViolationError(
                f"row {row.get('id')!r} has scope {found!r}, expected {self._scope!r}"
            )
        return metadata_dict_to_node(
            {"_node_content": meta[_NODE_CONTENT], "_node_type": meta[_NODE_TYPE]}
        )

    # -- add / delete ------------------------------------------------------

    async def async_add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        """Upsert nodes (which must already carry embeddings)."""
        docs = [self._document(node) for node in nodes]
        for doc in docs:
            if doc["embedding"] is None:
                raise ValueError(f"node {doc['id']!r} has no embedding")
        if docs:
            await self._store.upsert(docs, extra_fields={self._scope_field: self._scope})
        return [doc["id"] for doc in docs]

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        return run_sync(self.async_add(nodes, **add_kwargs))

    def _require(self, name: str) -> Any:
        method = getattr(self._store, name, None)
        if method is None:
            raise NotImplementedError(f"the wrapped store does not implement {name}()")
        return method

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node of one source document, within the scope."""
        scoped = {_REF_DOC: ref_doc_id, self._scope_field: self._scope}
        await self._require("delete")(filters=scoped)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        run_sync(self.adelete(ref_doc_id, **delete_kwargs))

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete by node ids and/or filters, within the scope."""
        if node_ids is None and filters is None:
            raise ValueError("pass node_ids or filters; use clear() to remove the whole scope")
        if node_ids is not None and not node_ids:
            return
        await self._require("delete")(ids=node_ids, filters=self._filters(filters))

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        run_sync(self.adelete_nodes(node_ids, filters, **delete_kwargs))

    async def aclear(self, **clear_kwargs: Any) -> None:
        """Delete everything in this store's scope (never other scopes)."""
        await self._require("delete")(filters={self._scope_field: self._scope})

    def clear(self) -> None:
        run_sync(self.aclear())

    # -- read --------------------------------------------------------------

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """Fetch nodes by id and/or filters, within the scope."""
        if node_ids is not None and not node_ids:
            return []
        rows = await self._require("get")(
            ids=node_ids,
            filters=self._filters(filters),
            metadata_fields=self._metadata_fields(),
        )
        return [self._node(row) for row in rows]

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        return run_sync(self.aget_nodes(node_ids, filters))

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Run a dense, hybrid or keyword query inside the scope."""
        filters = self._filters(query.filters)
        if query.node_ids or query.doc_ids:
            raise NotImplementedError("node_ids/doc_ids query restriction is not supported")
        top_k = query.similarity_top_k
        fields = self._metadata_fields()
        mode = query.mode
        if mode == VectorStoreQueryMode.DEFAULT:
            if query.query_embedding is None:
                raise ValueError("DEFAULT mode needs query_embedding")
            rows = await self._require("vector_search")(
                embedding=query.query_embedding,
                n_results=top_k,
                threshold=self._threshold,
                metadata_fields=fields,
                **filters,
            )
            sims = [1.0 / (1.0 + float(r["distance"])) for r in rows]
        elif mode == VectorStoreQueryMode.HYBRID:
            if query.query_embedding is None or not query.query_str:
                raise ValueError("HYBRID mode needs query_embedding and query_str")
            rows = await self._store.hybrid_search(
                query=query.query_str,
                embedding=query.query_embedding,
                n_results=top_k,
                threshold=self._threshold,
                metadata_fields=fields,
                strict=True,
                **filters,
            )
            sims = [float(r.get("_rrf_score", 0.0)) for r in rows]
        elif mode == VectorStoreQueryMode.TEXT_SEARCH:
            if not query.query_str:
                raise ValueError("TEXT_SEARCH mode needs query_str")
            rows = await self._require("keyword_search")(
                query=query.query_str,
                n_results=top_k,
                metadata_fields=fields,
                **filters,
            )
            sims = [float(r.get("keyword_score") or 0.0) for r in rows]
        else:
            raise NotImplementedError(f"query mode {mode.name} is not supported")
        nodes = [self._node(row) for row in rows]
        return VectorStoreQueryResult(
            nodes=nodes, similarities=sims, ids=[n.node_id for n in nodes]
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return run_sync(self.aquery(query, **kwargs))


__all__ = ["RagnorVectorStore"]
