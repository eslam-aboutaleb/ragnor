"""A ragnor retrieval strategy as a LlamaIndex ``BaseRetriever``."""

from __future__ import annotations

from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ragnor.core.validation import validate_identifier
from ragnor.integrations.llamaindex._async import run_sync
from ragnor.retrieval.strategies import RetrievalStrategy
from ragnor.retrieval.types import RetrievalRequest, RetrievedDocument


def _score(item: RetrievedDocument) -> float:
    """Best-first score: reranker, then RRF, then similarity from distance."""
    if item.reranker_score is not None:
        return float(item.reranker_score)
    if item.rrf_score > 0.0:
        return float(item.rrf_score)
    if item.distance is not None:
        return 1.0 / (1.0 + float(item.distance))
    return float(item.keyword_score or 0.0)


class RagnorRetriever(BaseRetriever):
    """Expose a ragnor strategy (dense, keyword, hybrid) to LlamaIndex.

    The isolation scope is mandatory and fixed at construction. Every query
    is pushed down with it, results are re-validated by ragnor
    (``strict=True``, so a cross-scope row raises ``ScopeViolationError``
    instead of being returned), and extra ``filters`` may not name the scope
    field. There is no per-query override.
    """

    def __init__(  # noqa: PLR0913
        self,
        strategy: RetrievalStrategy,
        *,
        isolation_scope: str,
        scope_field: str = "tenant_id",
        similarity_top_k: int = 5,
        threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> None:
        if not isolation_scope:
            raise ValueError("isolation_scope is required and must be non-empty")
        validate_identifier(scope_field, "scope_field")
        extra = dict(filters or {})
        if scope_field in extra:
            raise ValueError(f"filters may not set the isolation scope field {scope_field!r}")
        super().__init__()
        self._strategy = strategy
        self._scope = isolation_scope
        self._scope_field = scope_field
        self._top_k = similarity_top_k
        self._threshold = threshold
        self._filters = extra
        self._text_field = text_field
        self._metadata_fields = list(metadata_fields or [])

    def _request(self, query: str) -> RetrievalRequest:
        return RetrievalRequest(
            query=query,
            final_k=self._top_k,
            threshold=self._threshold,
            filters=dict(self._filters),
            isolation_scope=self._scope,
            scope_field=self._scope_field,
            text_field=self._text_field,
            metadata_fields=list(self._metadata_fields),
            strict=True,
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        rows = await self._strategy.retrieve(self._request(query_bundle.query_str))
        return [
            NodeWithScore(
                node=TextNode(
                    id_=str(item.id) if item.id is not None else None,
                    text=item.document,
                    metadata=dict(item.metadata),
                ),
                score=_score(item),
            )
            for item in (RetrievedDocument.from_dict(row) for row in rows)
        ]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return run_sync(self._aretrieve(query_bundle))


__all__ = ["RagnorRetriever"]
