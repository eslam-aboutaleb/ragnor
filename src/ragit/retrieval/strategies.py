"""Retrieval strategies over storage capability interfaces.

Strategies compose capability Protocols
(:class:`ragit.stores.base.VectorIndex` and
:class:`ragit.stores.base.LexicalIndex`) instead of
the fused :class:`ragit.stores.base.VectorStore`
contract, so retrieval topologies the fused contract
cannot express become possible:

* :class:`DenseRetrieval` -- vector search only;
* :class:`KeywordRetrieval` -- keyword search only,
  with **no embedding dependency**;
* :class:`HybridRetrieval` -- both stages, fused
  with Reciprocal Rank Fusion, optionally reranked.

Enhancements in this release:
* Typed :class:`~ragit.retrieval.types.RetrievalRequest` and
  :class:`~ragit.retrieval.types.RetrievalResponse` wrappers with
  full bidirectional bridges to the legacy dict API.
* Separate ``candidate_k`` from ``final_k`` so an optional
  reranker stage has enough candidate documents before final slicing.
* Backend capability declarations for filter operators, lexical search,
  document fetch, generation visibility, deletion, and transaction
  guarantees. Unsupported mandatory constraints fail closed.
* Push filters directly into each dense and lexical search stage.
* Scope validation before evidence assembly as defense in depth.
* Optional host-injected :class:`~ragit.retrieval.query_transform.QueryTransformer`
  with hard limits and timeouts.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ragit.retrieval.capabilities import (
    CapabilityUnsupportedError,
    get_backend_capabilities,
)
from ragit.retrieval.query_transform import (
    BoundedQueryTransformer,
    QueryTransformer,
)
from ragit.retrieval.types import (
    RetrievalRequest,
    RetrievalResponse,
    ScopeViolationError,
)
from ragit.stores._scoring import fuse_hybrid
from ragit.stores.base import EmbeddingFunction, LexicalIndex, VectorIndex
from ragit.types import RetrievalError

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Structural interface for a retrieval strategy."""

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents matching ``query``."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Structural interface for an optional reranking stage.

    Hosts inject their own reranker (e.g. LiteLLM, cross-encoder)
    into :class:`HybridRetrieval`.
    """

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank ``results`` for ``query``."""
        ...


def _coerce_request(
    query_or_request: str | RetrievalRequest,
    n_results: int,
    kwargs: dict[str, Any],
) -> tuple[RetrievalRequest, dict[str, Any]]:
    """Convert arguments to a canonical :class:`RetrievalRequest`."""
    clean_kwargs = dict(kwargs)
    if isinstance(query_or_request, RetrievalRequest):
        req = query_or_request
        return req, clean_kwargs

    candidate_k = clean_kwargs.pop("candidate_k", None)
    threshold = clean_kwargs.pop("threshold", None)
    strict = clean_kwargs.pop("strict", False)
    isolation_scope = clean_kwargs.pop("isolation_scope", None)
    scope_field = clean_kwargs.pop("scope_field", "tenant_id")
    mandatory_capabilities = clean_kwargs.pop("mandatory_capabilities", None)
    expected_generation = clean_kwargs.pop("expected_generation", None)
    text_field = clean_kwargs.pop("text_field", "text")
    metadata_fields = clean_kwargs.pop("metadata_fields", None)
    session = clean_kwargs.pop("session", None)

    req = RetrievalRequest(
        query=query_or_request,
        final_k=n_results,
        candidate_k=candidate_k,
        threshold=threshold,
        filters=clean_kwargs,
        mandatory_capabilities=set(mandatory_capabilities or ()),
        isolation_scope=isolation_scope,
        scope_field=scope_field,
        expected_generation=expected_generation,
        text_field=text_field,
        metadata_fields=metadata_fields or [],
        session=session,
        strict=strict,
    )
    return req, clean_kwargs


def _validate_scope_defense_in_depth(
    results: list[dict[str, Any]],
    req: RetrievalRequest,
) -> list[dict[str, Any]]:
    """Validate that every result belongs to the requested isolation scope."""
    if req.isolation_scope is None:
        return results

    validated: list[dict[str, Any]] = []
    for row in results:
        meta = row.get("metadata") or {}
        row_scope = meta.get(req.scope_field) or row.get(req.scope_field)
        if row_scope is not None and row_scope != req.isolation_scope:
            msg = (
                f"Scope violation: result {row.get('id')!r} belongs to scope "
                f"{row_scope!r}, expected {req.isolation_scope!r}."
            )
            logger.warning(msg)
            if req.strict:
                raise ScopeViolationError(msg)
            continue
        validated.append(row)
    return validated


class DenseRetrieval:
    """Dense-only retrieval over a :class:`VectorIndex`.

    Embeds the query with the injected embedder and
    calls the index's ``vector_search``.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: EmbeddingFunction,
        *,
        query_transformer: QueryTransformer | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._embedder = embedder
        self._query_transformer = (
            BoundedQueryTransformer(query_transformer) if query_transformer is not None else None
        )

    async def retrieve(
        self,
        query: str | RetrievalRequest,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by vector similarity."""
        req, remaining_kwargs = _coerce_request(query, n_results, kwargs)
        if req.threshold is None:
            raise ValueError("dense retrieval requires a distance threshold; pass threshold=...")

        # Validate mandatory constraints against backend capabilities
        caps = get_backend_capabilities(self._vector_index)
        caps.validate_constraints(req.mandatory_capabilities)
        effective_filters = req.effective_filters()
        caps.validate_filters(effective_filters)

        try:
            queries = [req.query]
            if self._query_transformer is not None:
                queries = await self._query_transformer.transform(req.query)

            all_results: list[dict[str, Any]] = []
            candidate_k = req.get_candidate_k()

            for q in queries:
                embeddings = await self._embedder([q])
                results = await self._vector_index.vector_search(
                    embedding=embeddings[0],
                    n_results=candidate_k,
                    threshold=req.threshold,
                    text_field=req.text_field,
                    metadata_fields=req.metadata_fields,
                    session=req.session,
                    **effective_filters,
                )
                all_results.extend(results)

            # Deduplicate by id if multi-query was used
            seen_ids: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for r in all_results:
                rid = str(r.get("id"))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    deduped.append(r)

            validated = _validate_scope_defense_in_depth(deduped, req)
            return validated[: req.final_k]
        except CapabilityUnsupportedError:
            raise
        except ScopeViolationError:
            raise
        except Exception as exc:
            logger.error("Dense retrieval failed: %s", exc)
            if req.strict:
                raise RetrievalError(f"Dense retrieval failed: {exc}") from exc
            return []

    async def retrieve_typed(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute retrieval and return a typed :class:`RetrievalResponse`."""
        rows = await self.retrieve(request)
        return RetrievalResponse.from_dicts(
            rows,
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )


class KeywordRetrieval:
    """Keyword-only retrieval over a :class:`LexicalIndex`.

    Calls the index's ``keyword_search`` with no embedding dependency.
    """

    def __init__(
        self,
        lexical_index: LexicalIndex,
        *,
        query_transformer: QueryTransformer | None = None,
    ) -> None:
        self._lexical_index = lexical_index
        self._query_transformer = (
            BoundedQueryTransformer(query_transformer) if query_transformer is not None else None
        )

    async def retrieve(
        self,
        query: str | RetrievalRequest,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by keyword search."""
        req, remaining_kwargs = _coerce_request(query, n_results, kwargs)

        caps = get_backend_capabilities(self._lexical_index)
        caps.validate_constraints(req.mandatory_capabilities)
        effective_filters = req.effective_filters()
        caps.validate_filters(effective_filters)

        try:
            queries = [req.query]
            if self._query_transformer is not None:
                queries = await self._query_transformer.transform(req.query)

            all_results: list[dict[str, Any]] = []
            candidate_k = req.get_candidate_k()

            for q in queries:
                results = await self._lexical_index.keyword_search(
                    query=q,
                    n_results=candidate_k,
                    text_field=req.text_field,
                    metadata_fields=req.metadata_fields,
                    session=req.session,
                    **effective_filters,
                )
                all_results.extend(results)

            seen_ids: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for r in all_results:
                rid = str(r.get("id"))
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    deduped.append(r)

            validated = _validate_scope_defense_in_depth(deduped, req)
            return validated[: req.final_k]
        except CapabilityUnsupportedError:
            raise
        except ScopeViolationError:
            raise
        except Exception as exc:
            logger.error("Keyword retrieval failed: %s", exc)
            if req.strict:
                raise RetrievalError(f"Keyword retrieval failed: {exc}") from exc
            return []

    async def retrieve_typed(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute retrieval and return a typed :class:`RetrievalResponse`."""
        rows = await self.retrieve(request)
        return RetrievalResponse.from_dicts(
            rows,
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )


class HybridRetrieval:
    """Hybrid retrieval over two capability stages, RRF-fused and optionally reranked.

    Supports candidate_k separately from final_k, filter pushdown,
    backend capability verification, scope validation, and optional query transformation.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        embedder: EmbeddingFunction,
        reranker: Reranker | None = None,
        *,
        query_transformer: QueryTransformer | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._lexical_index = lexical_index
        self._embedder = embedder
        self._reranker = reranker
        self._query_transformer = (
            BoundedQueryTransformer(query_transformer) if query_transformer is not None else None
        )

    async def retrieve(
        self,
        query: str | RetrievalRequest,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by hybrid vector + keyword search."""
        req, remaining_kwargs = _coerce_request(query, n_results, kwargs)
        if req.threshold is None:
            raise ValueError("hybrid retrieval requires a distance threshold; pass threshold=...")

        # Capability validation on both storage backends
        v_caps = get_backend_capabilities(self._vector_index)
        l_caps = get_backend_capabilities(self._lexical_index)
        v_caps.validate_constraints(req.mandatory_capabilities)
        l_caps.validate_constraints(req.mandatory_capabilities)

        effective_filters = req.effective_filters()
        v_caps.validate_filters(effective_filters)
        l_caps.validate_filters(effective_filters)

        try:
            candidate_k = req.get_candidate_k()
            queries = [req.query]
            if self._query_transformer is not None:
                queries = await self._query_transformer.transform(req.query)

            all_fused: list[dict[str, Any]] = []

            for q in queries:
                embeddings = await self._embedder([q])
                vector_results = await self._vector_index.vector_search(
                    embedding=embeddings[0],
                    n_results=candidate_k,
                    threshold=req.threshold,
                    text_field=req.text_field,
                    metadata_fields=req.metadata_fields,
                    session=req.session,
                    **effective_filters,
                )
                keyword_results = await self._lexical_index.keyword_search(
                    query=q,
                    n_results=candidate_k,
                    text_field=req.text_field,
                    metadata_fields=req.metadata_fields,
                    session=req.session,
                    **effective_filters,
                )
                fused = fuse_hybrid(vector_results, keyword_results, candidate_k)
                all_fused.extend(fused)

            # Deduplicate by id if multi-query generated multiple lists
            seen_ids: set[str] = set()
            candidates: list[dict[str, Any]] = []
            for item in all_fused:
                item_id = str(item.get("id"))
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    candidates.append(item)

            # Rerank candidate list if a reranker is provided
            if self._reranker is not None and candidates:
                candidates = await self._reranker.rerank(req.query, candidates)

            # Scope defense-in-depth validation
            validated = _validate_scope_defense_in_depth(candidates, req)

            # Return sliced to final_k
            return validated[: req.final_k]

        except CapabilityUnsupportedError:
            raise
        except ScopeViolationError:
            raise
        except Exception as exc:
            logger.error("Hybrid retrieval failed: %s", exc)
            if req.strict:
                raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc
            return []

    async def retrieve_typed(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute retrieval and return a typed :class:`RetrievalResponse`."""
        rows = await self.retrieve(request)
        return RetrievalResponse.from_dicts(
            rows,
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )
