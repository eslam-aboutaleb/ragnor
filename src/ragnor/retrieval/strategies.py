"""Retrieval strategies over storage capability interfaces.

Strategies compose capability Protocols
(:class:`ragnor.stores.base.VectorIndex` and
:class:`ragnor.stores.base.LexicalIndex`) instead of
the fused :class:`ragnor.stores.base.VectorStore`
contract, so retrieval topologies the fused contract
cannot express become possible:

* :class:`DenseRetrieval` -- vector search only;
* :class:`KeywordRetrieval` -- keyword search only,
  with **no embedding dependency**;
* :class:`HybridRetrieval` -- both stages, fused
  with Reciprocal Rank Fusion, optionally reranked.

Enhancements in this release:
* Typed :class:`~ragnor.retrieval.types.RetrievalRequest` and
  :class:`~ragnor.retrieval.types.RetrievalResponse` wrappers with
  full bidirectional bridges to the legacy dict API.
* Separate ``candidate_k`` from ``final_k`` so an optional
  reranker stage has enough candidate documents before final slicing.
* Backend capability declarations for filter operators, lexical search,
  document fetch, generation visibility, deletion, and transaction
  guarantees. Unsupported mandatory constraints fail closed.
* Push filters directly into each dense and lexical search stage.
* Scope validation before evidence assembly as defense in depth.
* Optional host-injected :class:`~ragnor.retrieval.query_transform.QueryTransformer`
  with hard limits and timeouts.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ragnor.core.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    OperationTimeoutError,
    ScopeDeniedError,
    UnsupportedCapabilityError,
)
from ragnor.retrieval.capabilities import (
    CapabilityUnsupportedError,
    get_backend_capabilities,
)
from ragnor.retrieval.query_transform import (
    BoundedQueryTransformer,
    QueryTransformer,
)
from ragnor.retrieval.types import (
    RetrievalRequest,
    RetrievalResponse,
    ScopeViolationError,
)
from ragnor.stores._scoring import fuse_hybrid
from ragnor.stores.base import EmbeddingFunction, LexicalIndex, VectorIndex
from ragnor.core.types import RetrievalError

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
    validated: list[dict[str, Any]] = []
    for row in results:
        meta = row.get("metadata") or {}
        row_scope = meta.get(req.scope_field, row.get(req.scope_field))
        scope_invalid = req.isolation_scope is not None and row_scope != req.isolation_scope
        generation_invalid = (
            req.expected_generation is not None
            and meta.get("generation") != req.expected_generation
        )
        if scope_invalid or generation_invalid:
            msg = (
                f"Scope violation: result {row.get('id')!r} belongs to scope "
                f"{row_scope!r}, expected {req.isolation_scope!r}; "
                f"generation {meta.get('generation')!r}, expected {req.expected_generation!r}."
            )
            logger.warning(msg)
            if req.strict:
                raise ScopeViolationError(msg)
            continue
        validated.append(row)
    return validated


def _required_metadata(req: RetrievalRequest) -> list[str]:
    fields = list(req.metadata_fields)
    if req.isolation_scope is not None:
        fields.append(req.scope_field)
    if req.expected_generation is not None:
        fields.append("generation")
    return list(dict.fromkeys(fields))


def _prepare_retrieval(backend: Any, req: RetrievalRequest) -> dict[str, Any]:
    caps = get_backend_capabilities(backend)
    caps.validate_constraints(req.mandatory_capabilities)
    effective_filters = req.effective_filters()
    caps.validate_filters(effective_filters)
    return effective_filters


def _dedupe_and_finalize(
    results: list[dict[str, Any]], req: RetrievalRequest
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        rid = str(r.get("id"))
        if rid not in seen_ids:
            seen_ids.add(rid)
            deduped.append(r)
    validated = _validate_scope_defense_in_depth(deduped, req)
    return validated[: req.final_k]


async def _query_vector(embedder: EmbeddingFunction, query: str) -> list[float]:
    if hasattr(embedder, "embed_query"):
        return await embedder.embed_query(query)
    return (await embedder([query]))[0]


async def _expand_queries(query: str, transformer: QueryTransformer | None) -> list[str]:
    if transformer is not None:
        return await transformer.transform(query)
    return [query]


class _BaseRetrievalStrategy:
    """Base class providing typed response wrapping across retrieval strategies."""

    async def retrieve(
        self,
        query: str | RetrievalRequest,
        n_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def retrieve_typed(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute retrieval and return a typed :class:`RetrievalResponse`."""
        rows = await self.retrieve(request)
        return RetrievalResponse.from_dicts(
            rows,
            query=request.query,
            candidate_k=request.get_candidate_k(),
            final_k=request.final_k,
        )


def _handle_retrieval_exception(exc: Exception, stage: str, strict: bool) -> list[dict[str, Any]]:
    if isinstance(
        exc,
        (
            CapabilityUnsupportedError,
            UnsupportedCapabilityError,
            ScopeViolationError,
            ScopeDeniedError,
            ConfigurationError,
        ),
    ):
        raise exc
    logger.error("%s retrieval failed: %s", stage, type(exc).__name__)
    if strict:
        if isinstance(exc, RetrievalError):
            raise exc
        if isinstance(exc, TimeoutError):
            raise OperationTimeoutError(
                f"{stage} retrieval timed out ({type(exc).__name__})"
            ) from exc
        raise BackendUnavailableError(f"{stage} retrieval failed ({type(exc).__name__})") from exc
    return []


class DenseRetrieval(_BaseRetrievalStrategy):
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

        effective_filters = _prepare_retrieval(self._vector_index, req)

        try:
            queries = await _expand_queries(req.query, self._query_transformer)
            all_results: list[dict[str, Any]] = []
            candidate_k = req.get_candidate_k()

            for q in queries:
                query_vector = await _query_vector(self._embedder, q)
                results = await self._vector_index.vector_search(
                    embedding=query_vector,
                    n_results=candidate_k,
                    threshold=req.threshold,
                    text_field=req.text_field,
                    metadata_fields=_required_metadata(req),
                    session=req.session,
                    **effective_filters,
                )
                all_results.extend(results)

            return _dedupe_and_finalize(all_results, req)
        except Exception as exc:
            return _handle_retrieval_exception(exc, "Dense", req.strict)


class KeywordRetrieval(_BaseRetrievalStrategy):
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

        effective_filters = _prepare_retrieval(self._lexical_index, req)

        try:
            queries = await _expand_queries(req.query, self._query_transformer)
            all_results: list[dict[str, Any]] = []
            candidate_k = req.get_candidate_k()

            for q in queries:
                results = await self._lexical_index.keyword_search(
                    query=q,
                    n_results=candidate_k,
                    text_field=req.text_field,
                    metadata_fields=_required_metadata(req),
                    session=req.session,
                    **effective_filters,
                )
                all_results.extend(results)

            return _dedupe_and_finalize(all_results, req)
        except Exception as exc:
            return _handle_retrieval_exception(exc, "Keyword", req.strict)


class HybridRetrieval(_BaseRetrievalStrategy):
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
                query_vector = await _query_vector(self._embedder, q)
                vector_results = await self._vector_index.vector_search(
                    embedding=query_vector,
                    n_results=candidate_k,
                    threshold=req.threshold,
                    text_field=req.text_field,
                    metadata_fields=_required_metadata(req),
                    session=req.session,
                    **effective_filters,
                )
                keyword_results = await self._lexical_index.keyword_search(
                    query=q,
                    n_results=candidate_k,
                    text_field=req.text_field,
                    metadata_fields=_required_metadata(req),
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

        except Exception as exc:
            return _handle_retrieval_exception(exc, "Hybrid", req.strict)
