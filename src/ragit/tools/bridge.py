"""Framework-neutral tool bridge for RAG operations.

Provides a unified, secure async API for retrieval, source verification,
and status monitoring. Framework integrations (MCP, agent frameworks, APIs)
call this bridge rather than duplicating retrieval and authorization logic.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragit.evidence.assembler import EvidenceAssembler
from ragit.evidence.source_lookup import (
    BoundedSpanResult,
    MemorySnapshotStore,
    SnapshotRecord,
    SourceLookup,
)
from ragit.evidence.tokenizer import ApproximateTokenizer
from ragit.evidence.types import (
    ContextBudget,
    EvidenceBundle,
    EvidenceStatus,
    SearchHit,
)
from ragit.telemetry.health import HealthChecker, HealthReport, HealthState
from ragit.tools.security import ScopeAuthorizer
from ragit.tools.validation import (
    validate_safe_filters,
    validate_safe_identifier,
    validate_safe_string,
)

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for an authorized collection registered in the bridge.

    Attributes:
        name: Collection identifier.
        retriever: Retrieval engine for this collection.
        source_lookup: Source document lookup engine.
        allowed_scopes: Set of valid tenant or partition scopes for this collection.
        metadata: Associated collection metadata.
    """

    name: str
    retriever: Any
    source_lookup: SourceLookup | None = None
    allowed_scopes: set[str] = field(default_factory=lambda: {"*"})
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBridge:
    """Framework-neutral bridge coordinating search, source lookup, and health.

    Enforces input sanitization, tenant scope isolation, bounded context
    budgets, and read/write permission guards.
    """

    def __init__(  # noqa: PLR0913
        self,
        retriever: Any | None = None,
        *,
        collections: Mapping[str, CollectionConfig | Any] | None = None,
        assembler: EvidenceAssembler | None = None,
        source_lookup: SourceLookup | None = None,
        health_checker: HealthChecker | None = None,
        authorizer: ScopeAuthorizer | None = None,
        allow_management: bool = False,
        default_collection: str = "default",
        default_budget_tokens: int = 4000,
    ) -> None:
        """Initialize KnowledgeBridge.

        Args:
            retriever: Default retriever for single-collection setups.
            collections: Optional mapping of collection names to retrievers or configs.
            assembler: Injected EvidenceAssembler (defaults to ApproximateTokenizer).
            source_lookup: Injected SourceLookup (defaults to MemorySnapshotStore).
            health_checker: Injected HealthChecker for status probes.
            authorizer: Injected ScopeAuthorizer for tenant validation.
            allow_management: Opt-in flag to enable management operations (sync, delete).
            default_collection: Name of the fallback default collection.
            default_budget_tokens: Default token budget for evidence assembly.
        """
        self.default_collection = default_collection
        self.default_budget_tokens = default_budget_tokens
        self.allow_management = allow_management
        self.authorizer = authorizer or ScopeAuthorizer(default_allow=True)
        self.assembler = assembler or EvidenceAssembler(tokenizer=ApproximateTokenizer())
        self.source_lookup = source_lookup or SourceLookup(MemorySnapshotStore())
        self.health_checker = health_checker

        # Registry of collections
        self._collections: dict[str, CollectionConfig] = {}
        if collections:
            for name, item in collections.items():
                if isinstance(item, CollectionConfig):
                    self._collections[name] = item
                else:
                    self._collections[name] = CollectionConfig(
                        name=name,
                        retriever=item,
                        source_lookup=self.source_lookup,
                    )

        if retriever is not None and default_collection not in self._collections:
            self._collections[default_collection] = CollectionConfig(
                name=default_collection,
                retriever=retriever,
                source_lookup=self.source_lookup,
            )

    def register_collection(
        self,
        name: str,
        retriever: Any,
        source_lookup: SourceLookup | None = None,
        allowed_scopes: set[str] | None = None,
    ) -> None:
        """Register a new collection in the bridge."""
        validate_safe_identifier(name, "collection name")
        self._collections[name] = CollectionConfig(
            name=name,
            retriever=retriever,
            source_lookup=source_lookup or self.source_lookup,
            allowed_scopes=allowed_scopes or {"*"},
        )

    def _get_collection(self, collection_name: str) -> CollectionConfig:
        """Look up collection configuration or raise KeyError."""
        if collection_name in self._collections:
            return self._collections[collection_name]
        if self._collections and self.default_collection in self._collections:
            return self._collections[self.default_collection]
        if self._collections and len(self._collections) == 1:
            return next(iter(self._collections.values()))
        raise KeyError(
            f"Collection {collection_name!r} not found. "
            f"Available collections: {list(self._collections.keys())}"
        )

    async def search_knowledge(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        collection: str = "default",
        filters: dict[str, Any] | None = None,
        candidate_k: int = 20,
        final_k: int = 5,
        budget_limit: int | ContextBudget | None = None,
        scope: str | None = None,
        client_id: str | None = None,
        distance_threshold: float | None = None,
    ) -> EvidenceBundle:
        """Execute authorized retrieval and evidence assembly.

        Args:
            query: User search query.
            collection: Target knowledge collection selector.
            filters: Narrowing metadata filters.
            candidate_k: Number of raw candidates to retrieve before assembly.
            final_k: Maximum number of evidence items in final bundle.
            budget_limit: Total token budget or preconfigured ContextBudget.
            scope: Isolation scope (e.g. tenant_id) to enforce.
            client_id: Client identifier for authentication/scope checks.
            distance_threshold: Maximum vector distance cutoff.

        Returns:
            Structured :class:`EvidenceBundle` with deduplicated, bounded evidence.
        """
        # 1. Security validation on inputs
        validate_safe_string(query, "query")
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")
        validate_safe_filters(filters)

        # 2. Scope authorization
        self.authorizer.enforce(client_id, collection, scope)

        cfg = self._get_collection(collection)
        retriever = cfg.retriever

        # 3. Retrieve candidates from retriever
        merged_filters = dict(filters or {})
        if scope is not None:
            merged_filters["isolation_scope"] = scope

        raw_results: Any = []
        try:
            if hasattr(retriever, "retrieve"):
                # HybridRetriever or typed retrieval strategy
                raw_results = await retriever.retrieve(
                    query=query,
                    n_results=candidate_k,
                    distance_threshold=distance_threshold,
                    **merged_filters,
                )
            elif hasattr(retriever, "hybrid_search"):
                # VectorStore direct call
                thresh = distance_threshold if distance_threshold is not None else 1.5
                std_meta = [
                    "source_id",
                    "document_id",
                    "revision",
                    "page",
                    "section",
                    "line",
                    "title",
                    "uri",
                    "start_offset",
                    "end_offset",
                    "freshness",
                    "isolation_scope",
                ]
                if filters:
                    for k in filters:
                        if k not in std_meta:
                            std_meta.append(k)
                raw_results = await retriever.hybrid_search(
                    query=query,
                    embedding=[],
                    n_results=candidate_k,
                    threshold=thresh,
                    metadata_fields=std_meta,
                    **merged_filters,
                )

            elif callable(retriever):
                raw_results = await retriever(query=query, n_results=candidate_k, **merged_filters)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval execution failed for collection %r: %s", collection, exc)
            budget = (
                budget_limit
                if isinstance(budget_limit, ContextBudget)
                else ContextBudget(
                    total_budget=budget_limit or self.default_budget_tokens,
                    max_evidence_items=final_k,
                )
            )
            return EvidenceBundle(
                items=[],
                status=EvidenceStatus.BACKEND_FAILURE,
                total_tokens=0,
                budget=budget,
                status_detail=f"Backend retrieval failure: {exc}",
            )

        # 4. Standardize candidate results into SearchHits
        hits: list[SearchHit] = []
        if isinstance(raw_results, Sequence):
            for res in raw_results:
                if isinstance(res, SearchHit):
                    hits.append(res)
                elif hasattr(res, "to_search_hit"):
                    hits.append(res.to_search_hit())
                elif hasattr(res, "document") and hasattr(res, "id"):
                    hits.append(SearchHit.from_retrieved_document(res))
                elif isinstance(res, dict):
                    hits.append(SearchHit.from_dict(res))

        # 5. Build budget
        if isinstance(budget_limit, ContextBudget):
            budget = budget_limit
        else:
            budget = ContextBudget(
                total_budget=budget_limit or self.default_budget_tokens,
                max_evidence_items=final_k,
            )

        # 6. Assemble evidence
        return self.assembler.assemble(hits, budget=budget)

    async def get_source(  # noqa: PLR0913, PLR0917
        self,
        source_ref: str,
        document_id: str | None = None,
        revision: str = "v1",
        location: dict[str, Any] | tuple[int, int] | None = None,
        collection: str = "default",
        scope: str | None = None,
        client_id: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        max_span_length: int | None = None,
    ) -> BoundedSpanResult:
        """Fetch an authorized, bounded text span and revision metadata.

        Args:
            source_ref: Evidence source reference or source repository identifier.
            document_id: Document identifier (parsed from source_ref if omitted).
            revision: Document revision snapshot identifier.
            location: Optional location map or offset tuple.
            collection: Knowledge collection selector.
            scope: Isolation scope.
            client_id: Client identifier for authorization.
            start_offset: Character start offset.
            end_offset: Character end offset.
            max_span_length: Maximum allowed span length.

        Returns:
            :class:`BoundedSpanResult` containing verified slice and provenance.
        """
        validate_safe_string(source_ref, "source_ref")
        if document_id is not None:
            validate_safe_identifier(document_id, "document_id")
        validate_safe_identifier(revision, "revision")
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        self.authorizer.enforce(client_id, collection, scope)

        cfg = self._get_collection(collection)
        lookup = cfg.source_lookup or self.source_lookup

        # Parse source_ref if document_id is not provided
        resolved_source_id = source_ref
        resolved_doc_id = document_id
        resolved_revision = revision

        if resolved_doc_id is None:
            if "@" in source_ref and ":" in source_ref:
                # Format: source_id@revision:document_id
                parts = source_ref.split("@", 1)
                resolved_source_id = parts[0]
                rev_doc = parts[1].split(":", 1)
                resolved_revision = rev_doc[0]
                resolved_doc_id = rev_doc[1]
            elif ":" in source_ref:
                # Format: source_id:document_id
                parts = source_ref.split(":", 1)
                resolved_source_id = parts[0]
                resolved_doc_id = parts[1]
            else:
                resolved_doc_id = source_ref

        # Parse location parameters
        start = start_offset
        end = end_offset

        if isinstance(location, tuple) and len(location) == 2:
            start = location[0]
            end = location[1]
        elif isinstance(location, dict):
            if "start_offset" in location:
                start = location.get("start_offset")
            if "end_offset" in location:
                end = location.get("end_offset")

        auth_scope = scope if scope is not None else "default"
        return lookup.lookup_span(
            source_id=resolved_source_id,
            document_id=resolved_doc_id,
            revision=resolved_revision,
            authorization_scope=auth_scope,
            start_offset=start,
            end_offset=end,
            max_span_length=max_span_length,
            strict=False,
        )

    async def knowledge_status(
        self,
        collection: str = "default",
        scope: str | None = None,
        client_id: str | None = None,
        include_expensive: bool = False,
    ) -> dict[str, Any]:
        """Inspect health, count, and freshness summary for a collection.

        Output is sanitized to exclude sensitive credentials, local paths, or internal URIs.

        Args:
            collection: Authorized collection selector.
            scope: Isolation scope to inspect.
            client_id: Client identifier for authorization check.
            include_expensive: Whether to run deep consistency verification.

        Returns:
            Sanitized dictionary containing state, document counts, and component health.
        """
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        self.authorizer.enforce(client_id, collection, scope)

        cfg = self._get_collection(collection)
        retriever = cfg.retriever

        # Run health check if checker is available
        if self.health_checker is not None:
            report: HealthReport = await self.health_checker.check_health(
                scope=scope,
                include_expensive=include_expensive,
            )
            raw_dict = report.to_dict()
            # Sanitize components to avoid leaking internal paths or connection strings
            sanitized_components = {}
            for name, comp in raw_dict.get("components", {}).items():
                details = comp.get("details", {})
                sanitized_details = {
                    k: v
                    for k, v in details.items()
                    if k not in ("connection_string", "password", "url", "dsn", "secret")
                }
                sanitized_components[name] = {
                    "state": comp.get("state"),
                    "reason": comp.get("reason"),
                    "latency_ms": comp.get("latency_ms"),
                    "details": sanitized_details,
                }

            return {
                "collection": collection,
                "scope": scope,
                "state": raw_dict.get("state", HealthState.READY.value),
                "timestamp": raw_dict.get("timestamp"),
                "reasons": raw_dict.get("reasons", []),
                "components": sanitized_components,
                "is_healthy": report.is_healthy,
            }

        # Fallback baseline probe on the retriever
        doc_count = 0
        state = HealthState.READY.value
        reason = ""
        try:
            if hasattr(retriever, "count"):
                doc_count = await retriever.count()
        except Exception as exc:  # noqa: BLE001
            state = HealthState.DEGRADED.value
            reason = f"Retrieval probe warning: {exc}"

        return {
            "collection": collection,
            "scope": scope,
            "state": state,
            "reason": reason,
            "document_count": doc_count,
            "is_healthy": state == HealthState.READY.value,
        }

    # -------------------------------------------------------------------------
    # Management Operations (Opt-in, Permission-Gated)
    # -------------------------------------------------------------------------

    async def sync_collection(  # noqa: PLR0913
        self,
        collection: str,
        source_id: str,
        documents: Sequence[Any],
        scope: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest or sync documents into a collection.

        Guarded by :attr:`allow_management`.

        Raises:
            PermissionError: If management tools are not permitted by host.
        """
        if not self.allow_management:
            raise PermissionError(
                "Management operations (sync/ingestion) are disabled by host security rules. "
                "Enable them explicitly with allow_management=True."
            )

        validate_safe_identifier(collection, "collection")
        validate_safe_string(source_id, "source_id")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        self.authorizer.enforce(client_id, collection, scope)

        cfg = self._get_collection(collection)
        retriever = cfg.retriever
        lookup = cfg.source_lookup or self.source_lookup

        # If retriever is or has a vector store, upsert
        store = getattr(retriever, "store", retriever)
        ingested_count = 0

        # Also register snapshots in source lookup for immutable citation tracking
        snapshot_store = getattr(lookup, "_store", None)

        for doc in documents:
            doc_id = getattr(doc, "id", None) or doc.get("id") if isinstance(doc, dict) else ""
            text = (
                getattr(doc, "text", None)
                or getattr(doc, "document", None)
                or (doc.get("text") or doc.get("document") if isinstance(doc, dict) else "")
            )
            revision = (
                getattr(doc, "revision", None)
                or (doc.get("revision") if isinstance(doc, dict) else None)
                or "v1"
            )
            metadata = (
                getattr(doc, "metadata", None)
                or (doc.get("metadata") if isinstance(doc, dict) else {})
                or {}
            )

            validate_safe_identifier(str(doc_id), "document id")
            validate_safe_string(text, "document text")
            validate_safe_identifier(str(revision), "document revision")

            if snapshot_store is not None and hasattr(snapshot_store, "put"):
                snapshot = SnapshotRecord(
                    source_id=source_id,
                    document_id=str(doc_id),
                    revision=str(revision),
                    content=text,
                    isolation_scope=scope,
                    metadata=dict(metadata),
                )
                snapshot_store.put(snapshot)

        if hasattr(store, "upsert"):
            await store.upsert(documents)
            ingested_count = len(documents)

        return {
            "status": "success",
            "collection": collection,
            "source_id": source_id,
            "documents_synced": ingested_count,
        }

    async def revoke_document(  # noqa: PLR0913
        self,
        collection: str,
        document_id: str,
        source_id: str = "default_source",
        scope: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Revoke/delete a document and its chunks from the collection.

        Guarded by :attr:`allow_management`.

        Raises:
            PermissionError: If management tools are not permitted by host.
        """
        if not self.allow_management:
            raise PermissionError(
                "Management operations (deletion/revocation) are disabled by host security rules. "
                "Enable them explicitly with allow_management=True."
            )

        validate_safe_identifier(collection, "collection")
        validate_safe_identifier(document_id, "document_id")
        validate_safe_string(source_id, "source_id")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        self.authorizer.enforce(client_id, collection, scope)

        cfg = self._get_collection(collection)
        retriever = cfg.retriever
        store = getattr(retriever, "store", retriever)

        deleted_count = 0
        if hasattr(store, "delete"):
            filters: dict[str, Any] = {"document_id": document_id}
            if scope is not None:
                filters["isolation_scope"] = scope
            deleted_count = await store.delete(filters=filters)

        return {
            "status": "revoked",
            "collection": collection,
            "document_id": document_id,
            "chunks_removed": deleted_count,
        }
