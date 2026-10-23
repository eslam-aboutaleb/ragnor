"""Framework-neutral tool bridge for RAG operations.

Provides a unified, secure async API for retrieval, source verification,
and status monitoring. Framework integrations (MCP, agent frameworks, APIs)
call this bridge rather than duplicating retrieval and authorization logic.
"""

from __future__ import annotations

import logging
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ragnor.evidence.assembler import EvidenceAssembler
from ragnor.evidence.references import parse_source_reference
from ragnor.retrieval.retriever import HybridRetriever
from ragnor.retrieval.strategies import DenseRetrieval, HybridRetrieval, KeywordRetrieval
from ragnor.retrieval.types import RetrievalRequest
from ragnor.evidence.types import ScopeAuthorizationError
from ragnor.evidence.source_lookup import (
    BoundedSpanResult,
    MemorySnapshotStore,
    SnapshotRecord,
    SourceLookup,
)
from ragnor.evidence.tokenizer import ApproximateTokenizer
from ragnor.evidence.types import (
    ContextBudget,
    EvidenceBundle,
    EvidenceStatus,
    SearchHit,
)
from ragnor.telemetry.health import HealthChecker, HealthReport, HealthState
from ragnor.tools.security import ScopeAuthorizer
from ragnor.tools.validation import (
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
    scope_field: str | None = None


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
        production_mode: bool = False,
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
            production_mode: Reject implicit allow-all authorization if no
                host ScopeAuthorizer is supplied. Backwards compatible default.
            default_collection: Name of the fallback default collection.
            default_budget_tokens: Default token budget for evidence assembly.
        """
        self.default_collection = default_collection
        self.default_budget_tokens = default_budget_tokens
        self.allow_management = allow_management
        if production_mode and (authorizer is None or not authorizer.has_explicit_restrictions):
            raise ValueError(
                "production KnowledgeBridge requires a restrictive host authorizer; "
                "the development default allows all scopes"
            )
        self.authorizer = authorizer or ScopeAuthorizer(default_allow=True)
        self.assembler = assembler or EvidenceAssembler(tokenizer=ApproximateTokenizer())
        self.source_lookup = source_lookup or SourceLookup(MemorySnapshotStore())
        self.health_checker = health_checker

        # Registry of collections
        self._collections: dict[str, CollectionConfig] = {}
        if collections:
            for name, item in collections.items():
                if isinstance(item, CollectionConfig):
                    if item.name != name:
                        raise ValueError("Collection registry key must match its configured name")
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
        scope_field: str | None = None,
    ) -> None:
        """Register a new collection in the bridge."""
        validate_safe_identifier(name, "collection name")
        self._collections[name] = CollectionConfig(
            name=name,
            retriever=retriever,
            source_lookup=source_lookup or self.source_lookup,
            allowed_scopes={"*"} if allowed_scopes is None else set(allowed_scopes),
            scope_field=scope_field,
        )

    def _get_collection(self, collection_name: str) -> CollectionConfig:
        """Look up collection configuration or raise KeyError."""
        if collection_name in self._collections:
            return self._collections[collection_name]
        raise KeyError(
            f"Collection {collection_name!r} not found. "
            f"Available collections: {list(self._collections.keys())}"
        )

    def _authorize(
        self, collection: str, scope: str | None, client_id: str | None
    ) -> CollectionConfig:
        cfg = self._get_collection(collection)
        self.authorizer.enforce(client_id, cfg.name, scope)
        if "*" not in cfg.allowed_scopes and scope not in cfg.allowed_scopes:
            raise ScopeAuthorizationError("Collection does not permit the requested scope")
        return cfg

    @staticmethod
    def _scope_field(cfg: CollectionConfig) -> str:
        if cfg.scope_field is not None:
            return cfg.scope_field
        if isinstance(
            cfg.retriever, (HybridRetriever, DenseRetrieval, HybridRetrieval, KeywordRetrieval)
        ):
            return "tenant_id"
        return "scope" if hasattr(cfg.retriever, "_manifest_store") else "isolation_scope"

    async def search_knowledge(  # noqa: PLR0913, PLR0917, PLR0915
        self,
        query: str,
        collection: str | None = None,
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
        # Security validation on inputs
        validate_safe_string(query, "query")
        collection = self.default_collection if collection is None else collection
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")
        validate_safe_filters(filters)

        # Scope authorization
        cfg = self._authorize(collection, scope, client_id)
        retriever = cfg.retriever

        # Retrieve candidates from retriever
        merged_filters = dict(filters or {})
        scope_field = self._scope_field(cfg)
        if scope is not None:
            merged_filters[scope_field] = scope
        for count in (candidate_k, final_k):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("candidate_k and final_k must be positive integers")

        raw_results: Any = []
        try:
            if isinstance(
                retriever, (HybridRetriever, DenseRetrieval, HybridRetrieval, KeywordRetrieval)
            ):
                threshold = distance_threshold
                if threshold is None and not isinstance(retriever, HybridRetriever):
                    threshold = 1.5
                request = RetrievalRequest(
                    query,
                    final_k=candidate_k,
                    candidate_k=candidate_k,
                    threshold=threshold,
                    filters=merged_filters,
                    isolation_scope=scope,
                    scope_field=scope_field,
                    strict=True,
                    metadata_fields=[
                        "source_id",
                        "document_id",
                        "doc_id",
                        "revision",
                        "page",
                        "section",
                        "line",
                        "title",
                        "uri",
                        "start_offset",
                        "end_offset",
                        "freshness",
                        scope_field,
                    ],
                )
                raw_results = (await retriever.retrieve_typed(request)).to_dicts()
            elif hasattr(retriever, "retrieve"):
                parameters = inspect.signature(retriever.retrieve).parameters
                options = dict(merged_filters)
                if "distance_threshold" in parameters:
                    options["distance_threshold"] = distance_threshold
                elif "threshold" in parameters and distance_threshold is not None:
                    options["threshold"] = distance_threshold
                raw_results = await retriever.retrieve(
                    query=query, n_results=candidate_k, **options
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
                    scope_field,
                    "doc_id",
                    "provenance",
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
                    strict=True,
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

        # Standardize candidate results into SearchHits
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

        # Build budget
        if isinstance(budget_limit, ContextBudget):
            budget = budget_limit
        else:
            budget = ContextBudget(
                total_budget=budget_limit or self.default_budget_tokens,
                max_evidence_items=final_k,
            )

        # Assemble evidence
        return self.assembler.assemble(hits, budget=budget)

    async def get_source(  # noqa: PLR0913, PLR0917
        self,
        source_ref: str,
        document_id: str | None = None,
        revision: str = "v1",
        location: dict[str, Any] | tuple[int, int] | None = None,
        collection: str | None = None,
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
        collection = self.default_collection if collection is None else collection
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        cfg = self._authorize(collection, scope, client_id)
        lookup = cfg.source_lookup or self.source_lookup

        resolved_source_id = source_ref
        resolved_doc_id = document_id
        resolved_revision = revision
        parsed_start = parsed_end = None
        if document_id is None:
            (resolved_source_id, resolved_doc_id, parsed_revision, parsed_start, parsed_end) = (
                parse_source_reference(source_ref)
            )
            resolved_revision = parsed_revision or revision

        # Parse location parameters
        start = parsed_start if start_offset is None else start_offset
        end = parsed_end if end_offset is None else end_offset

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
        collection: str | None = None,
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
        collection = self.default_collection if collection is None else collection
        validate_safe_identifier(collection, "collection")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        cfg = self._authorize(collection, scope, client_id)
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

    async def sync_collection(  # noqa: PLR0913, PLR0915
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

        collection = self.default_collection if collection is None else collection
        validate_safe_identifier(collection, "collection")
        validate_safe_string(source_id, "source_id")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        cfg = self._authorize(collection, scope, client_id)
        retriever = cfg.retriever
        lookup = cfg.source_lookup or self.source_lookup

        # If retriever is or has a vector store, upsert
        store = getattr(retriever, "store", retriever)
        ingested_count = 0

        # Also register snapshots in source lookup for immutable citation tracking
        snapshot_store = getattr(lookup, "_store", None)

        normalized: list[dict[str, Any]] = []
        snapshots: list[SnapshotRecord] = []
        scope_field = self._scope_field(cfg)
        for doc in documents:
            row = dict(doc) if isinstance(doc, Mapping) else vars(doc).copy()
            doc_id = str(row.get("id") or row.get("external_id") or row.get("source_name") or "")
            text = row.get("text", row.get("document", row.get("content", "")))
            metadata = dict(row.get("metadata") or {})
            revision = str(row.get("revision") or metadata.get("revision") or "v1")
            parent_id = str(metadata.get("document_id") or doc_id)
            validate_safe_identifier(doc_id, "document id")
            validate_safe_string(text, "document text")
            validate_safe_identifier(revision, "document revision")
            for key in {scope_field, "tenant_id", "scope", "isolation_scope"}:
                if key in metadata and metadata[key] != scope:
                    raise ScopeAuthorizationError(
                        "Document ownership conflicts with authorized scope"
                    )
            if "source_id" in metadata and metadata["source_id"] != source_id:
                raise ScopeAuthorizationError("Document source conflicts with authorized source")
            metadata.update(source_id=source_id, document_id=parent_id, revision=revision)
            if scope is not None:
                metadata[scope_field] = scope
            key = json.dumps([cfg.name, scope, source_id, doc_id], separators=(",", ":"))
            normalized.append(
                {
                    **row,
                    "id": hashlib.sha256(key.encode()).hexdigest(),
                    "text": text,
                    "metadata": metadata,
                }
            )
            snapshots.append(SnapshotRecord(source_id, parent_id, revision, text, scope, metadata))

        # Validate all immutable revisions before mutating vectors or snapshots.
        pending: dict[tuple[str, str], SnapshotRecord] = {}
        for snapshot in snapshots:
            identity = (snapshot.document_id, snapshot.revision)
            previous = pending.get(identity)
            if previous is not None and (
                previous.content != snapshot.content or previous.metadata != snapshot.metadata
            ):
                raise ValueError("Conflicting document revisions in the same batch")
            pending[identity] = snapshot
            if snapshot_store is not None:
                if hasattr(snapshot_store, "get_scoped"):
                    existing = snapshot_store.get_scoped(
                        source_id, snapshot.document_id, snapshot.revision, scope
                    )
                else:
                    existing = snapshot_store.get(
                        source_id, snapshot.document_id, snapshot.revision
                    )
                if (
                    existing is not None
                    and existing.isolation_scope == scope
                    and (
                        existing.content != snapshot.content
                        or existing.metadata != snapshot.metadata
                    )
                ):
                    raise ValueError("Cannot replace an immutable document revision")

        if hasattr(store, "upsert"):
            await store.upsert(normalized)
            ingested_count = len(normalized)
        if snapshot_store is not None:
            for snapshot in snapshots:
                snapshot_store.put(snapshot)

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

        collection = self.default_collection if collection is None else collection
        validate_safe_identifier(collection, "collection")
        validate_safe_identifier(document_id, "document_id")
        validate_safe_string(source_id, "source_id")
        if scope is not None:
            validate_safe_identifier(scope, "scope")

        cfg = self._authorize(collection, scope, client_id)
        retriever = cfg.retriever
        store = getattr(retriever, "store", retriever)

        lookup = cfg.source_lookup or self.source_lookup
        lookup.revoke(source_id, document_id, scope)
        deleted_count = 0
        if hasattr(store, "delete"):
            filters: dict[str, Any] = {"document_id": document_id, "source_id": source_id}
            if scope is not None:
                filters[self._scope_field(cfg)] = scope
            deleted_count = await store.delete(filters=filters)

        return {
            "status": "revoked",
            "collection": collection,
            "document_id": document_id,
            "chunks_removed": deleted_count,
        }
