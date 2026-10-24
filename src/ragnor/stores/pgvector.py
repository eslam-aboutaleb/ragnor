"""
pgvector implementation of the VectorStore interface.

Provides hybrid search (vector + PostgreSQL full-text search RRF), document counting, and upsert
operations against PostgreSQL tables with pgvector and tsvector columns.

Security model:
  - All user-supplied **values** are passed as bound parameters via SQLAlchemy's
    parameterized queries. They never appear in the SQL string, so they cannot
    inject SQL.
  - All **SQL identifiers** (table names, column names, filter keys) are
    validated against a strict allowlist regex (``^[A-Za-z_][A-Za-z0-9_]*$``)
    before interpolation into raw SQL strings. This prevents SQL injection even
    when identifiers are derived from configuration.
  - Raw SQL is used because the hybrid search query requires complex CTEs with
    window functions, pgvector operators, and PostgreSQL full-text search that are
    impractical to express in SQLAlchemy ORM/Core while remaining readable.

Configuration (no host imports): the store receives everything through
constructor arguments — ``session_factory`` (a
:class:`ragnor.db.session.SessionProvider`), ``embedding_dim``, the host's
``RagSettings``, and an optional ``embedding_fn`` used to auto-embed queries
when the caller passes an empty embedding.

Distance metric: the vector stage ranks by the store's metric
(:mod:`ragnor.stores.metrics`). The operator comes from an internal map — ``l2``
maps to pgvector's ``<->`` (the default) and ``cosine`` to ``<=>`` — so it
can never come from user input. pgvector's ``<=>`` is the exact cosine
distance in ``[0, 2]`` and needs no pre-normalization. Thresholds are
distances under that metric and are therefore metric-relative: never
comparable across metrics, so a host that switches metrics must re-tune.
The matching index opclass (``vector_l2_ops`` vs ``vector_cosine_ops``) is
host-owned DDL — ragnor never creates indexes.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from ragnor.core.config import RagSettings
from ragnor.db.sql_factory import SQLAlchemyFactory
from ragnor.stores.metrics import DistanceMetric, resolve_distance_metric
from ragnor.stores._scoring import format_search_hit, project_metadata
from ragnor.stores.base import EmbeddingFunction, VectorStore
from ragnor.stores.registry import vector_store_registry
from ragnor.core.exceptions import BackendUnavailableError, OperationTimeoutError
from ragnor.core.validation import (
    validate_delete_selectors,
    validate_embedding,
    validate_identifier,
    validate_search_identifiers,
    validate_upsert_identifiers,
)


if TYPE_CHECKING:
    from ragnor.db.session import SessionProvider
    from ragnor.db.sql_factory import SQLFactory

logger = logging.getLogger(__name__)

_DISTANCE_OPERATORS: dict[DistanceMetric, str] = {"l2": "<->", "cosine": "<=>"}
"""pgvector distance operator per resolved metric; never read from user input."""


def _keyword_score(row: Any) -> float | None:
    """Return the PostgreSQL full-text rank from a hybrid search row, if present.

    The column is absent for a vector-only hit, and the projection is not guaranteed
    across every caller, so membership is checked rather than assumed.
    """
    if "keyword_score" not in row:
        return None
    value = row["keyword_score"]
    return None if value is None else float(value)


@vector_store_registry.register("pgvector")
class PgVectorStore(VectorStore):
    """pgvector-based vector store implementation.

    Supports hybrid search combining vector similarity (the pgvector
    operator selected by the store's distance metric: ``<->`` for
    the default ``l2`` metric, ``<=>`` for ``cosine``) with
    PostgreSQL full-text search (tsvector/tsquery) using Reciprocal
    Rank Fusion. The metric is fixed at store-construction time: the
    distance operator is resolved once in ``__init__`` from
    ``self.metric``, so every search uses that metric for the store's
    lifetime.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        table_name: str,
        id_field: str = "id",
        embedding_dim: int | None = None,
        session_factory: SessionProvider | None = None,
        settings: RagSettings | None = None,
        embedding_fn: EmbeddingFunction | None = None,
        tsquery_config: str = "english",
        sql_factory: SQLFactory | None = None,
        metric: str | None = None,
    ):
        """Initialize the vector store with target table configuration.

        Args:
            table_name: Name of the database table containing vector data.
            id_field: Name of the primary key column. Defaults to "id".
            embedding_dim: Dimensionality of the embedding vectors. Required;
                pass the dimension configured for the active embedding model.
            session_factory: Session provider used when a method is called
                without an explicit session. When omitted and ``settings``
                carries a ``database_url``, one is created lazily through
                the SQL factory.
            settings: Host settings. Used only to resolve a default session
                factory when ``session_factory`` is omitted, and to read the
                optional ``rag_distance_metric`` field when ``metric`` is
                omitted.
            embedding_fn: Async callable that embeds a list of texts. Used
                only when ``hybrid_search`` is called with an empty embedding.
            tsquery_config: PostgreSQL text-search configuration used for
                ``plainto_tsquery`` (must match the tsvector column's
                configuration). Defaults to 'english'.
            sql_factory: SQL factory that runs every statement the
                store issues; defaults to the SQLAlchemy factory.
                Inject a custom :class:`ragnor.db.sql_factory.SQLFactory`
                to route the store's SQL through another stack.
            metric: Distance metric for the vector stage (``"l2"`` by
                default, ``"cosine"`` opt-in). Resolution order: this
                argument, the optional ``rag_distance_metric`` host
                settings field, then ``"l2"``.

        Raises:
            ValueError: If ``table_name`` or ``id_field`` contain unsafe
                characters, if ``embedding_dim`` is omitted, if
                ``tsquery_config`` is not a lowercase identifier, or if
                ``metric`` is not a supported distance metric.
        """
        validate_identifier(table_name, "table_name")
        validate_identifier(id_field, "id_field")
        self.table_name = table_name
        self.id_field = id_field
        if embedding_dim is None:
            raise ValueError(
                "embedding_dim is required; pass the dimension configured "
                "for the active embedding model"
            )
        self.embedding_dim = embedding_dim
        if not re.fullmatch(r"[a-z][a-z0-9_]*", tsquery_config):
            raise ValueError(
                f"tsquery_config must be a lowercase identifier; got {tsquery_config!r}"
            )
        self.tsquery_config = tsquery_config
        self._settings = settings
        self._session_factory = session_factory
        self._resolved_session_factory: SessionProvider | None = session_factory
        self._embedding_fn = embedding_fn
        self._sql = sql_factory or SQLAlchemyFactory()
        self.metric: DistanceMetric = resolve_distance_metric(metric, settings)
        self._distance_operator = _DISTANCE_OPERATORS[self.metric]

    def _resolve_session_factory(self) -> SessionProvider:
        """Return the session factory, creating one from settings if needed.

        The factory is resolved at most once per store instance so the
        underlying engine is never duplicated.

        Raises:
            RuntimeError: If neither a session factory nor settings with a
                ``database_url`` were provided.
        """
        if self._resolved_session_factory is None:
            if self._settings is None:
                raise RuntimeError(
                    "PgVectorStore needs a session_factory or settings with "
                    "a database_url to open its own sessions"
                )
            self._resolved_session_factory = self._sql.create_session_factory(
                self._settings.database_url
            )
        return self._resolved_session_factory

    async def _execute_search_query(  # noqa: PLR0913
        self,
        session: Any,
        stmt: str,
        params: dict[str, Any],
        text_field: str,
        metadata_fields: list[str],
        *,
        distance_field: str | None = None,
        keyword_field: str | None = None,
    ) -> list[dict[str, Any]]:
        owns_session = session is None
        if owns_session:
            session_factory = self._resolve_session_factory()
            async with session_factory() as owned_session:
                rows = await self._sql.fetch_all(owned_session, stmt, params)
        else:
            rows = await self._sql.fetch_all(session, stmt, params)

        results = []
        for row in rows:
            dist = row[distance_field] if distance_field and distance_field in row else None
            kw = row[keyword_field] if keyword_field and keyword_field in row else None
            results.append(
                format_search_hit(
                    doc_id=str(row[self.id_field]),
                    document=row[text_field],
                    metadata=project_metadata(row, metadata_fields),
                    distance=dist,
                    keyword_score=kw,
                )
            )
        return results

    @property
    def capabilities(self) -> Any:
        """Declared store capabilities."""
        from ragnor.retrieval.capabilities import DEFAULT_PGVECTOR_CAPABILITIES  # noqa: PLC0415

        return DEFAULT_PGVECTOR_CAPABILITIES

    async def hybrid_search(  # noqa: PLR0913, PLR0915, PLR0917
        self,
        query: str,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Hybrid search with vector + PostgreSQL full-text search RRF.

        Executes a two-stage retrieval:
          1. Vector search using the pgvector distance operator for the
             store's metric (``<->`` L2 by default, ``<=>`` cosine) with
             a distance threshold.
          2. PostgreSQL full-text search using tsvector/tsquery.

        Results from both stages are merged using Reciprocal Rank Fusion (RRF),
        which combines the ranks from each retrieval method to produce a unified
        ranking. This approach is robust when one retrieval method fails to
        find relevant documents.

        Args:
            query: Natural language search query.
            embedding: Query embedding vector. If empty, the query is embedded
                automatically using the injected ``embedding_fn``.
            n_results: Maximum number of results to return.
            threshold: Maximum distance under the store's metric (L2 or
                cosine) for the vector search part. Smaller values enforce
                stricter similarity matching, and thresholds are
                metric-relative: never comparable across metrics.
            text_field: Name of the column containing the document text.
            metadata_fields: List of additional column names to include in the
                result metadata dict.
            session: Optional external SQLAlchemy async session. If provided,
                this session is reused instead of creating a new one. When an
                external session is supplied, the caller is responsible for
                committing or rolling back the transaction.
            strict: When ``True``, a failed search raises
                :class:`ragnor.core.types.RetrievalError`. The default ``False``
                preserves the historical behavior of logging the error and
                returning an empty list.
            **filters: Additional equality filters applied to both search CTEs
                (e.g., ``tenant_id="..."``). Keys must be valid SQL identifiers;
                values are bound as parameters to prevent SQL injection.

        Returns:
            list[dict]: A list of result dicts sorted by RRF score, each
            containing:
                - ``id`` (str): The document id.
                - ``document`` (str): The retrieved text.
                - ``metadata`` (dict): Structured metadata fields.
                - ``distance`` (float): Distance under the store's metric
                  (L2 or cosine) from the vector search, or ``None`` for
                  keyword-only hits.
                - ``keyword_score`` (float): PostgreSQL full-text rank, or
                  ``None`` for vector-only hits.
                - ``_rrf_score`` (float): Combined RRF relevance score.

        Raises:
            ragnor.core.types.RetrievalError: If the search fails and
                ``strict=True``.
        """
        validate_identifier(text_field, "text_field")
        for key in filters:
            validate_identifier(key, "filter key")
        metadata_fields = metadata_fields or []
        for field in metadata_fields:
            validate_identifier(field, "metadata_field")

        if n_results < 1:
            raise ValueError(f"n_results must be a positive integer; got {n_results}")

        if not embedding:
            if self._embedding_fn is None:
                raise ValueError(
                    "embedding_fn is required to auto-embed the query; "
                    "pass an embedding or construct the store with embedding_fn"
                )
            embedding_list = await self._embedding_fn([query])
            embedding = embedding_list[0]

        validate_embedding(embedding, self.embedding_dim, label="query embedding")
        metadata_fields = metadata_fields or []

        # Build SELECT clause for metadata fields so they are returned in
        # both the vector and keyword CTEs. This ensures metadata is available
        # regardless of which search method found the document.
        vector_select = [
            self.id_field,
            text_field,
        ]
        vector_select.extend(metadata_fields)
        vector_select_sql = ", ".join(vector_select)

        keyword_select = [
            self.id_field,
            text_field,
        ]
        keyword_select.extend(metadata_fields)
        keyword_select_sql = ", ".join(keyword_select)

        if metadata_fields:
            # Each COALESCE column is aliased with its field name.
            # Without the alias PostgreSQL labels every one of them
            # "coalesce", so the result mapping never contains the
            # field names and every result would carry empty metadata.
            meta_coalesce = ", ".join(f"COALESCE(v.{f}, k.{f}) AS {f}" for f in metadata_fields)
        else:
            meta_coalesce = ""

        # The outer projection is assembled as a list so that an empty
        # ``metadata_fields`` contributes no element. Interpolating the metadata
        # expression as ``f"{meta_coalesce},"`` emits a bare comma when it is empty,
        # producing ``AS document, , v.distance`` — a syntax error that is then
        # swallowed by the broad ``except`` below and reported as "no results".
        outer_select = [
            f"COALESCE(v.{self.id_field}, k.{self.id_field}) AS id",
            f"COALESCE(v.{text_field}, k.{text_field}) AS document",
        ]
        if metadata_fields:
            outer_select.append(meta_coalesce)
        outer_select.extend(
            [
                "v.distance AS distance",
                "k.score AS keyword_score",
                (
                    "COALESCE(1.0 / (60 + v.rank), 0.0) "
                    "+ COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score"
                ),
            ]
        )
        outer_select_sql = ",\n                 ".join(outer_select)

        # Build JOIN condition between the two CTEs. The primary join is on
        # the document ID field, with optional additional filters applied
        # to both sides to ensure consistent scoping.
        join_conditions = [f"v.{self.id_field} = k.{self.id_field}"]

        # Build query parameters and WHERE fragments from validated filters.
        # All user-supplied values are bound as parameters to prevent SQL
        # injection; only pre-validated identifiers are interpolated.
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        params: dict[str, Any] = {
            "query": query,
            "embedding": embedding_str,
            "distance_threshold": threshold,
            "n_results": n_results,
            "tsquery_config": self.tsquery_config,
        }
        # Filter parameters are namespaced with a ``filter_`` prefix so a
        # filter key can never shadow the reserved bind parameters above
        # (``query``, ``embedding``, ``distance_threshold``, ``n_results``).
        filter_parts: list[str] = []
        for key, value in filters.items():
            param_name = f"filter_{key}"
            filter_parts.append(f"{key} = :{param_name}")
            params[param_name] = value

        extra_where = " AND ".join(filter_parts) if filter_parts else None
        if extra_where:
            join_conditions.append(" AND ".join(f"v.{key} = :filter_{key}" for key in filters))

        join_sql = " AND ".join(join_conditions)

        # The text-search expression is built once and interpolated
        # so the SQL string stays readable; the config value itself
        # is a bound parameter (cast to regconfig), never interpolated.
        tsquery_expr = "plainto_tsquery(cast(:tsquery_config as regconfig), :query)"

        # The distance operator comes from the internal metric map only,
        # never from user input, so the SQL stays injection-safe. The
        # expression is interpolated (not bound) because pgvector
        # operators cannot be parameterized.
        distance_expr = f"embedding {self._distance_operator} cast(:embedding as vector)"

        stmt = f"""
            WITH vector_search AS (
                SELECT {vector_select_sql},
                       {distance_expr} AS distance,
                       ROW_NUMBER() OVER (
                           ORDER BY
                               {distance_expr},
                               {self.id_field}
                       ) AS rank
                FROM {self.table_name}
                WHERE {distance_expr} < :distance_threshold
                {f"AND {extra_where}" if extra_where else ""}
                LIMIT :n_results
            ),
            keyword_search AS (
                SELECT {keyword_select_sql},
                       ts_rank(tsvector, {tsquery_expr}) AS score,
                       ROW_NUMBER() OVER (
                           ORDER BY
                               ts_rank(tsvector, {tsquery_expr}) DESC,
                               {self.id_field}
                       ) AS rank
                FROM {self.table_name}
                WHERE tsvector @@ {tsquery_expr}
                {f"AND {extra_where}" if extra_where else ""}
                LIMIT :n_results
            )
            SELECT
                {outer_select_sql}
            FROM vector_search v
            FULL OUTER JOIN keyword_search k ON {join_sql}
            ORDER BY rrf_score DESC, {self.id_field}
            LIMIT :n_results
        """

        try:
            if session is not None:
                rows = await self._sql.fetch_all(session, stmt, params)
            else:
                session_factory = self._resolve_session_factory()
                async with session_factory() as owned_session:
                    rows = await self._sql.fetch_all(owned_session, stmt, params)

            retrieved = []
            for row in rows:
                metadata = {}
                for field in metadata_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]

                retrieved.append(
                    {
                        "id": str(row["id"]) if row["id"] is not None else None,
                        "document": row["document"],
                        "metadata": metadata,
                        "distance": float(row["distance"]) if row["distance"] is not None else None,
                        "keyword_score": _keyword_score(row),
                        "_rrf_score": float(row["rrf_score"]),
                    }
                )

            logger.debug(
                "hybrid_search on %s: found %d results.",
                self.table_name,
                len(retrieved),
            )
            return retrieved
        except Exception as exc:
            logger.error("Hybrid search failed on %s: %s", self.table_name, type(exc).__name__)
            if strict:
                if isinstance(exc, TimeoutError):
                    raise OperationTimeoutError(
                        f"Hybrid search timed out on {self.table_name} ({type(exc).__name__})"
                    ) from exc
                raise BackendUnavailableError(
                    f"Hybrid search failed on {self.table_name} ({type(exc).__name__})"
                ) from exc
            return []

    async def vector_search(  # noqa: PLR0913, PLR0917
        self,
        embedding: list[float],
        n_results: int,
        threshold: float,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Standalone dense vector search (the hybrid search's vector stage).

        One round-trip: a single ``SELECT`` ordered by the store's
        distance expression (``embedding <-> :embedding`` for ``l2``,
        ``embedding <=> :embedding`` for ``cosine``) with the
        threshold and equality filters applied as bound
        parameters.

        Args:
            embedding: Query embedding vector (required; the
                method does not auto-embed).
            n_results: Maximum number of results to return.
            threshold: Maximum distance under the store's metric
                (L2 or cosine) for vector search; metric-relative,
                never comparable across metrics.
            text_field: Name of the column containing the
                document text.
            metadata_fields: Additional column names to
                include in the result metadata dict.
            session: Optional external session. When
                supplied, the caller owns the transaction
                (commit/rollback). When omitted, the store
                opens its own session.
            **filters: Additional filters (e.g., tenant_id).

        Returns:
            list[dict]: Result dicts ordered by distance
            ascending (tie-broken by id), each carrying
            ``id``, ``document``, ``metadata``,
            ``distance``, and ``keyword_score`` (``None``).

        Raises:
            ValueError: If identifiers are unsafe,
                ``n_results`` is not positive, or the
                embedding has the wrong dimension.
            Exception: Re-raises database errors.
        """
        validate_search_identifiers(text_field, filters, metadata_fields)
        metadata_fields = metadata_fields or []
        if n_results < 1:
            raise ValueError(f"n_results must be a positive integer; got {n_results}")

        validate_embedding(embedding, self.embedding_dim, label="query embedding")

        # The distance operator comes from the internal metric map only,
        # never from user input, so the SQL stays injection-safe.
        distance_expr = f"embedding {self._distance_operator} cast(:embedding as vector)"

        select_columns = [self.id_field, text_field, *metadata_fields]
        select_sql = ", ".join(select_columns)

        filter_parts = []
        params: dict[str, Any] = {
            "embedding": f"[{','.join(str(x) for x in embedding)}]",
            "distance_threshold": threshold,
            "n_results": n_results,
        }
        for key, value in filters.items():
            param_name = f"filter_{key}"
            filter_parts.append(f"{key} = :{param_name}")
            params[param_name] = value

        where_clause = f"WHERE {distance_expr} < :distance_threshold"
        if filter_parts:
            where_clause += " AND " + " AND ".join(filter_parts)

        stmt = f"""
            SELECT {select_sql},
                   {distance_expr} AS distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY {distance_expr}, {self.id_field}
            LIMIT :n_results
        """

        return await self._execute_search_query(
            session=session,
            stmt=stmt,
            params=params,
            text_field=text_field,
            metadata_fields=metadata_fields,
            distance_field="distance",
        )

    async def keyword_search(  # noqa: PLR0913, PLR0917
        self,
        query: str,
        n_results: int,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
        session: Any = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Standalone keyword search (the hybrid search's keyword stage).

        One round-trip: a single ``SELECT`` using PostgreSQL
        full-text search (``tsvector @@ plainto_tsquery``)
        ordered by ``ts_rank`` descending, with equality
        filters applied as bound parameters.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            text_field: Column storing the text matching plainto_tsquery.
            metadata_fields: Extra metadata column names for the output dict.
            session: Optional caller transaction session; otherwise a new
                session is acquired for the duration of this read.
            **filters: Bound equality filters for row selection.

        Returns:
            list[dict]: Result dicts ordered by
            ``ts_rank`` descending (tie-broken by id),
            each carrying ``id``, ``document``,
            ``metadata``, ``distance`` (``None``), and
            ``keyword_score``.

        Raises:
            ValueError: If identifiers are unsafe or
                ``n_results`` is not positive.
            Exception: Re-raises database errors.
        """
        validate_search_identifiers(text_field, filters, metadata_fields)
        metadata_fields = metadata_fields or []
        if n_results < 1:
            raise ValueError(f"n_results must be a positive integer; got {n_results}")

        select_columns = [self.id_field, text_field, *metadata_fields]
        select_sql = ", ".join(select_columns)
        tsquery_expr = "plainto_tsquery(cast(:tsquery_config as regconfig), :query)"

        filter_parts = []
        params: dict[str, Any] = {
            "query": query,
            "tsquery_config": self.tsquery_config,
            "n_results": n_results,
        }
        for key, value in filters.items():
            param_name = f"filter_{key}"
            filter_parts.append(f"{key} = :{param_name}")
            params[param_name] = value

        where_clause = f"tsvector @@ {tsquery_expr}"
        if filter_parts:
            where_clause += " AND " + " AND ".join(filter_parts)

        stmt = f"""
            SELECT {select_sql},
                   ts_rank(tsvector, {tsquery_expr}) AS score
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY ts_rank(tsvector, {tsquery_expr}) DESC, {self.id_field}
            LIMIT :n_results
        """

        return await self._execute_search_query(
            session=session,
            stmt=stmt,
            params=params,
            text_field=text_field,
            metadata_fields=metadata_fields,
            keyword_field="score",
        )

    async def get_embeddings(self, ids: list[str], session: Any = None) -> dict[str, list[float]]:
        if not ids:
            return {}
        import json  # noqa: PLC0415

        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        stmt = (
            f"SELECT {self.id_field}, embedding FROM {self.table_name} "
            f"WHERE {self.id_field} IN ({placeholders})"
        )
        params = {f"id_{i}": doc_id for i, doc_id in enumerate(ids)}
        if session is None:
            async with self._resolve_session_factory()() as owned:
                rows = await self._sql.fetch_all(owned, stmt, params)
        else:
            rows = await self._sql.fetch_all(session, stmt, params)
        return {
            str(row[self.id_field]): (
                json.loads(row["embedding"])
                if isinstance(row["embedding"], str)
                else list(row["embedding"])
            )
            for row in rows
        }

    async def delete(
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
    ) -> int:
        """Delete documents selected by ``ids`` and/or ``filters``.

        At least one selector must be supplied; an empty
        ``ids`` list is rejected. When both are supplied,
        both restrictions apply (conjunction). The deletion
        count is reported by the statement itself
        (``RETURNING`` over the deleted rows), so the
        returned count is the number of rows the database
        deleted.

        Args:
            ids: Document ids to delete.
            filters: Equality filters (e.g., tenant_id).
            session: Optional external session. When
                supplied, the caller owns the transaction
                (commit/rollback). When omitted, the store
                opens its own session and commits.

        Returns:
            int: Number of deleted documents.

        Raises:
            ValueError: If no selector is supplied, ``ids``
                is empty, or a filter key is unsafe.
            Exception: Re-raises database errors.
        """
        validate_delete_selectors(ids, filters)

        conditions = []
        params: dict[str, Any] = {}
        if ids is not None:
            conditions.append(f"{self.id_field} = ANY(:ids)")
            params["ids"] = list(ids)
        for key, value in (filters or {}).items():
            param_name = f"filter_{key}"
            conditions.append(f"{key} = :{param_name}")
            params[param_name] = value

        stmt = f"""
            WITH deleted AS (
                DELETE FROM {self.table_name}
                WHERE {" AND ".join(conditions)}
                RETURNING 1
            )
            SELECT count(*) FROM deleted
        """

        owns_session = session is None
        if owns_session:
            session_factory = self._resolve_session_factory()
            async with session_factory() as owned_session:
                count = await self._sql.fetch_scalar(owned_session, stmt, params)
                await owned_session.commit()
        else:
            count = await self._sql.fetch_scalar(session, stmt, params)
        return int(count) if count is not None else 0

    async def get(  # noqa: PLR0913, PLR0917
        self,
        ids: list[Any] | None = None,
        filters: dict[str, Any] | None = None,
        session: Any = None,
        text_field: str = "text",
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents selected by ``ids`` and/or ``filters``.

        The selection is a single ``SELECT`` by primary key
        (``{id_field} = ANY(:ids)``) and/or equality
        filters. With no selector, every document is
        returned.

        Args:
            ids: Document ids to fetch.
            filters: Equality filters (e.g., tenant_id).
            session: Optional external session. When
                supplied, the caller owns the transaction
                (commit/rollback). When omitted, the store
                opens its own session.
            text_field: Name of the column containing the
                document text.
            metadata_fields: Additional column names to
                include in the result metadata dict.

        Returns:
            list[dict]: ``id``/``document``/``metadata``
            dicts (score keys are meaningless for direct
            fetches).

        Raises:
            ValueError: If any identifier is unsafe.
            Exception: Re-raises database errors.
        """
        validate_search_identifiers(text_field, filters, metadata_fields)
        metadata_fields = metadata_fields or []

        select_columns = [self.id_field, text_field, *metadata_fields]
        select_sql = ", ".join(select_columns)

        conditions = []
        params: dict[str, Any] = {}
        if ids is not None:
            conditions.append(f"{self.id_field} = ANY(:ids)")
            params["ids"] = list(ids)
        for key, value in (filters or {}).items():
            param_name = f"filter_{key}"
            conditions.append(f"{key} = :{param_name}")
            params[param_name] = value

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        stmt = f"""
            SELECT {select_sql}
            FROM {self.table_name}
            {where_clause}
        """

        owns_session = session is None
        if owns_session:
            session_factory = self._resolve_session_factory()
            async with session_factory() as owned_session:
                rows = await self._sql.fetch_all(owned_session, stmt, params)
        else:
            rows = await self._sql.fetch_all(session, stmt, params)

        return [
            {
                "id": str(row[self.id_field]),
                "document": row[text_field],
                "metadata": project_metadata(row, metadata_fields),
            }
            for row in rows
        ]

    async def count(
        self,
        session: Any = None,
        strict: bool = False,
        **filters: Any,
    ) -> int:
        """Count documents in the store.

        Args:
            session: Optional external SQLAlchemy async session. If provided,
                this session is reused instead of creating a new one. When an
                external session is supplied, the caller is responsible for
                committing or rolling back the transaction.
            strict: When ``True``, a failed count raises
                :class:`ragnor.core.types.RetrievalError`. The default ``False``
                preserves the historical behavior of logging the error and
                returning ``0``.
            **filters: Optional equality filters (e.g., ``tenant_id="..."``).
                Keys must be valid SQL identifiers; values are bound as
                parameters to prevent SQL injection.

        Returns:
            int: Number of matching documents, or 0 on error.

        Raises:
            ragnor.core.types.RetrievalError: If the count fails and
                ``strict=True``.
        """
        for key in filters:
            validate_identifier(key, "filter key")

        params: dict[str, Any] = {}
        filter_parts: list[str] = []
        for key, value in filters.items():
            param_name = f"filter_{key}"
            filter_parts.append(f"{key} = :{param_name}")
            params[param_name] = value

        where_clause = f"WHERE {' AND '.join(filter_parts)}" if filter_parts else ""

        stmt = f"""
            SELECT COUNT(*) FROM {self.table_name}
            {where_clause}
        """

        try:
            if session is not None:
                count = await self._sql.fetch_scalar(session, stmt, params)
            else:
                session_factory = self._resolve_session_factory()
                async with session_factory() as owned_session:
                    count = await self._sql.fetch_scalar(owned_session, stmt, params)
            return int(count) if count else 0
        except Exception as exc:
            logger.error("Count failed on %s: %s", self.table_name, type(exc).__name__)
            if strict:
                if isinstance(exc, TimeoutError):
                    raise OperationTimeoutError(
                        f"Count timed out on {self.table_name} ({type(exc).__name__})"
                    ) from exc
                raise BackendUnavailableError(
                    f"Count failed on {self.table_name} ({type(exc).__name__})"
                ) from exc
            return 0

    @staticmethod
    def _collect_metadata_keys(documents: list[dict[str, Any]]) -> list[str]:
        """Return every metadata key used by ``documents``, in first-seen order.

        Each key is validated as a safe SQL identifier while collecting, so
        the caller can interpolate the keys into the INSERT column list.
        """
        all_meta_keys: list[str] = []
        seen_meta_keys: set[str] = set()
        for doc in documents:
            for key in doc.get("metadata", {}):
                validate_identifier(key, "metadata key")
                if key not in seen_meta_keys:
                    seen_meta_keys.add(key)
                    all_meta_keys.append(key)
        return all_meta_keys

    @staticmethod
    def _build_values_clauses(  # noqa: PLR0913, PLR0917
        documents: list[dict[str, Any]],
        id_field: str,
        text_field: str,
        embedding_field: str,
        all_meta_keys: list[str],
        extra_fields: dict[str, Any] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Build one VALUES row per document and its bind parameters.

        Returns the list of ``(placeholder, ...)`` clauses and the flat
        ``params`` mapping the clauses refer to. Parameter names are
        namespaced by row index so rows never collide.
        """
        values_clauses: list[str] = []
        params: dict[str, Any] = {}

        for idx, doc in enumerate(documents):
            row_placeholders = []
            for key in [id_field, text_field, embedding_field]:
                # Column parameters use a ``_col_`` prefix so a metadata
                # (or extra) key equal to a column name can never shadow
                # the column's bind value.
                param_name = f"_col_{key}__{idx}"
                if key == id_field:
                    params[param_name] = doc.get("id")
                elif key == embedding_field:
                    params[param_name] = f"[{','.join(str(x) for x in doc[key])}]"
                else:
                    params[param_name] = doc[key]
                row_placeholders.append(f":{param_name}")
            for meta_key in all_meta_keys:
                param_name = f"{meta_key}__{idx}"
                params[param_name] = doc.get("metadata", {}).get(meta_key)
                row_placeholders.append(f":{param_name}")
            if extra_fields:
                for extra_key, extra_val in extra_fields.items():
                    param_name = f"_extra_{extra_key}__{idx}"
                    params[param_name] = extra_val
                    row_placeholders.append(f":{param_name}")
            values_clauses.append(f"({', '.join(row_placeholders)})")

        return values_clauses, params

    async def upsert(  # noqa: PLR0913, PLR0917
        self,
        documents: list[dict[str, Any]],
        session: Any = None,
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Upsert documents into the store.

        Inserts or updates documents using PostgreSQL's ``ON CONFLICT ... DO UPDATE``
        syntax. This allows the ingestion pipeline to re-run without creating
        duplicate rows.

        Args:
            documents: List of document dicts with keys: id, text, metadata,
                embedding. Metadata keys are included as additional columns.
            session: Optional external SQLAlchemy async session. If provided,
                this session is reused instead of creating a new one. The caller
                is responsible for committing the transaction after this call.
            text_field: Name of the text column. Defaults to "text".
            embedding_field: Name of the embedding column. Defaults to "embedding".
            id_field: Name of the primary key column. Defaults to the store's
                configured ``id_field``.
            extra_fields: Additional static fields to include in every INSERT
                (e.g., ``{"tenant_id": "..."}``).

        Raises:
            ValueError: If any column name contains unsafe characters.
            Exception: Re-raises database errors after logging.
        """
        id_field = validate_upsert_identifiers(
            text_field, embedding_field, id_field, self.id_field, extra_fields
        )

        if not documents:
            return

        owns_session = session is None
        try:
            for doc in documents:
                validate_embedding(
                    doc["embedding"],
                    self.embedding_dim,
                    label="document embedding",
                )

            all_meta_keys = self._collect_metadata_keys(documents)

            columns = [id_field, text_field, embedding_field]
            if extra_fields:
                columns.extend(extra_fields.keys())
            columns.extend(all_meta_keys)

            for col in columns:
                validate_identifier(col, "upsert column")

            non_id_columns = [c for c in columns if c != id_field]
            update_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_id_columns)

            values_clauses, params = self._build_values_clauses(  # noqa: PLR0913, PLR0917
                documents,
                id_field,
                text_field,
                embedding_field,
                all_meta_keys,
                extra_fields,
            )

            values_sql = ",\n    ".join(values_clauses)
            columns_str = ", ".join(columns)

            stmt = f"""
                INSERT INTO {self.table_name} ({columns_str})
                VALUES
                    {values_sql}
                ON CONFLICT ({id_field}) DO UPDATE SET {update_str}
            """

            if owns_session:
                session_factory = self._resolve_session_factory()
                async with session_factory() as owned_session:
                    await self._sql.execute(owned_session, stmt, params)
                    await owned_session.commit()
            else:
                await self._sql.execute(session, stmt, params)
            logger.info("Upserted %d documents to %s.", len(documents), self.table_name)
        except Exception as exc:
            logger.error("Upsert failed on %s: %s", self.table_name, type(exc).__name__)
            raise
