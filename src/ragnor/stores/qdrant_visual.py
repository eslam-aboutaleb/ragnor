"""Qdrant multi-vector adapter for the visual page index (EX-10).

Uses Qdrant's native multi-vector support (``MultiVectorConfig`` with the
``MAX_SIM`` comparator, qdrant-client >= 1.10), which implements the same
late-interaction scoring as :func:`ragnor.retrieval.visual.maxsim`. pgvector has
no multi-vector MaxSim operator, so it is not claimed here: hosts on pgvector
should keep a Qdrant collection for visual pages or store one vector per patch
and aggregate in SQL (not provided).

One collection per visual model id/revision/dimension (``<prefix>_<slug>``), so
embeddings from different models can never share a collection. A sentinel point
records the model and is verified on every attach; the vector size and
comparator are verified too. Pages are staged with ``state="staged"`` and become
visible only when their generation is committed.
"""

from __future__ import annotations

import time
import uuid
import warnings
from collections.abc import Iterable
from typing import Any

from ragnor.retrieval.visual import (
    MultiVector,
    SourceReference,
    VisualHit,
    VisualModel,
    VisualPage,
    VisualRetrievalError,
    _check_model,
    normalize,
    validate_multivector,
)

_SENTINEL = uuid.uuid5(uuid.NAMESPACE_URL, "ragnor-visual-model-sentinel")


def _models():
    from qdrant_client import models  # noqa: PLC0415 - optional dependency

    return models


def _point_id(tenant: str, page_id: str, generation: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant}\0{page_id}\0{generation}"))


class QdrantVisualIndex:
    """:class:`ragnor.retrieval.visual.VisualIndex` backed by Qdrant."""

    def __init__(
        self,
        client: Any,
        model: VisualModel,
        *,
        collection_prefix: str = "ragnor_visual",
        on_disk: bool = False,
        quantization: str | None = None,
    ) -> None:
        if quantization not in (None, "int8"):
            raise ValueError("quantization must be None or 'int8'")
        self.model = model
        self._client = client
        self.collection = f"{collection_prefix}_{model.slug}"
        self._seq = 0
        self._ensure_collection(on_disk, quantization)

    # ---------------------------------------------------------------- schema

    def _ensure_collection(self, on_disk: bool, quantization: str | None) -> None:
        m = _models()
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                self.collection,
                vectors_config=m.VectorParams(
                    size=self.model.dim,
                    distance=m.Distance.COSINE,
                    on_disk=on_disk,
                    multivector_config=m.MultiVectorConfig(
                        comparator=m.MultiVectorComparator.MAX_SIM
                    ),
                ),
                quantization_config=(
                    m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8))
                    if quantization == "int8"
                    else None
                ),
            )
            self._client.upsert(
                self.collection,
                [
                    m.PointStruct(
                        id=str(_SENTINEL),
                        vector=[[1.0] + [0.0] * (self.model.dim - 1)],
                        payload={
                            "kind": "model",
                            "model_id": self.model.model_id,
                            "revision": self.model.revision,
                            "dim": self.model.dim,
                        },
                    )
                ],
            )
        self._verify_collection()
        for field in ("tenant", "state", "document_id", "page_id", "generation"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # local mode ignores payload indexes
                    self._client.create_payload_index(
                        self.collection, field, _models().PayloadSchemaType.KEYWORD
                    )
            except Exception:  # noqa: BLE001, S110 - already indexed
                pass

    def _verify_collection(self) -> None:
        info = self._client.get_collection(self.collection)
        params = info.config.params.vectors
        multi = getattr(params, "multivector_config", None)
        if (
            params is None
            or getattr(params, "size", None) != self.model.dim
            or multi is None
            or str(multi.comparator).lower().split(".")[-1] != "max_sim"
        ):
            raise VisualRetrievalError("model_mismatch")
        found = self._client.retrieve(self.collection, [str(_SENTINEL)], with_payload=True)
        payload = found[0].payload if found else None
        if not payload or (
            payload.get("model_id"),
            payload.get("revision"),
            payload.get("dim"),
        ) != (self.model.model_id, self.model.revision, self.model.dim):
            raise VisualRetrievalError("model_mismatch")

    # ------------------------------------------------------------- generations

    def _filter(self, *conditions: tuple[str, Any]) -> Any:
        m = _models()
        return m.Filter(
            must=[
                m.FieldCondition(key=key, match=m.MatchValue(value=value))
                for key, value in conditions
            ]
        )

    def begin(self, tenant: str, model: VisualModel) -> str:
        _check_model(self.model, model)
        self._seq += 1
        return f"{tenant}|{time.time_ns()}|{self._seq}"

    def stage(
        self, generation: str, page: VisualPage, vectors: MultiVector, model: VisualModel
    ) -> None:
        _check_model(self.model, model)
        validate_multivector(vectors, model)
        tenant = generation.split("|", 1)[0]
        if page.tenant != tenant:
            raise VisualRetrievalError("unauthorized")
        m = _models()
        try:
            self._client.upsert(
                self.collection,
                [
                    m.PointStruct(
                        id=_point_id(tenant, page.page_id, generation),
                        vector=normalize(vectors),
                        payload={
                            "kind": "page",
                            "state": "staged",
                            "generation": generation,
                            "tenant": tenant,
                            "page_id": page.page_id,
                            "source_id": page.source.source_id,
                            "document_id": page.source.document_id,
                            "revision": page.source.revision,
                            "page": page.page,
                            "artifact_ref": page.artifact_ref,
                            "width": page.width,
                            "height": page.height,
                            "content_sha256": page.content_sha256,
                            "classification": page.classification,
                            "reason": page.reason,
                        },
                    )
                ],
            )
        except Exception as exc:
            raise VisualRetrievalError("index_failed") from exc

    def commit(self, generation: str) -> None:
        tenant = generation.split("|", 1)[0]
        count = self._client.count(
            self.collection,
            count_filter=self._filter(("generation", generation), ("state", "staged")),
            exact=True,
        ).count
        if count == 0:
            # Empty or unknown generation: nothing to publish.
            return
        try:
            staged = self._filter(("generation", generation), ("state", "staged"))
            self._client.set_payload(self.collection, {"state": "live"}, points=staged)
            # Supersede earlier live copies of the same pages (replay / re-index).
            offset = None
            while True:
                points, offset = self._client.scroll(
                    self.collection,
                    scroll_filter=self._filter(("generation", generation)),
                    with_payload=["page_id"],
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    self._retire_older(tenant, point.payload["page_id"], generation)
                if offset is None:
                    break
        except Exception as exc:
            self.rollback(generation)
            raise VisualRetrievalError("index_failed") from exc

    def _retire_older(self, tenant: str, page_id: str, generation: str) -> None:
        m = _models()
        older = m.Filter(
            must=[
                m.FieldCondition(key="tenant", match=m.MatchValue(value=tenant)),
                m.FieldCondition(key="page_id", match=m.MatchValue(value=page_id)),
                m.FieldCondition(key="state", match=m.MatchValue(value="live")),
            ],
            must_not=[m.FieldCondition(key="generation", match=m.MatchValue(value=generation))],
        )
        self._client.delete(self.collection, points_selector=m.FilterSelector(filter=older))

    def rollback(self, generation: str) -> None:
        m = _models()
        self._client.delete(
            self.collection,
            points_selector=m.FilterSelector(
                filter=self._filter(("generation", generation), ("state", "staged"))
            ),
        )

    def revoke_document(self, tenant: str, document_id: str) -> int:
        m = _models()
        selector = self._filter(("tenant", tenant), ("document_id", document_id))
        count = self._client.count(self.collection, count_filter=selector, exact=True).count
        self._client.delete(self.collection, points_selector=m.FilterSelector(filter=selector))
        return count

    # ------------------------------------------------------------------ search

    def search(
        self,
        tenant: str,
        query: MultiVector,
        model: VisualModel,
        limit: int,
        document_ids: Iterable[str] | None = None,
    ) -> list[VisualHit]:
        _check_model(self.model, model)
        validate_multivector(query, model)
        m = _models()
        must = [
            m.FieldCondition(key="tenant", match=m.MatchValue(value=tenant)),
            m.FieldCondition(key="state", match=m.MatchValue(value="live")),
        ]
        if document_ids is not None:
            must.append(
                m.FieldCondition(key="document_id", match=m.MatchAny(any=list(document_ids)))
            )
        try:
            result = self._client.query_points(
                self.collection,
                query=normalize(query),
                query_filter=m.Filter(must=must),
                limit=max(limit, 1),
                with_payload=True,
            )
        except Exception as exc:
            raise VisualRetrievalError("index_failed") from exc
        hits = []
        seen: set[str] = set()
        for point in result.points:
            p = point.payload
            if p["page_id"] in seen:
                continue  # replay overlap: keep the best-scoring copy
            seen.add(p["page_id"])
            if len(hits) == limit:
                break
            page = VisualPage(
                tenant=p["tenant"],
                page_id=p["page_id"],
                source=SourceReference(p["source_id"], p["document_id"], p["revision"]),
                page=p["page"],
                artifact_ref=p["artifact_ref"],
                width=p["width"],
                height=p["height"],
                content_sha256=p["content_sha256"],
                classification=p["classification"],
                reason=p.get("reason", ""),
            )
            hits.append(VisualHit(p["page_id"], float(point.score), page))
        return hits
