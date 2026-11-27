# ruff: noqa: PLR0913, PLR0917
"""Opt-in page-image retrieval fallback for pages text extraction cannot represent (EX-10).

Engineering drawings, multi-panel figures, complicated forms and scanned pages
can be routed to a visual (multi-vector, ColPali style) index instead of forcing
repeated VLM repair. Everything here is provider neutral and dependency free:
no model weights, no vector database client and no image library are imported.
Hosts opt in by passing a :class:`VisualEmbeddingProvider` and a
:class:`VisualIndex`; the optional Qdrant adapter lives in
``ragnor.stores.qdrant_visual``.

Design rules enforced by the code:

* A page keeps its text representation. A page may also have a visual one;
  :func:`fuse_results` returns one result per (tenant, document, page).
* Evidence returned to callers is a *page image location plus an authorized
  source reference*. It never contains invented text quotes. Optional VLM output
  is wrapped by :func:`derive_interpretation` and always flagged ``derived``.
* Image delivery re-checks authorization and revocation on every request, and
  private artifact references may not be public URLs.
* Index/model compatibility is checked on every write and query, so patch
  embeddings from a different model id, revision or dimension are never reused.
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

Vector = Sequence[float]
MultiVector = Sequence[Vector]

_BAD_REF_PREFIXES = ("http://", "https://", "data:", "ftp://", "file:", "//")


class VisualRetrievalError(RuntimeError):
    """Content-free failure with a fixed ``code``; provider text is never included."""

    codes = frozenset(
        {
            "model_mismatch",
            "invalid_vectors",
            "invalid_image",
            "image_too_large",
            "unauthorized",
            "revoked",
            "remote_not_allowed",
            "provider_failed",
            "index_failed",
            "cancelled",
            "resource_limit",
            "unknown_generation",
            "invalid_page",
        }
    )

    def __init__(self, code: str):
        self.code = code if code in self.codes else "provider_failed"
        super().__init__(f"Visual retrieval failed: {self.code}")


# ----------------------------------------------------------------------- models


@dataclass(frozen=True)
class VisualModel:
    """Explicit visual embedding identity; ids and revisions are mandatory."""

    model_id: str
    revision: str
    dim: int

    def __post_init__(self) -> None:
        for value in (self.model_id, self.revision):
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError("Visual models need an explicit model id and revision")
        if type(self.dim) is not int or not 1 <= self.dim <= 4096:
            raise ValueError("Visual model dimension must be between 1 and 4096")

    @property
    def slug(self) -> str:
        return hashlib.sha256(f"{self.model_id}\0{self.revision}\0{self.dim}".encode()).hexdigest()[
            :16
        ]


def validate_multivector(vectors: MultiVector, model: VisualModel, max_patches: int = 4096) -> None:
    """Reject wrong shapes, non-finite values and zero vectors (fail closed)."""
    if not 1 <= len(vectors) <= max_patches:
        raise VisualRetrievalError("invalid_vectors")
    for vec in vectors:
        if len(vec) != model.dim:
            raise VisualRetrievalError("invalid_vectors")
        norm = 0.0
        for x in vec:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise VisualRetrievalError("invalid_vectors")
            norm += x * x
        if norm == 0.0:
            raise VisualRetrievalError("invalid_vectors")


def normalize(vectors: MultiVector) -> list[list[float]]:
    out = []
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        out.append([x / norm for x in vec])
    return out


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    document_id: str
    revision: str


@dataclass(frozen=True)
class VisualPage:
    """Immutable reference to a private page image and its provenance."""

    tenant: str
    page_id: str
    source: SourceReference
    page: int
    artifact_ref: str
    width: int
    height: int
    content_sha256: str
    classification: str = "needs_review"
    reason: str = ""

    def __post_init__(self) -> None:
        for key in ("tenant", "page_id", "artifact_ref", "classification"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or len(value) > 500:
                raise ValueError(f"VisualPage.{key} must be a non-empty string")
        if self.artifact_ref.strip().lower().startswith(_BAD_REF_PREFIXES):
            raise ValueError("Visual artifacts must be private references, not public URLs")
        if type(self.page) is not int or self.page < 1:
            raise ValueError("Page number must be a positive integer")
        if any(type(v) is not int or v < 1 for v in (self.width, self.height)):
            raise ValueError("Page dimensions must be positive integers")
        if len(self.content_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.content_sha256
        ):
            raise ValueError("content_sha256 must be a SHA-256 hex digest")

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.tenant, self.source.document_id, self.page)

    def to_pixels(self, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Convert a normalized box into pixel coordinates for this immutable image."""
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Boxes must be normalized to the page")
        return (
            round(x0 * self.width),
            round(y0 * self.height),
            round(x1 * self.width),
            round(y1 * self.height),
        )

    def to_normalized(self, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("Pixel box is outside the page image")
        return (x0 / self.width, y0 / self.height, x1 / self.width, y1 / self.height)


# ---------------------------------------------------------------------- images


@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int = 20_000_000
    max_pixels: int = 50_000_000
    max_side: int = 20_000


def image_dimensions(data: bytes, limits: ImageLimits = ImageLimits()) -> tuple[int, int]:
    """Read PNG/JPEG dimensions from the header without decoding pixels."""
    if len(data) > limits.max_bytes:
        raise VisualRetrievalError("image_too_large")
    width = height = 0
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
    elif data[:2] == b"\xff\xd8":
        width, height = _jpeg_size(data)
    if width < 1 or height < 1:
        raise VisualRetrievalError("invalid_image")
    if width > limits.max_side or height > limits.max_side or width * height > limits.max_pixels:
        raise VisualRetrievalError("image_too_large")
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int]:
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            return 0, 0
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if length < 2:
            return 0, 0
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if pos + 9 > len(data):
                return 0, 0
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return width, height
        pos += 2 + length
    return 0, 0


# ------------------------------------------------------------------- providers


class VisualEmbeddingProvider(Protocol):
    """Multi-vector page/query embeddings (e.g. ColPali/ColQwen adapters).

    ``remote`` providers send page images off-host and need explicit opt-in.
    """

    model: VisualModel
    remote: bool

    def embed_pages(self, images: Sequence[bytes]) -> list[MultiVector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> MultiVector:
        raise NotImplementedError


@dataclass(frozen=True)
class VisualHit:
    page_id: str
    score: float
    page: VisualPage | None = None


class VisualIndex(Protocol):
    """Generation-staged multi-vector index scoped by tenant.

    Staged pages are invisible until :meth:`commit`; :meth:`rollback` discards
    them. Every call carries the model so incompatible embeddings fail closed.
    """

    model: VisualModel

    def begin(self, tenant: str, model: VisualModel) -> str:
        raise NotImplementedError

    def stage(
        self, generation: str, page: VisualPage, vectors: MultiVector, model: VisualModel
    ) -> None:
        raise NotImplementedError

    def commit(self, generation: str) -> None:
        raise NotImplementedError

    def rollback(self, generation: str) -> None:
        raise NotImplementedError

    def revoke_document(self, tenant: str, document_id: str) -> int:
        raise NotImplementedError

    def search(
        self,
        tenant: str,
        query: MultiVector,
        model: VisualModel,
        limit: int,
        document_ids: Iterable[str] | None = None,
    ) -> list[VisualHit]:
        raise NotImplementedError


def _check_model(index_model: VisualModel, model: VisualModel) -> None:
    if model != index_model:
        raise VisualRetrievalError("model_mismatch")


def maxsim(query: MultiVector, page: MultiVector) -> float:
    """Late-interaction score: sum over query patches of the best cosine match."""
    return sum(max(sum(q * p for q, p in zip(qv, pv, strict=True)) for pv in page) for qv in query)


class InMemoryVisualIndex:
    """Reference index with staging, commit, rollback and revocation."""

    def __init__(self, model: VisualModel) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._seq = 0
        self._staged: dict[str, dict[str, Any]] = {}
        self._live: dict[tuple[str, str], tuple[VisualPage, list[list[float]], int]] = {}
        self._revoked: set[tuple[str, str]] = set()

    def begin(self, tenant: str, model: VisualModel) -> str:
        _check_model(self.model, model)
        with self._lock:
            self._seq += 1
            gen = f"{tenant}:{self._seq}"
            self._staged[gen] = {"tenant": tenant, "pages": {}, "n": self._seq}
            return gen

    def stage(
        self, generation: str, page: VisualPage, vectors: MultiVector, model: VisualModel
    ) -> None:
        _check_model(self.model, model)
        validate_multivector(vectors, model)
        with self._lock:
            entry = self._staged.get(generation)
            if entry is None:
                raise VisualRetrievalError("unknown_generation")
            if page.tenant != entry["tenant"]:
                raise VisualRetrievalError("unauthorized")
            entry["pages"][page.page_id] = (page, normalize(vectors))

    def commit(self, generation: str) -> None:
        with self._lock:
            entry = self._staged.pop(generation, None)
            if entry is None:
                raise VisualRetrievalError("unknown_generation")
            for page_id, (page, vecs) in entry["pages"].items():
                self._live[(entry["tenant"], page_id)] = (page, vecs, entry["n"])
                self._revoked.discard((entry["tenant"], page.source.document_id))

    def rollback(self, generation: str) -> None:
        with self._lock:
            self._staged.pop(generation, None)

    def revoke_document(self, tenant: str, document_id: str) -> int:
        with self._lock:
            self._revoked.add((tenant, document_id))
            return sum(
                1
                for (t, _), (p, _, _) in self._live.items()
                if t == tenant and p.source.document_id == document_id
            )

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
        allowed = set(document_ids) if document_ids is not None else None
        q = normalize(query)
        with self._lock:
            rows = [
                (page, vecs)
                for (t, _), (page, vecs, _) in self._live.items()
                if t == tenant
                and (tenant, page.source.document_id) not in self._revoked
                and (allowed is None or page.source.document_id in allowed)
            ]
        hits = [VisualHit(p.page_id, maxsim(q, v), p) for p, v in rows]
        hits.sort(key=lambda h: (-h.score, h.page_id))
        return hits[: max(limit, 0)]

    def vector_count(self) -> int:
        with self._lock:
            return sum(len(v) for _, v, _ in self._live.values())


# ---------------------------------------------------------------------- policy


@dataclass(frozen=True)
class RoutingDecision:
    visual: bool
    keep_text: bool
    reason: str


@dataclass(frozen=True)
class VisualRoutingPolicy:
    """Route designated families and unstructurable/needs-review pages to images."""

    families: frozenset[str] = frozenset()
    statuses: frozenset[str] = frozenset({"needs_ocr", "partial", "failed"})

    def route(self, family: str | None, status: str, has_text: bool) -> RoutingDecision:
        if family is not None and family in self.families:
            return RoutingDecision(True, has_text, "designated_family")
        if status in self.statuses:
            return RoutingDecision(True, has_text, "unstructurable")
        return RoutingDecision(False, True, "text_representation_sufficient")


# -------------------------------------------------------------------- indexing


class CancelFlag(Protocol):
    @property
    def cancelled(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class IndexReport:
    generation: str
    pages_indexed: int
    patches: int


class VisualIndexer:
    """Embeds page images and publishes them atomically (stage -> commit).

    Any failure, cancellation or resource error rolls the generation back, so a
    partially written snapshot is never visible. Re-running the same input is a
    replay: pages are upserted by ``page_id`` inside a fresh generation.
    """

    def __init__(
        self,
        provider: VisualEmbeddingProvider,
        index: VisualIndex,
        *,
        allow_remote: bool = False,
        limits: ImageLimits = ImageLimits(),
        batch_size: int = 8,
        max_patches: int = 4096,
    ) -> None:
        if provider.remote and not allow_remote:
            raise VisualRetrievalError("remote_not_allowed")
        _check_model(index.model, provider.model)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._provider, self._index = provider, index
        self._allow_remote, self._limits = allow_remote, limits
        self._batch, self._max_patches = batch_size, max_patches

    def needs_reindex(self, index_model: VisualModel) -> bool:
        """True when stored embeddings came from a different model id/revision/dim."""
        return index_model != self._provider.model

    def index(
        self,
        tenant: str,
        pages: Sequence[tuple[VisualPage, bytes]],
        cancel: CancelFlag | None = None,
    ) -> IndexReport:
        model = self._provider.model
        for page, data in pages:
            if page.tenant != tenant:
                raise VisualRetrievalError("unauthorized")
            if hashlib.sha256(data).hexdigest() != page.content_sha256:
                raise VisualRetrievalError("invalid_image")
            width, height = image_dimensions(data, self._limits)
            if (width, height) != (page.width, page.height):
                raise VisualRetrievalError("invalid_image")
        generation = self._index.begin(tenant, model)
        patches = 0
        try:
            for start in range(0, len(pages), self._batch):
                if cancel is not None and cancel.cancelled:
                    raise VisualRetrievalError("cancelled")
                batch = pages[start : start + self._batch]
                try:
                    vectors = self._provider.embed_pages([d for _, d in batch])
                except MemoryError as exc:
                    raise VisualRetrievalError("resource_limit") from exc
                except VisualRetrievalError:
                    raise
                except Exception as exc:
                    raise VisualRetrievalError("provider_failed") from exc
                if len(vectors) != len(batch):
                    raise VisualRetrievalError("invalid_vectors")
                for (page, _), vecs in zip(batch, vectors, strict=True):
                    validate_multivector(vecs, model, self._max_patches)
                    self._index.stage(generation, page, vecs, model)
                    patches += len(vecs)
            if cancel is not None and cancel.cancelled:
                raise VisualRetrievalError("cancelled")
            self._index.commit(generation)
        except BaseException:
            self._index.rollback(generation)
            raise
        return IndexReport(generation, len(pages), patches)


# ---------------------------------------------------------------- authorization


class PageAuthorizer(Protocol):
    """Host policy: may this principal see this page? Called on every delivery."""

    def is_authorized(self, tenant: str, principal: str, page: VisualPage) -> bool:
        raise NotImplementedError


class ArtifactStore(Protocol):
    def read(self, artifact_ref: str) -> bytes:
        raise NotImplementedError


@dataclass(frozen=True)
class VisualEvidence:
    """Evidence record: page-image location and source reference, never quotes."""

    tenant: str
    source: SourceReference
    page: int
    page_id: str
    artifact_ref: str
    score: float
    representations: tuple[str, ...] = ("visual",)
    derived: bool = False
    text_hit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "visual_page",
            "source_id": self.source.source_id,
            "document_id": self.source.document_id,
            "revision": self.source.revision,
            "page": self.page,
            "page_id": self.page_id,
            "artifact_ref": self.artifact_ref,
            "score": self.score,
            "representations": list(self.representations),
            "derived": self.derived,
            "text_hit_id": self.text_hit_id,
        }


class VisualRetriever:
    """Query -> authorized page evidence, and guarded image delivery."""

    def __init__(
        self,
        provider: VisualEmbeddingProvider,
        index: VisualIndex,
        authorizer: PageAuthorizer,
        *,
        allow_remote: bool = False,
        artifacts: ArtifactStore | None = None,
        limits: ImageLimits = ImageLimits(),
    ) -> None:
        if provider.remote and not allow_remote:
            raise VisualRetrievalError("remote_not_allowed")
        _check_model(index.model, provider.model)
        self._provider, self._index, self._auth = provider, index, authorizer
        self._artifacts, self._limits = artifacts, limits

    def _allowed(self, tenant: str, principal: str, page: VisualPage) -> bool:
        if page.tenant != tenant:
            return False
        try:
            return bool(self._auth.is_authorized(tenant, principal, page))
        except Exception:
            return False  # authorization errors fail closed

    def search(
        self, tenant: str, principal: str, query: str, limit: int = 5
    ) -> list[VisualEvidence]:
        try:
            qvec = self._provider.embed_query(query)
        except VisualRetrievalError:
            raise
        except Exception as exc:
            raise VisualRetrievalError("provider_failed") from exc
        hits = self._index.search(tenant, qvec, self._provider.model, limit * 4 + 8)
        out = []
        for hit in hits:
            page = hit.page
            if page is None or not self._allowed(tenant, principal, page):
                continue
            out.append(
                VisualEvidence(
                    tenant, page.source, page.page, page.page_id, page.artifact_ref, hit.score
                )
            )
            if len(out) == limit:
                break
        return out

    def fetch_image(self, tenant: str, principal: str, page: VisualPage) -> bytes:
        """Return image bytes only after authorization, size and format checks."""
        if self._artifacts is None:
            raise VisualRetrievalError("index_failed")
        if not self._allowed(tenant, principal, page):
            raise VisualRetrievalError("unauthorized")
        data = self._artifacts.read(page.artifact_ref)
        image_dimensions(data, self._limits)
        if hashlib.sha256(data).hexdigest() != page.content_sha256:
            raise VisualRetrievalError("invalid_image")
        return data


# ----------------------------------------------------------------------- fusion


class ScoreCalibrator:
    """Empirical-CDF calibration so visual MaxSim and text scores are comparable.

    Fit on a sample of scores from each retriever (for example the benchmark
    queries); a raw score maps to the fraction of sample scores at or below it.
    """

    def __init__(self, sample: Iterable[float]) -> None:
        values = sorted(float(s) for s in sample if math.isfinite(s))
        if not values:
            raise ValueError("Calibration needs at least one finite score")
        self._values = values

    def __call__(self, score: float) -> float:
        lo, hi = 0, len(self._values)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._values[mid] <= score:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(self._values)


@dataclass(frozen=True)
class TextPageHit:
    hit_id: str
    tenant: str
    source: SourceReference
    page: int
    score: float


@dataclass(frozen=True)
class FusedResult:
    evidence: VisualEvidence | None
    text_hit_id: str | None
    source: SourceReference
    page: int
    score: float
    representations: tuple[str, ...]


def fuse_results(
    visual: Sequence[VisualEvidence],
    text: Sequence[TextPageHit],
    *,
    visual_calibrator: ScoreCalibrator,
    text_calibrator: ScoreCalibrator,
    current_revisions: Mapping[str, str] | None = None,
    limit: int = 10,
) -> list[FusedResult]:
    """Merge text and visual hits per page, dropping stale revisions.

    A page present in both representations yields ONE result (best calibrated
    score) that keeps the visual evidence and the text hit id. When
    ``current_revisions`` is given, hits for superseded document revisions are
    discarded so stale references never surface.
    """

    def fresh(source: SourceReference) -> bool:
        return (
            current_revisions is None
            or current_revisions.get(source.document_id, source.revision) == source.revision
        )

    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for ev in visual:
        if not fresh(ev.source):
            continue
        key = (ev.tenant, ev.source.document_id, ev.page)
        slot = merged.setdefault(key, {"source": ev.source})
        score = visual_calibrator(ev.score)
        if "visual" not in slot or score > slot["vscore"]:
            slot["visual"], slot["vscore"] = ev, score
    for hit in text:
        if not fresh(hit.source):
            continue
        key = (hit.tenant, hit.source.document_id, hit.page)
        slot = merged.setdefault(key, {"source": hit.source})
        score = text_calibrator(hit.score)
        if "text" not in slot or score > slot["tscore"]:
            slot["text"], slot["tscore"] = hit, score
    results = []
    for (_, _, page), slot in merged.items():
        reps = tuple(r for r in ("text", "visual") if r in slot)
        score = max(slot.get("vscore", 0.0), slot.get("tscore", 0.0))
        ev = slot.get("visual")
        text_id = slot["text"].hit_id if "text" in slot else None
        if ev is not None:
            ev = VisualEvidence(
                ev.tenant,
                ev.source,
                ev.page,
                ev.page_id,
                ev.artifact_ref,
                ev.score,
                reps,
                text_hit_id=text_id,
            )
        results.append(FusedResult(ev, text_id, slot["source"], page, score, reps))
    results.sort(key=lambda r: (-r.score, r.source.document_id, r.page))
    return results[:limit]


def derive_interpretation(
    evidence: VisualEvidence,
    interpreter: Callable[[VisualEvidence], str],
) -> dict[str, Any]:
    """Wrap optional downstream VLM output so it is always flagged as derived."""
    try:
        text = interpreter(evidence)
    except Exception as exc:
        raise VisualRetrievalError("provider_failed") from exc
    return {
        "kind": "derived_interpretation",
        "derived": True,
        "text": str(text),
        "derived_from": evidence.to_dict(),
    }


__all__ = [
    "ArtifactStore",
    "FusedResult",
    "ImageLimits",
    "InMemoryVisualIndex",
    "IndexReport",
    "PageAuthorizer",
    "RoutingDecision",
    "ScoreCalibrator",
    "SourceReference",
    "TextPageHit",
    "VisualEmbeddingProvider",
    "VisualEvidence",
    "VisualHit",
    "VisualIndex",
    "VisualIndexer",
    "VisualModel",
    "VisualPage",
    "VisualRetrievalError",
    "VisualRetriever",
    "VisualRoutingPolicy",
    "derive_interpretation",
    "fuse_results",
    "image_dimensions",
    "maxsim",
    "normalize",
    "validate_multivector",
]
