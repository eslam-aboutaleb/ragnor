"""Per-page PDF profiler built on pypdf (``ragnor[pdf]``).

Implements the :class:`ragnor.extraction.PageProfiler` port for documents and
adds :meth:`PdfPageProfiler.profile_pages` for per-page facts. It reads only
the PDF structure and text-positioning operators; it renders nothing and
calls no engine.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from io import BytesIO
from typing import Any

from ragnor.extraction.routing import PageKind, PageProfile
from ragnor.extraction.types import DocumentProfile

_STATE_ONLY = {
    b"q", b"Q", b"cm", b"w", b"J", b"j", b"M", b"d", b"ri", b"i", b"gs", b"CS", b"cs",
    b"SC", b"SCN", b"sc", b"scn", b"G", b"g", b"RG", b"rg", b"K", b"k",
}  # fmt: skip
_TEXT_OPS = {
    b"BT", b"ET", b"Tc", b"Tw", b"Tz", b"TL", b"Tf", b"Tr", b"Ts", b"Td", b"TD", b"Tm",
    b"T*", b"Tj", b"TJ", b"'", b'"', b"BMC", b"BDC", b"EMC", b"MP", b"DP",
}  # fmt: skip


def _quadrant(a: float, b: float) -> int:
    angle = math.degrees(math.atan2(b, a)) % 360
    return int(round(angle / 90.0)) % 4 * 90


class PdfPageProfiler:
    """Classify each page as digital, scan, mixed, blank or unknown."""

    def __init__(self, *, max_pages: int = 1000, max_bytes: int = 50_000_000) -> None:
        self.max_pages = max_pages
        self.max_bytes = max_bytes

    def _reader(self, data: bytes) -> Any:
        if len(data) > self.max_bytes:
            raise ValueError("PDF exceeds max_bytes")
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PDF profiling requires pypdf; install 'ragnor[pdf]'") from exc
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF requires host decryption")
        if len(reader.pages) > self.max_pages:
            raise ValueError(f"PDF has {len(reader.pages)} pages, exceeding max_pages")
        return reader

    def profile_pages(self, data: bytes) -> list[PageProfile]:
        reader = self._reader(data)
        return [self._page(p, i) for i, p in enumerate(reader.pages, 1)]

    async def profile(self, data: bytes, *, name: str = "") -> DocumentProfile:
        pages = await asyncio.to_thread(self.profile_pages, data)
        text = [p for p in pages if p.char_count]
        return DocumentProfile(
            media_type="application/pdf",
            page_count=len(pages),
            has_text_layer=bool(text),
            has_images=any(p.has_images for p in pages),
            size_bytes=len(data),
            source_sha256=hashlib.sha256(data).hexdigest(),
        )

    def _page(self, page: Any, number: int) -> PageProfile:
        try:
            return self._profile(page, number)
        except Exception:
            return PageProfile(number, PageKind.UNKNOWN, reason="page_profile_failed")

    def _profile(self, page: Any, number: int) -> PageProfile:
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("bad page size")
        rotation = int(page.get("/Rotate", 0) or 0) % 360
        if rotation % 90:
            raise ValueError("bad rotation")
        votes: dict[int, int] = {}
        xs: list[float] = []

        def visit(text: str, cm: list[float], tm: list[float], _fd: Any, _fs: Any) -> None:
            if not text.strip():
                return
            a = tm[0] * cm[0] + tm[1] * cm[2]
            b = tm[0] * cm[1] + tm[1] * cm[3]
            q = _quadrant(a, b)
            votes[q] = votes.get(q, 0) + len(text.strip())
            if q == 0:
                xs.append(tm[4] * cm[0] + tm[5] * cm[2] + cm[4])

        text = (page.extract_text(visitor_text=visit) or "").strip()
        resources = page.get("/Resources", {})
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        xobjects = resources.get("/XObject", {})
        xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
        subtypes = [o.get_object().get("/Subtype") for o in xobjects.values()]
        images = "/Image" in subtypes
        forms = any(s != "/Image" for s in subtypes)
        annots = bool(page.get("/Annots"))
        contents = page.get_contents()
        ops = [op for _, op in (contents.operations if contents is not None else [])]
        drawing = any(
            op in {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"sh"} for op in ops
        )
        inline_image = b"INLINE IMAGE" in ops
        images = images or inline_image
        unknown = any(op not in _STATE_ONLY | _TEXT_OPS for op in ops)
        content = any(op not in _STATE_ONLY for op in ops)
        orientation = max(votes, key=lambda k: (votes[k], -k)) if votes else 0
        columns = _columns(xs, width)
        if text and not (images or forms or annots or unknown):
            kind, reason = PageKind.DIGITAL, "text_only"
        elif text:
            kind, reason = PageKind.MIXED, "text_with_visual_content"
        elif not (images or forms or content or annots):
            kind, reason = PageKind.BLANK, "no_content_operations"
        else:
            kind, reason = PageKind.SCAN, "visual_content_without_text"
        return PageProfile(
            number,
            kind,
            width,
            height,
            rotation,
            orientation,
            len(text),
            images,
            drawing or forms,
            annots,
            columns,
            reason,
        )


def _columns(xs: list[float], width: float) -> int:
    """Count distinct left-edge clusters; >1 hints at a multi-column page."""
    if len(xs) < 6:
        return 1
    gap = width * 0.12
    edges = sorted(xs)
    clusters: list[list[float]] = [[edges[0]]]
    for x in edges[1:]:
        if x - clusters[-1][-1] > gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    # A cluster must hold a real share of lines, else it is an indent or stray.
    real = [c for c in clusters if len(c) >= max(3, len(xs) // 6)]
    return max(1, len(real))
