# ruff: noqa: PLR0913, PLR0917
"""Deterministic pixel-derived fake visual embedder (no model weights).

It decodes real rasterized pages with Pillow and emits one patch vector per inked
grid cell. It validates plumbing, security and fusion; it says nothing about the
quality of a real ColPali/ColQwen model.
"""

from __future__ import annotations

import hashlib
import io

from ragnor.retrieval.visual import (
    SourceReference,
    VisualModel,
    VisualPage,
)

GRID = 4
BINS = 4
DIM = GRID * GRID + BINS
MODEL = VisualModel("fake-colpali", "r1", DIM)


def png(cells: dict[tuple[int, int], int], size: int = 32) -> bytes:
    """Raster 'drawing' with no text: grey blocks (0..3 darkness) in grid cells."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    step = size // GRID
    for (row, col), level in cells.items():
        shade = 255 - int((level + 1) * 255 / BINS) + 1
        draw.rectangle(
            [col * step, row * step, (col + 1) * step - 1, (row + 1) * step - 1], fill=max(shade, 0)
        )
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def _vec(cell: int, level: int) -> list[float]:
    v = [0.0] * DIM
    v[cell] = 1.0
    v[GRID * GRID + level] = 1.0
    return v


class FakeVisualProvider:
    """Patch vectors computed from real decoded pixels; query = 'c<cell>b<level>' tokens."""

    remote = False

    def __init__(self, model: VisualModel = MODEL, remote: bool = False, fail=None):
        self.model = model
        self.remote = remote
        self.fail = fail
        self.calls = 0

    def embed_pages(self, images):
        from PIL import Image  # noqa: PLC0415

        self.calls += 1
        if self.fail:
            raise self.fail
        out = []
        for data in images:
            image = Image.open(io.BytesIO(data)).convert("L")
            step = image.width // GRID
            patches = []
            for row in range(GRID):
                for col in range(GRID):
                    crop = image.crop((col * step, row * step, (col + 1) * step, (row + 1) * step))
                    mean = sum(crop.tobytes()) / (step * step)
                    if mean < 250:
                        level = min(BINS - 1, int((255 - mean) * BINS / 255))
                        patches.append(_vec(row * GRID + col, level))
            out.append(patches or [_vec(0, 0)])
        return out

    def embed_query(self, text: str):
        vectors = []
        for token in text.split():
            cell, level = token[1:].split("b")
            vectors.append(_vec(int(cell), int(level)))
        return vectors


def make_page(tenant, doc, page_no, data, revision="v1", ref=None) -> VisualPage:
    from ragnor.retrieval.visual import image_dimensions  # noqa: PLC0415

    width, height = image_dimensions(data)
    return VisualPage(
        tenant=tenant,
        page_id=f"{doc}-p{page_no}",
        source=SourceReference("src", doc, revision),
        page=page_no,
        artifact_ref=ref or f"private://{tenant}/{doc}/{page_no}",
        width=width,
        height=height,
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
