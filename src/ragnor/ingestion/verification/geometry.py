"""Small box helpers for verification, exact and dependency-free."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ragnor.ingestion.extraction_types import BBox


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def covered_fraction(target: BBox, others: Iterable[BBox]) -> float:
    """Exact fraction of ``target`` covered by the union of ``others``.

    Coordinate compression over the clipped boxes avoids rasterization error, so
    empty page whitespace elsewhere can never count as coverage.
    """
    total = area(target)
    if total <= 0:
        return 0.0
    clipped = [
        (max(target[0], b[0]), max(target[1], b[1]), min(target[2], b[2]), min(target[3], b[3]))
        for b in others
    ]
    clipped = [b for b in clipped if b[0] < b[2] and b[1] < b[3]]
    if not clipped:
        return 0.0
    xs = sorted({v for b in clipped for v in (b[0], b[2])})
    covered = 0.0
    for left, right in zip(xs, xs[1:], strict=False):
        spans = sorted((b[1], b[3]) for b in clipped if b[0] <= left and b[2] >= right)
        length, end = 0.0, None
        for lo, hi in spans:
            if end is None or lo > end:
                length += hi - lo
                end = hi
            elif hi > end:
                length += hi - end
                end = hi
        covered += (right - left) * length
    return min(1.0, covered / total)


def region_ref(page: int, kind: str, box: BBox, taken: set[str]) -> str:
    quantized = tuple(round(v * 10_000) for v in box)
    base = "rv_" + hashlib.sha256(f"{page}|{kind}|{quantized}".encode()).hexdigest()[:12]
    ref, n = base, 1
    while ref in taken:
        n += 1
        ref = f"{base}.{n}"
    taken.add(ref)
    return ref
