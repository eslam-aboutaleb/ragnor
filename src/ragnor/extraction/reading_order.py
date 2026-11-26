"""Deterministic reading-order reconciliation with an uncertainty state.

Works on the *displayed* frame (page rotation applied). Headers, footers,
footnotes and sidebars are kept and tagged, never deleted: headers come first,
then the main flow, sidebars, footnotes, footers last.

Main flow uses a constrained XY-cut: full-width regions split the page into
bands; inside a band, a vertical whitespace gap across all regions defines
columns (left to right, or right to left for ``rtl`` pages); otherwise regions
are sorted top to bottom. Ties break on ``(y0, x0, region_id)``.
Regions whose order cannot be justified are tagged ``ambiguous``.
"""

from __future__ import annotations

from dataclasses import replace

from ragnor.extraction.types import BoundingBox, OrderState, Page, Region, RegionKind

FULL_WIDTH_RATIO = 0.7
MIN_GAP_RATIO = 0.02
OVERLAP_RATIO = 0.5

_FRONT = {RegionKind.HEADER}
_BACK = [RegionKind.SIDEBAR, RegionKind.FOOTNOTE, RegionKind.FOOTER]


def _key(item: tuple[Region, BoundingBox]) -> tuple[float, float, str]:
    r, b = item
    return (round(b.y0, 6), round(b.x0, 6), r.region_id)


def _columns(items: list[tuple[Region, BoundingBox]], min_gap: float):
    spans = sorted((b.x0, b.x1) for _, b in items)
    groups: list[list[float]] = []
    for x0, x1 in spans:
        if groups and x0 - groups[-1][1] < min_gap:
            groups[-1][1] = max(groups[-1][1], x1)
        else:
            groups.append([x0, x1])
    if len(groups) < 2:
        return None
    cols = [[] for _ in groups]
    for it in items:
        mid = (it[1].x0 + it[1].x1) / 2
        idx = next(i for i, g in enumerate(groups) if g[0] <= mid <= g[1])
        cols[idx].append(it)
    return cols


def _flow(items: list[tuple[Region, BoundingBox]], width: float, rtl: bool, ambiguous: set[str]):
    if len(items) <= 1:
        return list(items)
    x_lo = min(b.x0 for _, b in items)
    x_hi = max(b.x1 for _, b in items)
    group_w = max(x_hi - x_lo, 1e-9)
    full = [it for it in items if it[1].width >= FULL_WIDTH_RATIO * group_w and group_w > 0.0]
    if full and len(full) < len(items):
        out: list[tuple[Region, BoundingBox]] = []
        bands: list[list[tuple[Region, BoundingBox]]] = [[]]
        full_sorted = sorted(full, key=_key)
        cuts = [(f, f[1].y0, f[1].y1) for f in full_sorted]
        for it in sorted((i for i in items if i not in full), key=_key):
            centre = (it[1].y0 + it[1].y1) / 2
            idx = sum(1 for _, _, y1 in cuts if centre > y1)
            while len(bands) <= len(cuts):
                bands.append([])
            bands[idx].append(it)
        for i, f in enumerate(full_sorted):
            out.extend(_flow(bands[i], width, rtl, ambiguous))
            out.append(f)
        out.extend(_flow(bands[len(full_sorted)], width, rtl, ambiguous))
        return out
    cols = _columns(items, MIN_GAP_RATIO * width)
    if cols is not None:
        ordered = list(reversed(cols)) if rtl else cols
        out = []
        for col in ordered:
            out.extend(_flow(col, width, rtl, ambiguous))
        return out
    ordered = sorted(
        items, key=(lambda it: (round(it[1].y0, 6), -it[1].x1, it[0].region_id)) if rtl else _key
    )
    for i, (ra, a) in enumerate(ordered):
        for rb, b in ordered[i + 1 :]:
            y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            if y_overlap > OVERLAP_RATIO * min(a.height, b.height) and (
                a.x1 <= b.x0 or b.x1 <= a.x0
            ):
                ambiguous.update({ra.region_id, rb.region_id})
    return ordered


def reconcile_reading_order(page: Page) -> Page:
    """Return ``page`` with explicit ``order`` and ``order_state`` recomputed."""
    shown = {r.region_id: r.bbox.to_display() for r in page.regions}
    ambiguous: set[str] = set()
    ids = list(shown)
    for i, a in enumerate(ids):  # heavy overlap makes any order a guess
        for b in ids[i + 1 :]:
            inter = shown[a].overlap_area(shown[b])
            if inter > OVERLAP_RATIO * min(shown[a].area, shown[b].area):
                ambiguous.update({a, b})
    pairs = [(r, shown[r.region_id]) for r in page.regions]
    width = max((b.page_width for _, b in pairs), default=1.0)
    rtl = page.direction == "rtl"
    front = sorted((p for p in pairs if p[0].kind in _FRONT), key=_key)
    back = [sorted((p for p in pairs if p[0].kind is k), key=_key) for k in _BACK]
    main = [p for p in pairs if p[0].kind not in _FRONT and p[0].kind not in _BACK]
    sequence = [*front, *_flow(main, width, rtl, ambiguous), *(i for grp in back for i in grp)]
    regions = tuple(
        replace(
            r,
            order=index,
            order_state=OrderState.AMBIGUOUS if r.region_id in ambiguous else OrderState.CERTAIN,
        )
        for index, (r, _) in enumerate(sequence)
    )
    return replace(page, regions=regions)
