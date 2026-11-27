"""Caption association for figures and tables."""

from __future__ import annotations

from dataclasses import replace

from ragnor.extraction.types import BoundingBox, Page, RegionKind

TARGETS = frozenset({RegionKind.FIGURE, RegionKind.TABLE})
MIN_X_OVERLAP = 0.5
MAX_GAP_RATIO = 0.15
AMBIGUITY_MARGIN = 0.25


def _x_overlap(a: BoundingBox, b: BoundingBox) -> float:
    inter = min(a.x1, b.x1) - max(a.x0, b.x0)
    return max(inter, 0.0) / min(a.width, b.width)


def _gap(caption: BoundingBox, target: BoundingBox) -> float | None:
    if caption.y0 >= target.y1 - 1e-6:  # caption below
        return caption.y0 - target.y1
    if caption.y1 <= target.y0 + 1e-6:  # caption above
        return target.y0 - caption.y1
    return None


def link_captions(page: Page) -> tuple[Page, list[str]]:
    """Associate captions with the nearest figure/table above or below.

    Existing provider links are kept. A caption is linked only when exactly one
    target is clearly nearest (horizontal overlap, bounded gap, no near-tie);
    every other caption is left unlinked and returned in the ambiguity list.
    A target takes at most one caption (the nearest wins).
    """
    shown = {r.region_id: r.bbox.to_display() for r in page.regions}
    limit = MAX_GAP_RATIO * next(iter(shown.values())).page_height if shown else 0.0
    targets = [r for r in page.regions if r.kind in TARGETS]
    taken = {r.region_id: r.caption_id for r in targets if r.caption_id}
    ambiguous: list[str] = []
    proposals: list[tuple[float, str, str]] = []
    for cap in sorted(
        (r for r in page.regions if r.kind is RegionKind.CAPTION), key=lambda r: r.region_id
    ):
        if cap.caption_target:
            continue
        options = []
        for t in targets:
            gap = _gap(shown[cap.region_id], shown[t.region_id])
            if (
                gap is not None
                and gap <= limit
                and _x_overlap(shown[cap.region_id], shown[t.region_id]) >= MIN_X_OVERLAP
            ):
                options.append((gap, t.region_id))
        options.sort()
        if not options or (
            len(options) > 1
            and options[1][0] - options[0][0] <= AMBIGUITY_MARGIN * max(options[0][0], 1e-9) + 1e-6
        ):
            ambiguous.append(cap.region_id)
        else:
            proposals.append((options[0][0], cap.region_id, options[0][1]))
    links: dict[str, str] = {}
    for _g, cap_id, tgt_id in sorted(proposals):
        if tgt_id in taken or tgt_id in links.values():
            ambiguous.append(cap_id)
        else:
            links[cap_id] = tgt_id
    by_target = {t: c for c, t in links.items()}
    regions = tuple(
        replace(r, caption_target=links[r.region_id])
        if r.region_id in links
        else replace(r, caption_id=by_target[r.region_id])
        if r.region_id in by_target
        else r
        for r in page.regions
    )
    return replace(page, regions=regions), sorted(set(ambiguous))
