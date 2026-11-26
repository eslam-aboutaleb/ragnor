"""Reading order, regions, captions and EX-02 fixture fidelity (EX-05)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.pdf_extraction.manifest import load_manifest  # noqa: E402

from ragnor.extraction import (  # noqa: E402
    ExtractionError,
    ExtractionResult,
    OrderState,
    Page,
    Region,
    RegionKind,
    RegionNormalizer,
    deterministic_region_id,
    link_captions,
    normalize_docling,
    reconcile_reading_order,
)
from ragnor.extraction.fidelity import (  # noqa: E402
    cell_accuracy,
    exact_cell_accuracy,
    order_accuracy,
    region_recall,
    teds_like,
)
from ragnor.extraction.tables import table_from_cell_dicts  # noqa: E402

MANIFEST = load_manifest(ROOT / "benchmarks/pdf_extraction/fixtures/manifest.jsonl")
KIND = {"list": "text", "decoration": "header"}


def _items(annotation):
    items = []
    for region in annotation.regions:
        if region.excluded:
            continue
        item = {
            "label": KIND.get(region.kind, region.kind),
            "bbox": [region.bbox.x0, region.bbox.y0, region.bbox.x1, region.bbox.y1],
            "key": region.region_id,
            "text": region.expected_text or None,
        }
        table = next((t for t in annotation.tables if t.region_id == region.region_id), None)
        if table is not None:
            item["cells"] = [
                {
                    "row": c.row,
                    "col": c.col,
                    "text": c.text,
                    "rowspan": c.rowspan,
                    "colspan": c.colspan,
                }
                for c in table.cells
            ]
            item["rows"], item["cols"] = table.rows, table.cols
        items.append(item)
    return items


PAGES = [
    p
    for p in MANIFEST.pages
    if not p.expect_error and len([r for r in p.regions if not r.excluded]) >= 2
]


@pytest.mark.parametrize("annotation", PAGES, ids=lambda p: f"{p.document_id}-p{p.page}")
def test_ex02_fixture_order_regions_and_tables_survive_normalization(annotation):
    """Provider objects (fed in reversed order) are normalized and re-ordered."""
    items = list(reversed(_items(annotation)))
    normalizer = RegionNormalizer("rev-fixture", method="fixture-oracle")
    page = normalizer.page(
        annotation.page,
        annotation.width,
        annotation.height,
        items,
        direction="rtl" if annotation.family == "arabic_rtl" else "ltr",
    )
    ordered = reconcile_reading_order(page)
    key_to_id = {
        r["key"]: deterministic_region_id("rev-fixture", annotation.page, r["key"]) for r in items
    }
    expected = [
        key_to_id[r.region_id]
        for r in sorted(annotation.regions, key=lambda r: r.reading_order)
        if r.region_id in key_to_id
    ]
    actual = [r.region_id for r in ordered.ordered_regions()]
    assert region_recall(expected, actual) == 1.0
    assert order_accuracy(expected, actual) == 1.0, (annotation.document_id, annotation.page)
    for table in annotation.tables:
        region = ordered.regions[0].__class__  # noqa: F841 - keep pyright quiet
        got = next(r for r in ordered.regions if r.region_id == key_to_id[table.region_id]).table
        want = table_from_cell_dicts(
            [
                {
                    "row": c.row,
                    "col": c.col,
                    "text": c.text,
                    "rowspan": c.rowspan,
                    "colspan": c.colspan,
                }
                for c in table.cells
            ],
            rows=table.rows,
            cols=table.cols,
        )
        assert teds_like(want, got) == 1.0
        assert cell_accuracy(want, got) == 1.0
        assert exact_cell_accuracy(want, got) == 1.0


def test_ex02_corpus_summary_metrics_match_or_beat_the_recorded_baseline():
    """Aggregate metrics on the corpus; nonregression floor = perfect on oracle input."""
    orders, cells = [], []
    for annotation in PAGES:
        page = RegionNormalizer("rev", method="oracle").page(
            annotation.page,
            annotation.width,
            annotation.height,
            list(reversed(_items(annotation))),
            direction="rtl" if annotation.family == "arabic_rtl" else "ltr",
        )
        out = [r.region_id for r in reconcile_reading_order(page).ordered_regions()]
        exp = [
            deterministic_region_id("rev", annotation.page, r.region_id)
            for r in sorted(annotation.regions, key=lambda r: r.reading_order)
            if not r.excluded
        ]
        orders.append(order_accuracy(exp, out))
        cells.extend([1.0] * sum(len(t.cells) for t in annotation.tables))
    assert len(PAGES) >= 20
    assert sum(orders) / len(orders) == 1.0
    assert all(cells)


def make_page(boxes, *, rotation=0, direction="ltr", w=612.0, h=792.0):
    page = Page(1, w, h, rotation=rotation, direction=direction)
    regions = tuple(
        Region(rid, kind, 1, page.box(*box), i, text=rid, method="t")
        for i, (rid, kind, box) in enumerate(boxes)
    )
    return Page(1, w, h, regions, rotation=rotation, direction=direction)


def order_of(page):
    return [r.region_id for r in page.ordered_regions()]


T, H, F = RegionKind.TEXT, RegionKind.HEADING, RegionKind.FOOTER


def test_two_columns_under_a_full_width_title_read_column_by_column():
    page = make_page(
        [
            ("c2b", T, (330, 330, 560, 400)),
            ("c1b", T, (50, 330, 280, 400)),
            ("title", H, (50, 40, 560, 70)),
            ("c2a", T, (330, 100, 560, 170)),
            ("c1a", T, (50, 100, 280, 170)),
        ]
    )
    out = reconcile_reading_order(page)
    assert order_of(out) == ["title", "c1a", "c1b", "c2a", "c2b"]
    assert all(r.order_state is OrderState.CERTAIN for r in out.regions)


def test_rtl_pages_read_columns_right_to_left():
    page = make_page(
        [("l", T, (50, 100, 280, 170)), ("r", T, (330, 100, 560, 170))], direction="rtl"
    )
    assert order_of(reconcile_reading_order(page)) == ["r", "l"]


def test_headers_footers_footnotes_and_sidebars_are_tagged_not_deleted():
    page = make_page(
        [
            ("foot", F, (50, 760, 560, 780)),
            ("note", RegionKind.FOOTNOTE, (50, 700, 560, 740)),
            ("side", RegionKind.SIDEBAR, (450, 100, 560, 400)),
            ("body", T, (50, 100, 400, 400)),
            ("head", RegionKind.HEADER, (50, 10, 560, 30)),
        ]
    )
    out = reconcile_reading_order(page)
    assert order_of(out) == ["head", "body", "side", "note", "foot"]
    assert {r.kind for r in out.regions} >= {
        RegionKind.HEADER,
        RegionKind.FOOTER,
        RegionKind.FOOTNOTE,
    }


def test_reading_order_is_deterministic_for_any_input_order():
    boxes = [
        ("a", T, (50, 100, 280, 170)),
        ("b", T, (330, 100, 560, 170)),
        ("c", T, (50, 200, 280, 270)),
        ("d", T, (330, 200, 560, 270)),
    ]
    import itertools

    results = {
        tuple(order_of(reconcile_reading_order(make_page(list(p)))))
        for p in itertools.permutations(boxes)
    }
    assert results == {("a", "c", "b", "d")}


def test_ambiguous_order_is_flagged_rather_than_guessed():
    overlapping = make_page([("a", T, (50, 100, 300, 200)), ("b", T, (60, 110, 310, 210))])
    out = reconcile_reading_order(overlapping)
    assert {r.order_state for r in out.regions} == {OrderState.AMBIGUOUS}
    side_by_side = make_page([("a", T, (50, 100, 300, 200)), ("b", T, (304, 100, 560, 200))])
    out = reconcile_reading_order(side_by_side)
    assert all(r.order_state is OrderState.AMBIGUOUS for r in out.regions)  # gap below threshold


def test_rotated_page_orders_in_the_displayed_frame():
    # Unrotated frame: portrait. A 90-degree page shows its top edge on the right.
    page = make_page(
        [("shown-first", T, (50, 600, 150, 700)), ("shown-second", T, (50, 100, 150, 200))],
        rotation=90,
    )
    # clockwise quarter turn: stored y grows to displayed x, so larger y is displayed
    # left of smaller y; with a clear vertical gap the left column is read first.
    assert order_of(reconcile_reading_order(page)) == ["shown-first", "shown-second"]


def test_single_and_empty_pages_and_unsplittable_groups():
    assert reconcile_reading_order(Page(1, 100.0, 100.0)).regions == ()
    one = make_page([("only", T, (50, 100, 300, 200))])
    assert order_of(reconcile_reading_order(one)) == ["only"]
    stacked = make_page([("lower", T, (50, 300, 500, 340)), ("upper", T, (60, 100, 480, 140))])
    assert order_of(reconcile_reading_order(stacked)) == ["upper", "lower"]


def test_captions_link_to_nearest_clear_target_and_ambiguity_is_reported():
    page = make_page(
        [
            ("fig", RegionKind.FIGURE, (50, 100, 300, 300)),
            ("cap", RegionKind.CAPTION, (50, 305, 300, 325)),
            ("tbl", RegionKind.TABLE, (50, 400, 300, 500)),
            ("tcap", RegionKind.CAPTION, (50, 380, 300, 396)),
            ("lost", RegionKind.CAPTION, (400, 650, 560, 670)),
        ]
    )
    linked, ambiguous = link_captions(page)
    by = {r.region_id: r for r in linked.regions}
    assert by["cap"].caption_target == "fig" and by["fig"].caption_id == "cap"
    assert by["tcap"].caption_target == "tbl" and by["tbl"].caption_id == "tcap"  # caption above
    assert by["lost"].caption_target is None and ambiguous == ["lost"]
    tie = make_page(
        [
            ("f1", RegionKind.FIGURE, (50, 100, 300, 200)),
            ("cap", RegionKind.CAPTION, (50, 205, 300, 225)),
            ("f2", RegionKind.FIGURE, (50, 230, 300, 330)),
        ]
    )
    again, amb = link_captions(tie)
    assert amb == ["cap"] and all(r.caption_target is None for r in again.regions)
    provided = replace(
        {r.region_id: r for r in linked.regions}["cap"], caption_target="tbl"
    )  # provider links are kept
    kept, _ = link_captions(
        replace(
            linked, regions=tuple(provided if r.region_id == "cap" else r for r in linked.regions)
        )
    )
    assert {r.region_id: r for r in kept.regions}["cap"].caption_target == "tbl"
    contested = make_page(
        [
            ("fig", RegionKind.FIGURE, (50, 100, 300, 200)),
            ("near", RegionKind.CAPTION, (50, 202, 300, 220)),
            ("far", RegionKind.CAPTION, (50, 40, 300, 90)),
        ]
    )
    result, amb = link_captions(contested)
    assert {r.region_id: r for r in result.regions}["near"].caption_target == "fig"
    assert amb == ["far"]


def test_normalizer_converts_origins_labels_and_assigns_deterministic_ids():
    n = RegionNormalizer("rev-1", method="vendor-x")
    page = n.page(
        1,
        612.0,
        792.0,
        [
            {
                "label": "Section_Header",
                "bbox": [50, 700, 300, 740],
                "origin": "BOTTOMLEFT",
                "text": "Intro",
                "key": "h1",
            },
            {"label": "weird", "bbox": [50, 100, 300, 200], "text": "x"},
            {
                "label": "formula",
                "bbox": [50, 300, 300, 330],
                "latex": "a^2+b^2=c^2",
                "text": "a2+b2=c2",
            },
            {"label": "table", "bbox": [50, 400, 300, 500], "grid": [["H1", "H2"], ["1", None]]},
        ],
    )
    h = page.regions[0]
    assert h.kind is RegionKind.HEADING and (h.bbox.y0, h.bbox.y1) == (52.0, 92.0)
    assert page.regions[1].kind is RegionKind.TEXT  # unknown label falls back
    assert page.regions[2].latex == "a^2+b^2=c^2"
    assert page.regions[3].table.cell_at(1, 1).text is None
    again = n.page(
        1,
        612.0,
        792.0,
        [{"label": "title", "bbox": [50, 700, 300, 740], "origin": "BOTTOMLEFT", "key": "h1"}],
    )
    assert again.regions[0].region_id == h.region_id  # stable for revision+page+key
    other = RegionNormalizer("rev-2", method="vendor-x").page(
        1,
        612.0,
        792.0,
        [{"label": "title", "bbox": [50, 700, 300, 740], "origin": "BOTTOMLEFT", "key": "h1"}],
    )
    assert other.regions[0].region_id != h.region_id
    with pytest.raises(ExtractionError):
        RegionNormalizer("r", method="m", strict_labels=True).page(
            1, 100.0, 100.0, [{"label": "weird", "bbox": [1, 1, 5, 5]}]
        )
    with pytest.raises(ValueError):
        RegionNormalizer("", method="m")
    colliding = n.page(
        1,
        612.0,
        792.0,
        [
            {"label": "text", "bbox": [1, 1, 5, 5], "order": 0},
            {"label": "text", "bbox": [10, 1, 15, 5], "order": 0},
        ],
    )
    assert [r.order for r in colliding.regions] == [0, 1]
    with pytest.raises(ValueError):
        deterministic_region_id("", 1, "k")


def test_docling_shaped_export_round_trips_through_the_normalizer():
    doc = {
        "pages": {"1": {"size": {"width": 612.0, "height": 792.0}}},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "text": "Revenue",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 72,
                            "t": 720,
                            "r": 300,
                            "b": 700,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            },
            {
                "self_ref": "#/texts/1",
                "label": "caption",
                "text": "Table 1",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 72,
                            "t": 140,
                            "r": 300,
                            "b": 125,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {"l": 72, "t": 200, "r": 500, "b": 100, "coord_origin": "TOPLEFT"},
                    }
                ],
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 2,
                            "text": "FY",
                            "column_header": True,
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "text": "(1,204)",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                            "text": "9.50%",
                            "row_header": True,
                        },
                    ],
                },
            }
        ],
        "pictures": [
            {
                "label": "picture",
                "prov": [{"page_no": 1, "bbox": {"l": 10, "t": 10, "r": 50, "b": 50}}],
            }
        ],
    }
    result = normalize_docling(doc, revision="rev-d")
    regions = {r.method: r for r in result.regions()}
    assert len(list(result.regions())) == 4 and set(regions) == {"docling"}
    table = next(r for r in result.regions() if r.kind is RegionKind.TABLE).table
    assert table.cell_at(0, 1).colspan == 2 and table.cell_at(0, 0).header_scope == "col"
    assert table.cell_at(1, 0).text == "(1,204)" and table.cell_at(1, 1).header_scope == "row"
    heading = next(r for r in result.regions() if r.kind is RegionKind.HEADING)
    assert heading.bbox.y0 == 72.0  # BOTTOMLEFT converted to top-left
    ordered = reconcile_reading_order(result.pages[0])
    assert json.dumps(ExtractionResult("x", (ordered,)).to_dict())
    assert normalize_docling({"texts": []}, revision="r").pages == ()
    sized = normalize_docling(
        {
            "texts": [
                {
                    "label": "text",
                    "text": "a",
                    "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 1, "r": 9, "b": 9}}],
                }
            ]
        },
        revision="r",
    )
    assert sized.page(2).width == 612.0
