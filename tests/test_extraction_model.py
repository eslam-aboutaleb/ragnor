"""Typed structured-document contracts: invariants and lossless round-trips."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ragnor.extraction import (
    AssetRef,
    BoundingBox,
    Cell,
    CoordinateSpace,
    DerivedContent,
    DocumentProfile,
    ExtractionAttempt,
    ExtractionError,
    ExtractionLimitError,
    ExtractionResult,
    HeaderScope,
    OrderState,
    Page,
    Region,
    RegionKind,
    Table,
    TextSpan,
    from_json,
    to_json,
)
from tests.extraction_doubles import sample_result


def test_result_round_trips_losslessly_with_stable_serialization():
    result = sample_result()
    payload = to_json(result)
    again = from_json(payload)
    assert again == result
    assert to_json(again) == payload
    table = again.region("r-table").table
    assert (table.cell_at(1, 0).rowspan, table.cell_at(0, 1).colspan) == (2, 2)
    assert table.cell_at(2, 2).text is None  # missing stays missing, not ""
    assert table.cell_at(1, 1).text == "1,234.50"  # exact value, no normalization
    assert [r.region_id for r in again.regions()] == ["r-head", "r-body", "r-table"]


def test_region_order_is_explicit_not_index_based():
    page = sample_result().pages[0]
    shuffled = Page(1, 612.0, 792.0, tuple(reversed(page.regions)))
    assert [r.region_id for r in shuffled.ordered_regions()] == ["r-head", "r-body", "r-table"]
    with pytest.raises(ValueError, match="Duplicate region order"):
        Page(1, 612.0, 792.0, (page.regions[0], replace(page.regions[1], order=0)))
    with pytest.raises(ValueError, match="Duplicate region id"):
        Page(
            1,
            612.0,
            792.0,
            (page.regions[0], replace(page.regions[1], region_id="r-head", order=5)),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x0": 10, "x1": 10},
        {"x0": float("nan")},
        {"x1": 700},
        {"rotation": 45},
        {"page_width": 0},
        {"unit": "furlong"},
        {"space": "elsewhere"},
    ],
)
def test_bounding_box_rejects_invalid_geometry(kwargs):
    base = {"x0": 0, "y0": 0, "x1": 100, "y1": 100, "page_width": 612, "page_height": 792}
    with pytest.raises(ValueError):
        BoundingBox(**{**base, **kwargs})


def test_coordinate_invariants_and_conversions():
    bottom_left = BoundingBox(
        10, 700, 110, 780, 612, 792, space=CoordinateSpace.PDF_POINTS_BOTTOM_LEFT
    )
    top = bottom_left.to_top_left()
    assert (top.y0, top.y1, top.space) == (12, 92, CoordinateSpace.PDF_POINTS_TOP_LEFT)
    assert top.to_top_left() is top
    box = BoundingBox(10, 20, 110, 70, 612, 792, rotation=90)
    shown = box.to_display()
    assert (shown.page_width, shown.page_height, shown.rotation) == (792, 612, 0)
    assert (shown.x0, shown.y0, shown.x1, shown.y1) == (722, 10, 772, 110)
    assert BoundingBox(10, 20, 110, 70, 612, 792, rotation=180).to_display().x0 == 502
    quarter = BoundingBox(10, 20, 110, 70, 612, 792, rotation=270).to_display()
    assert (quarter.x0, quarter.y0, quarter.x1, quarter.y1) == (20, 502, 70, 602)
    n = BoundingBox.normalized(0.1, 0.1, 0.5, 0.5)
    assert n.area == pytest.approx(0.16)
    assert n.overlap_area(BoundingBox.normalized(0.3, 0.3, 0.9, 0.9)) == pytest.approx(0.04)
    assert n.overlap_area(BoundingBox.normalized(0.6, 0.6, 0.9, 0.9)) == 0
    with pytest.raises(ValueError, match="unit page"):
        BoundingBox(0, 0, 1, 1, 2, 2, unit="normalized")
    with pytest.raises(ValueError, match="four"):
        BoundingBox.from_dict({"box": [1, 2], "page_width": 1, "page_height": 1})


def test_region_frame_must_match_page_frame():
    other = BoundingBox(0, 0, 10, 10, 100, 100)
    with pytest.raises(ValueError, match="frame differs"):
        Page(1, 612.0, 792.0, (Region("a", RegionKind.TEXT, 1, other, 0),))
    with pytest.raises(ValueError, match="does not match"):
        Page(2, 612.0, 792.0, (sample_result().pages[0].regions[0],))


def test_table_invariants():
    with pytest.raises(ValueError, match="Overlapping"):
        Table(2, 2, (Cell(0, 0, "a", colspan=2), Cell(0, 1, "b")))
    with pytest.raises(ValueError, match="span exceeds"):
        Table(1, 1, (Cell(0, 0, "a", colspan=2),))
    with pytest.raises(ValueError, match="positive"):
        Table(0, 1, ())
    with pytest.raises(ValueError, match="supported size"):
        Table(1000, 1000, ())
    with pytest.raises(ValueError, match="Cell instances"):
        Table(1, 1, ("x",))
    with pytest.raises(ValueError, match="coordinates"):
        Cell(-1, 0, "x")
    with pytest.raises(ValueError, match="string or None"):
        Cell(0, 0, 5)
    with pytest.raises(ValueError, match="header scope"):
        Cell(0, 0, "x", header_scope="diagonal")
    sparse = Table(2, 2, (Cell(0, 0, "a"), Cell(1, 1, "d")))
    assert sparse.uncovered_positions() == [(0, 1), (1, 0)]
    assert Cell(0, 0, None).missing


def test_region_invariants():
    box = BoundingBox(0, 0, 10, 10, 100, 100)
    with pytest.raises(ValueError, match="region"):
        Region("bad id", RegionKind.TEXT, 1, box, 0)
    with pytest.raises(ValueError, match="page"):
        Region("a", RegionKind.TEXT, 0, box, 0)
    with pytest.raises(ValueError, match="order"):
        Region("a", RegionKind.TEXT, 1, box, -1)
    with pytest.raises(ValueError, match="BoundingBox"):
        Region("a", RegionKind.TEXT, 1, (0, 0, 1, 1), 0)
    with pytest.raises(ValueError, match="Only table"):
        Region("a", RegionKind.TEXT, 1, box, 0, table=Table(1, 1, ()))
    with pytest.raises(ValueError, match="Region table"):
        Region("a", RegionKind.TABLE, 1, box, 0, table="x")
    with pytest.raises(ValueError, match="Only formula"):
        Region("a", RegionKind.TEXT, 1, box, 0, latex="x")
    with pytest.raises(ValueError, match="caption target"):
        Region("a", RegionKind.TEXT, 1, box, 0, caption_target="b")
    with pytest.raises(ValueError, match="itself"):
        Region("a", RegionKind.CAPTION, 1, box, 0, caption_target="a")
    with pytest.raises(ValueError, match="Confidence"):
        Region("a", RegionKind.TEXT, 1, box, 0, confidence=1.5)
    with pytest.raises(ValueError, match="section"):
        Region("a", RegionKind.TEXT, 1, box, 0, section_path=("x",) * 40)
    with pytest.raises(ValueError, match="references"):
        Region("a", RegionKind.TEXT, 1, box, 0, page_refs=(0,))
    with pytest.raises(ValueError, match="region kind"):
        Region("a", "banner", 1, box, 0)
    assert Region("a", RegionKind.TEXT, 1, box, 0).confidence is None  # never invented


def test_provider_confidence_is_optional_and_preserved():
    result = sample_result()
    assert result.region("r-body").confidence == 0.9
    assert result.region("r-head").confidence is None
    assert "confidence" not in result.to_dict()["pages"][0]["regions"][0]


def test_derived_content_is_never_verified_chart_numbers():
    with pytest.raises(ValueError, match="never be marked verified"):
        DerivedContent("chart_estimate", "bars ~ 40", "vlm", verified=True)
    with pytest.raises(ValueError, match="Confidence"):
        DerivedContent("description", "x", "vlm", confidence=3)
    with pytest.raises(ValueError, match="method"):
        DerivedContent("description", "x", "")
    assert DerivedContent("description", "a chart", "vlm", 0.5).verified is False


def test_asset_refs_hold_references_not_secrets():
    ok = AssetRef("page-1", "page_image", "s3://bucket/key.png", "a" * 64, "image/png")
    assert AssetRef.from_dict(ok.to_dict()) == ok
    for bad in (
        {"uri": "https://user:pw@host/x.png"},
        {"uri": "https://host/x.png?X-Amz-Signature=abc"},
        {"uri": "https://host/x.png#frag"},
        {"sha256": "XYZ"},
        {"kind": "bad kind"},
        {"uri": "x" * 3000},
    ):
        with pytest.raises(ValueError):
            AssetRef("a", "page_image", **bad) if "kind" not in bad else AssetRef("a", **bad)
    with pytest.raises(ValueError, match="asset"):
        AssetRef("bad id", "crop")


def test_text_span_profile_attempt_validation_and_round_trip():
    assert TextSpan.from_dict(TextSpan(2, 5).to_dict()) == TextSpan(2, 5)
    for args in ((5, 2), (-1, 3), (1.0, 2)):
        with pytest.raises(ValueError):
            TextSpan(*args)
    with pytest.raises(ValueError):
        TextSpan(0, 1, "")
    profile = DocumentProfile("application/pdf", 3, True, False, ("en", "ar"), 10, "b" * 64)
    assert DocumentProfile.from_dict(profile.to_dict()) == profile
    for bad in (
        {"media_type": ""},
        {"media_type": "x", "page_count": -1},
        {"media_type": "x", "source_sha256": "z"},
    ):
        with pytest.raises(ValueError):
            DocumentProfile(**bad)
    attempt = ExtractionAttempt("fake", "1", "partial", "partial_output", "f" * 8, 2)
    assert ExtractionAttempt.from_dict(attempt.to_dict()) == attempt
    with pytest.raises(ValueError):
        ExtractionAttempt("bad id", "1", "failed")


def test_full_field_round_trip_for_every_optional_member():
    page = Page(1, 100.0, 100.0, image=AssetRef("img", "page_image"), direction="rtl")
    box = page.box(1, 1, 50, 50)
    region = Region(
        "fig",
        RegionKind.FIGURE,
        1,
        box,
        0,
        text="t",
        section_path=("A", "B"),
        assets=(AssetRef("crop", "crop"),),
        derived=(DerivedContent("description", "a pie chart", "vlm", 0.4),),
        confidence=0.5,
        order_state=OrderState.AMBIGUOUS,
        source_span=TextSpan(0, 1),
        page_refs=(1, 2),
        caption_id="cap",
    )
    formula = Region(
        "f",
        RegionKind.FORMULA,
        1,
        page.box(60, 60, 90, 90),
        1,
        latex="E=mc^2",
        continues_from="fig",
    )
    cell = Cell(0, 0, "x", header_scope=HeaderScope.ROW, bbox=box, source_span=TextSpan(0, 1))
    table = Region("t", RegionKind.TABLE, 1, page.box(2, 2, 3, 3), 2, table=Table(1, 1, (cell,)))
    result = ExtractionResult(
        "r",
        (replace(page, regions=(region, formula, table)),),
        profile=DocumentProfile("application/pdf"),
        reason="partial_output",
        status="partial",
        attempts=(ExtractionAttempt("fake", "1", "failed", "provider_error"),),
    )
    assert from_json(to_json(result)) == result


def test_from_json_fail_closed():
    good = json.loads(to_json(sample_result()))
    with pytest.raises(ExtractionError) as err:
        ExtractionResult.from_dict({**good, "schema_version": 99})
    assert err.value.code == "schema_version"
    with pytest.raises(ExtractionError) as err:
        from_json("{not json")
    assert err.value.code == "invalid_document"
    broken = json.loads(to_json(sample_result()))
    broken["pages"][0]["regions"][0]["bbox"]["box"] = [0, 0, 9999, 9999]
    with pytest.raises(ExtractionError):
        from_json(json.dumps(broken))
    with pytest.raises(ExtractionLimitError):
        from_json("x" * 70_000_000)
    with pytest.raises(ExtractionError):
        ExtractionResult.from_dict("nope")
    assert ExtractionError("made-up").code == "provider_failed"


def test_result_level_invariants():
    page = sample_result().pages[0]
    with pytest.raises(ValueError, match="revision"):
        ExtractionResult("", (page,))
    with pytest.raises(ValueError, match="Duplicate page"):
        ExtractionResult("r", (page, page))
    with pytest.raises(ValueError, match="Duplicate region id"):
        ExtractionResult("r", (page, Page(2, 612.0, 792.0, (replace(page.regions[0], page=2),))))
    assert sample_result().page(1) is not None and sample_result().page(9) is None
    assert sample_result().region("missing") is None
    assert sample_result().complete
