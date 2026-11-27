"""Table fidelity, continuation, citation and route behaviour (EX-05)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace

import pytest
from tests.extraction_doubles import FakeProvider
from hypothesis import given, settings
from hypothesis import strategies as st

from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.extraction import (
    BoundingBox,
    Capability,
    DerivedContent,
    ExtractionConfigError,
    ExtractionError,
    ExtractionResult,
    HeaderScope,
    Page,
    ProviderRegistry,
    Region,
    RegionKind,
    to_raw_documents,
)
from ragnor.extraction.citation import citation_from_metadata, resolve_citation
from ragnor.extraction.routes import apply_routes
from ragnor.extraction.serializers import region_text, table_to_html
from ragnor.extraction.tables import (
    link_table_continuations,
    logical_table,
    merge_declared_table_boxes,
    split_table_at,
    stack_tables,
    table_from_cell_dicts,
    table_from_grid,
)
from ragnor.ingestion.pipeline import IngestionPipeline
from ragnor.stores.memory import InMemoryVectorStore


def test_grid_ragged_rejected_none_is_missing_and_values_stay_exact():
    with pytest.raises(ValueError):
        table_from_grid([["a", "b"], ["c"]])
    with pytest.raises(ValueError):
        table_from_grid([])
    with pytest.raises(ValueError):
        table_from_grid([["a"]], header_rows=5)
    t = table_from_grid([["Item", "FY24"], ["Revenue", "(1,204.50)"], ["Margin", None]])
    assert t.cell_at(1, 1).text == "(1,204.50)"  # no numeric reformatting
    assert t.cell_at(2, 1).text is None and t.cell_at(0, 0).header_scope is HeaderScope.COL
    assert t.cell_at(1, 0).header_scope is None


def test_cell_dicts_support_spans_and_reject_overlap_or_empty():
    t = table_from_cell_dicts(
        [
            {"row": 0, "col": 0, "text": "Group", "colspan": 2, "header_scope": "col"},
            {"row": 1, "col": 0, "text": "a"},
            {"row": 1, "col": 1, "text": "b"},
        ]
    )
    assert (t.rows, t.cols) == (2, 2) and t.cell_at(0, 1).text == "Group"
    with pytest.raises(ValueError):
        table_from_cell_dicts([])
    with pytest.raises(ValueError):
        table_from_cell_dicts([{"row": 0, "col": 0, "colspan": 2}, {"row": 0, "col": 1}])


def test_split_and_stack_round_trip_and_refuse_unsafe_inputs():
    t = table_from_grid([["h1", "h2"], ["a", "b"], ["c", "d"]])
    top, bottom = split_table_at(t, 2)
    assert (top.rows, bottom.rows) == (2, 1)
    assert stack_tables([top, bottom]) == t
    with pytest.raises(ValueError):
        split_table_at(t, 0)
    spanning = table_from_cell_dicts(
        [
            {"row": 0, "col": 0, "rowspan": 2, "text": "x"},
            {"row": 0, "col": 1, "text": "y"},
            {"row": 1, "col": 1, "text": "z"},
        ]
    )
    with pytest.raises(ValueError):
        split_table_at(spanning, 1)
    with pytest.raises(ValueError):
        stack_tables([t, table_from_grid([["x"]])])
    with pytest.raises(ValueError):
        stack_tables([])


def _tbl_region(rid, page, order, grid, box=(50, 100, 500, 200), **kw):
    p = Page(page, 612.0, 792.0)
    return Region(
        rid,
        RegionKind.TABLE,
        page,
        p.box(*box),
        order,
        table=table_from_grid(grid),
        method="t",
        **kw,
    )


def test_merge_only_when_declared_and_never_by_similarity():
    page = Page(
        1,
        612.0,
        792.0,
        (
            _tbl_region("a", 1, 0, [["h", "h"], ["1", "2"]], (50, 100, 500, 150)),
            _tbl_region("b", 1, 1, [["3", "4"]], (50, 150, 500, 200)),
            _tbl_region("c", 1, 2, [["5", "6"]], (50, 300, 500, 350)),
        ),
    )
    untouched = merge_declared_table_boxes(page, {}, revision="r")
    assert [r.region_id for r in untouched.regions] == ["a", "b", "c"]  # similar but undeclared
    merged = merge_declared_table_boxes(page, {"g": ["a", "b"]}, revision="r")
    assert len(merged.regions) == 2 and merged.regions[0].table.rows == 3
    assert merged.regions[0].bbox.y0 == 100 and merged.regions[0].bbox.y1 == 200
    with pytest.raises(ExtractionError):
        merge_declared_table_boxes(page, {"g": ["a", "missing"]}, revision="r")
    wide = replace(page, regions=(page.regions[0], _tbl_region("w", 1, 1, [["1", "2", "3"]])))
    with pytest.raises(ExtractionError):
        merge_declared_table_boxes(wide, {"g": ["a", "w"]}, revision="r")


def _two_pages(second_grid, *, declared=False, first_grid=None):
    first = first_grid or [["Name", "Amount"], ["a", "1"]]
    r1 = _tbl_region("t1", 1, 0, first)
    r2 = _tbl_region("t2", 2, 0, second_grid, continues_from="t1" if declared else None)
    return ExtractionResult("rev", (Page(1, 612.0, 792.0, (r1,)), Page(2, 612.0, 792.0, (r2,))))


def test_continuation_linked_on_repeated_header_or_declaration_not_on_mismatch():
    repeated = link_table_continuations(_two_pages([["Name", "Amount"], ["b", "2"]]))
    t2 = repeated.region("t2")
    assert t2.continues_from == "t1" and repeated.region("t1").page_refs == (1, 2)
    table, chain = logical_table(repeated, "t2")
    assert chain == ("t1", "t2") and table.rows == 3  # repeated header dropped once
    assert [table.cell_at(r, 0).text for r in range(3)] == ["Name", "a", "b"]
    assert logical_table(repeated, "t1")[1] == chain  # same chain from the head

    declared = link_table_continuations(_two_pages([["x", "y"], ["b", "2"]], declared=True))
    table, _ = logical_table(declared, "t1")
    assert table.rows == 4  # no equal header: nothing dropped

    mismatch = _two_pages([["Other", "Cols"], ["b", "2"]])
    assert link_table_continuations(mismatch) is mismatch
    width = _two_pages([["Name", "Amount", "Extra"], ["b", "2", "3"]])
    assert link_table_continuations(width) is width
    with pytest.raises(ExtractionError):
        logical_table(mismatch, "nope")


def test_continuation_skips_page_gaps_and_furniture_and_three_page_chains():
    gap = ExtractionResult(
        "rev",
        (
            Page(1, 612.0, 792.0, (_tbl_region("t1", 1, 0, [["h"], ["1"]]),)),
            Page(3, 612.0, 792.0, (_tbl_region("t3", 3, 0, [["h"], ["2"]]),)),
        ),
    )
    assert link_table_continuations(gap) is gap
    pg = Page(2, 612.0, 792.0)
    footer = Region("f1", RegionKind.FOOTER, 2, pg.box(0, 760, 600, 780), 9, text="p2", method="t")
    head = Region("h2", RegionKind.HEADER, 2, pg.box(0, 0, 600, 20), 0, text="hdr", method="t")
    body = _tbl_region("t2", 2, 1, [["h"], ["2"]])
    third = _tbl_region("t3", 3, 0, [["h"], ["3"]])
    three = ExtractionResult(
        "rev",
        (
            Page(1, 612.0, 792.0, (_tbl_region("t1", 1, 0, [["h"], ["1"]]),)),
            Page(2, 612.0, 792.0, (head, body, footer)),
            Page(3, 612.0, 792.0, (third,)),
        ),
    )
    linked = link_table_continuations(three)
    assert logical_table(linked, "t3")[1] == ("t1", "t2", "t3")
    assert linked.region("t2").page_refs == (1, 2, 3)
    assert logical_table(linked, "t2")[0].rows == 4
    empty = ExtractionResult("rev", (Page(1, 612.0, 792.0), Page(2, 612.0, 792.0)))
    assert link_table_continuations(empty) is empty
    cyc = replace(
        linked,
        pages=tuple(
            replace(
                p,
                regions=tuple(
                    replace(r, continues_from="t3") if r.region_id == "t1" else r for r in p.regions
                ),
            )
            for p in linked.pages
        ),
    )
    with pytest.raises(ExtractionError):
        logical_table(cyc, "t3")


def test_html_is_escaped_and_financial_values_are_exact_in_text():
    t = table_from_grid(
        [["Item", "<script>alert(1)</script>"], ["Net", "(1,204.50)"], ["Pct", "9.50%"]]
    )
    region = Region(
        "t", RegionKind.TABLE, 1, BoundingBox(0, 0, 10, 10, 100, 100), 0, table=t, method="x"
    )
    text = region_text(region)
    html = table_to_html(t)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "(1,204.50)" in text and "9.50%" in text


@settings(max_examples=60, deadline=None)
@given(
    st.integers(1, 6).flatmap(
        lambda n: st.lists(
            st.lists(st.one_of(st.none(), st.text(max_size=8)), min_size=n, max_size=n),
            min_size=1,
            max_size=6,
        )
    ),
    st.integers(0, 3),
)
def test_grid_properties_occupancy_unique_and_stable_serialization(grid, headers):
    t = table_from_grid(grid, header_rows=min(headers, len(grid)))
    assert len(t.occupancy()) == t.rows * t.cols and not t.uncovered_positions()
    assert len({(c.row, c.col) for c in t.cells}) == len(t.cells)
    if len(grid) > 1:
        top, bottom = split_table_at(t, 1)
        assert stack_tables([top, bottom]) == t
    p = Page(1, 612.0, 792.0)
    r = Region("t", RegionKind.TABLE, 1, p.box(1, 1, 50, 50), 0, table=t, method="x")
    a = json.dumps(r.to_dict(), sort_keys=True)
    assert a == json.dumps(r.to_dict(), sort_keys=True)


# ---- citation ---------------------------------------------------------------


def _embed(texts):
    out = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        out.append([b / 255 for b in digest[:8]])
    return out


async def _aembed(texts):
    return _embed(texts)


def _document_result():
    p = Page(1, 612.0, 792.0)
    t = table_from_grid([["Item", "FY24"], ["Revenue", "1,204.50"], ["Margin", "9.5%"]])
    t = replace(
        t,
        cells=tuple(
            replace(
                c,
                bbox=p.box(60 + c.col * 100, 300 + c.row * 20, 150 + c.col * 100, 318 + c.row * 20),
            )
            for c in t.cells
        ),
    )
    return ExtractionResult(
        "rev-cite",
        (
            Page(
                1,
                612.0,
                792.0,
                (
                    Region(
                        "intro",
                        RegionKind.TEXT,
                        1,
                        p.box(50, 50, 560, 200),
                        0,
                        text="long " * 400 + "marker-sentence at the end",
                        method="m",
                    ),
                    Region(
                        "fin", RegionKind.TABLE, 1, p.box(50, 290, 400, 370), 1, table=t, method="m"
                    ),
                    Region(
                        "chart",
                        RegionKind.FIGURE,
                        1,
                        p.box(50, 400, 400, 500),
                        2,
                        derived=(
                            DerivedContent("chart_estimate", "roughly 42 units", "vlm-fake", 0.4),
                        ),
                        method="m",
                    ),
                ),
            ),
        ),
    )


def test_retrieval_hit_resolves_to_page_region_bbox_and_cell_even_after_chunking():
    result = _document_result()
    docs = to_raw_documents(result, source_name="annual.pdf")
    assert all("roughly 42" not in d.content for d in docs)  # derived estimate never indexed
    pipeline = IngestionPipeline(None, None, SlidingWindowChunker(50, 5), None, None)  # type: ignore[arg-type]
    chunks = pipeline._chunk(docs)  # noqa: SLF001
    assert len(chunks) > 3  # the long region was split
    store = InMemoryVectorStore(embedding_dim=8, embedding_fn=_aembed)

    async def run():
        await store.upsert(
            [
                {
                    "id": c.id,
                    "text": c.text,
                    "embedding": _embed([c.text])[0],
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
        )
        return await store.keyword_search(
            "marker-sentence", 3, metadata_fields=["structured", "source"]
        ), await store.keyword_search("1,204.50", 3, metadata_fields=["structured", "source"])

    text_hits, table_hits = asyncio.run(run())
    cit = citation_from_metadata(text_hits[0]["metadata"])
    resolved = resolve_citation(result, cit)
    assert (resolved.region.region_id, resolved.citation.page, resolved.cell) == ("intro", 1, None)
    assert resolved.bbox == result.region("intro").bbox

    tcit = citation_from_metadata(table_hits[0]["metadata"])
    cell = resolve_citation(result, tcit, cell_text="1,204.50")
    assert cell.region.region_id == "fin" and cell.cell.text == "1,204.50"
    assert cell.bbox == cell.cell.bbox and cell.bbox != result.region("fin").bbox
    assert resolve_citation(result, tcit, cell_text="nope").cell is None


def test_citation_fails_closed_on_stale_revision_bad_metadata_and_duplicates():
    result = _document_result()
    cit = citation_from_metadata(to_raw_documents(result, source_name="a.pdf")[0].metadata)
    with pytest.raises(ExtractionError):
        resolve_citation(replace(result, revision="rev-other"), cit)
    with pytest.raises(ExtractionError):
        resolve_citation(result, replace(cit, region_id="ghost"))
    with pytest.raises(ExtractionError):
        resolve_citation(result, replace(cit, page=2))
    for bad in (
        {},
        {"structured": "x"},
        {"structured": {"revision": "r"}},
        {"structured": {"revision": "r", "page": "z", "region_id": "a"}},
    ):
        with pytest.raises(ExtractionError):
            citation_from_metadata(bad)
    dup = table_from_grid([["a", "7"], ["b", "7"]])
    p = Page(1, 612.0, 792.0)
    res = ExtractionResult(
        "r",
        (
            Page(
                1,
                612.0,
                792.0,
                (Region("t", RegionKind.TABLE, 1, p.box(1, 1, 9, 9), 0, table=dup, method="m"),),
            ),
        ),
    )
    assert (
        resolve_citation(res, replace(cit, revision="r", region_id="t"), cell_text="7").cell is None
    )


def test_derived_chart_estimate_cannot_be_verified():
    with pytest.raises(ValueError):
        DerivedContent("chart_estimate", "x", "m", verified=True)
    assert DerivedContent("description", "x", "m", verified=True).verified


# ---- routes -----------------------------------------------------------------


class _RefiningProvider(FakeProvider):
    id = "refiner"

    async def extract_region(self, data, region):
        if self.options.get("fail"):
            raise RuntimeError("secret-token leaked")
        if self.options.get("wrong_id"):
            return replace(region, region_id="other")
        return replace(region, text="refined", method="refiner") if region.table is None else region


def _registry(*, remote=False, enabled=False):
    reg = ProviderRegistry()
    reg.register(
        "refiner",
        _RefiningProvider,
        formats=["pdf"],
        capabilities=[Capability.TABLES, Capability.FIGURES, Capability.FORMULAS],
        remote=remote,
        version="1.0",
    )
    if enabled:
        reg.enable_remote("refiner")
    return reg


def test_routes_refine_only_routed_kinds_and_keep_base_on_failure():
    result = _document_result()
    out = asyncio.run(apply_routes(_registry(), result, b"", {"text": "refiner"}))
    assert out.region("intro").text == "refined" and out.region("fin") == result.region("fin")
    failing = asyncio.run(
        apply_routes(
            _registry(),
            result,
            b"",
            {RegionKind.TEXT: "refiner"},
            options={"refiner": {"fail": True}},
        )
    )
    assert failing == result  # base kept, provider message not recorded
    mismatched = asyncio.run(
        apply_routes(
            _registry(), result, b"", {"text": "refiner"}, options={"refiner": {"wrong_id": True}}
        )
    )
    assert mismatched == result


def test_routes_refuse_unknown_providers_and_unenabled_remote_before_sending_data():
    result = _document_result()
    with pytest.raises(ExtractionConfigError):
        asyncio.run(apply_routes(_registry(), result, b"", {"text": "missing"}))
    with pytest.raises(ExtractionConfigError):
        asyncio.run(apply_routes(_registry(remote=True), result, b"", {"text": "refiner"}))
    ok = asyncio.run(
        apply_routes(_registry(remote=True, enabled=True), result, b"", {"text": "refiner"})
    )
    assert ok.region("intro").text == "refined"
