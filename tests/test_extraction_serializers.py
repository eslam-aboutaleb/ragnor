"""Serializers, legacy bridges and pipeline metadata propagation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.extraction import (
    BoundingBox,
    Cell,
    DerivedContent,
    ExtractionLimitError,
    ExtractionStatus,
    HeaderScope,
    Page,
    Region,
    RegionKind,
    Table,
    from_page_outcomes,
    region_metadata,
    region_text,
    result_to_markdown,
    result_to_text,
    table_to_html,
    table_to_markdown,
    table_to_text,
    to_raw_documents,
)
from ragnor.ingestion import IngestionPipeline
from ragnor.ingestion.extraction_types import (
    ExtractionCell,
    ExtractionRegion,
    ExtractionTable,
    ExtractionWord,
    PageExtractionOutcome,
    PageExtractionStatus,
)
from tests.extraction_doubles import sample_result


def test_html_is_faithful_escaped_and_marks_missing_cells():
    html = table_to_html(sample_result().region("r-table").table)
    assert '<th scope="col" colspan="2">Q1-Q2</th>' in html
    assert '<td rowspan="2">Revenue</td>' in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html and "<script>" not in html
    assert '<td data-missing="true"></td>' in html
    sparse = Table(1, 2, (Cell(0, 0, "a"),))
    assert table_to_html(sparse) == '<table><tr><td>a</td><td data-absent="true"></td></tr></table>'


def test_markdown_only_when_lossless():
    simple = Table(
        2,
        2,
        (
            Cell(0, 0, "A|B", header_scope=HeaderScope.COL),
            Cell(0, 1, "N", header_scope=HeaderScope.COL),
            Cell(1, 0, "x"),
            Cell(1, 1, "(1,234.50)"),
        ),
    )
    assert table_to_markdown(simple) == "| A\\|B | N |\n| --- | --- |\n| x | (1,234.50) |"
    assert table_to_markdown(sample_result().region("r-table").table) is None  # spans + missing
    no_header = replace(simple, cells=tuple(replace(c, header_scope=None) for c in simple.cells))
    assert table_to_markdown(no_header) is None
    for text in ("a\nb", "<b>", " padded"):
        t = Table(1, 1, (Cell(0, 0, text, header_scope=HeaderScope.COL),))
        assert table_to_markdown(t) is None
    assert table_to_markdown(Table(1, 2, (Cell(0, 0, "a", header_scope=HeaderScope.COL),))) is None


def test_text_projection_is_exact_and_bounded():
    table = sample_result().region("r-table").table
    assert table_to_text(table).splitlines()[1] == "Revenue | 1,234.50 | (12%)"
    with pytest.raises(ExtractionLimitError):
        table_to_text(table, max_cells=4)
    with pytest.raises(ExtractionLimitError):
        table_to_text(table, max_chars=10)
    with pytest.raises(ExtractionLimitError):
        table_to_html(table, max_cells=4)
    with pytest.raises(ExtractionLimitError):
        table_to_html(table, max_chars=20)
    with pytest.raises(ExtractionLimitError):
        table_to_markdown(table, max_cells=4)


def test_region_text_labels_derived_content_and_excludes_chart_estimates():
    page = Page(1, 100.0, 100.0)
    fig = Region(
        "fig",
        RegionKind.FIGURE,
        1,
        page.box(0, 0, 50, 50),
        0,
        derived=(
            DerivedContent("description", "bar chart of sales", "vlm"),
            DerivedContent("chart_estimate", "Q1 about 40", "vlm"),
        ),
    )
    text = region_text(fig)
    assert "[derived, unverified: description] bar chart of sales" in text
    assert "Q1 about 40" not in text
    assert "Q1 about 40" in region_text(fig, include_estimates=True)
    assert region_text(fig, include_derived=False) == ""
    verified = replace(fig, derived=(DerivedContent("description", "ok", "human", verified=True),))
    assert "[verified: description] ok" in region_text(verified)
    formula = Region("f", RegionKind.FORMULA, 1, page.box(0, 0, 9, 9), 1, latex="E=mc^2")
    assert region_text(formula) == "E=mc^2"


def test_text_and_markdown_projection_tag_not_delete_furniture():
    page = Page(1, 612.0, 792.0)
    base = sample_result().pages[0]
    header = Region(
        "hdr", RegionKind.HEADER, 1, page.box(50, 10, 560, 30), 10, text="ACME Confidential"
    )
    result = replace(sample_result(), pages=(replace(base, regions=(*base.regions, header)),))
    assert "ACME" not in result_to_text(result)
    assert "ACME" in result_to_text(result, include_furniture=True)
    assert result.region("hdr").kind is RegionKind.HEADER  # still present and typed
    md = result_to_markdown(result)
    assert md.startswith("# Results") and "<table>" in md and "ACME" not in md


def test_raw_documents_per_region_carry_page_bbox_and_revision_into_chunks():
    result = sample_result("rev-42")
    docs = to_raw_documents(result, source_name="r.pdf", external_id="doc-1")
    assert len(docs) == 3
    table_doc = docs[2]
    assert table_doc.location["region_id"] == "r-table" and table_doc.location["page"] == 1
    assert table_doc.metadata["structured"]["revision"] == "rev-42"
    assert table_doc.metadata["structured"]["table"] == {"rows": 3, "cols": 3}
    assert "<" not in table_doc.content.replace("<script>x</script>", "")  # no markup added
    # Same step the real IngestionPipeline performs before embedding.
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._chunker = SlidingWindowChunker(chunk_size=5, overlap=1)
    chunks = pipeline._chunk(docs)
    by_region = {c.metadata["region_id"] for c in chunks}
    assert by_region == {"r-head", "r-body", "r-table"}
    for chunk in chunks:
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["location"]["bbox"] == chunk.metadata["structured"]["bbox"]
        assert chunk.metadata["structured"]["revision"] == "rev-42"
        assert chunk.metadata["source"] == "r.pdf"


def test_raw_documents_page_granularity_and_budgets():
    result = sample_result()
    page_docs = to_raw_documents(result, source_name="r.pdf", granularity="page")
    assert len(page_docs) == 1 and page_docs[0].location == {"page": 1}
    ids = [r["region_id"] for r in page_docs[0].metadata["structured"]["regions"]]
    assert ids == ["r-head", "r-body", "r-table"]
    with pytest.raises(ValueError, match="granularity"):
        to_raw_documents(result, source_name="x", granularity="word")
    # an over-budget table never enters the index text
    big = Table(200, 200, tuple(Cell(r, c, "x") for r in range(200) for c in range(200)))
    page = Page(1, 612.0, 792.0)
    region = Region("big", RegionKind.TABLE, 1, page.box(0, 0, 600, 700), 0, table=big)
    huge = replace(result, pages=(replace(page, regions=(region,)),))
    (doc,) = to_raw_documents(huge, source_name="b.pdf")
    assert "omitted from index text" in doc.content and len(doc.content) < 200
    assert doc.metadata["structured"]["truncated"] is True
    long_text = replace(region, kind=RegionKind.TEXT, table=None, text="w " * 60_000)
    (cut,) = to_raw_documents(
        replace(huge, pages=(replace(page, regions=(long_text,)),)), source_name="t"
    )
    assert cut.metadata["structured"]["truncated"] is True and len(cut.content) == 100_000


def test_region_metadata_is_content_free_and_json_safe():
    import json

    meta = region_metadata(sample_result().region("r-table"), "rev")
    assert json.dumps(meta)
    assert "Revenue" not in json.dumps(meta)
    assert meta["bbox_space"] == "pdf-points-top-left" and meta["page_size"] == [612.0, 792.0]


def test_from_page_outcomes_lifts_legacy_evidence():
    words = (
        ExtractionWord("Hello", (0.1, 0.1, 0.2, 0.15), 90.0),
        ExtractionWord("world", (0.22, 0.1, 0.3, 0.15)),
        ExtractionWord("below", (0.1, 0.2, 0.2, 0.25)),
    )
    table = ExtractionTable(
        (0.1, 0.5, 0.9, 0.9),
        1,
        2,
        (
            ExtractionCell(0, 0, "a", (0.1, 0.5, 0.5, 0.9)),
            ExtractionCell(0, 1, "b", (0.5, 0.5, 0.9, 0.9)),
        ),
    )
    ok = PageExtractionOutcome(
        1,
        PageExtractionStatus.EXTRACTED,
        "Hello world below",
        words=words,
        regions=(ExtractionRegion("text", (0.05, 0.05, 0.6, 0.3)),),
        tables=(table,),
        provider="tesseract",
    )
    plain = PageExtractionOutcome(2, PageExtractionStatus.EXTRACTED, "just text")
    blank = PageExtractionOutcome(3, PageExtractionStatus.NEEDS_OCR, "")
    result = from_page_outcomes([ok, plain, blank], revision="r")
    assert result.status is ExtractionStatus.PARTIAL
    first = result.page(1)
    assert [r.kind for r in first.ordered_regions()] == [RegionKind.TEXT, RegionKind.TABLE]
    assert first.regions[0].text == "Hello world below"
    assert first.regions[0].confidence is None  # word confidences are not region confidences
    assert first.regions[1].table.cell_at(0, 1).text == "b"
    assert result.page(2).regions[0].bbox == BoundingBox.normalized(0, 0, 1, 1)
    assert result.page(3).regions == () and result.page(3).status == "partial"
    done = from_page_outcomes([plain], revision="r", method="custom")
    assert done.complete and done.page(2).regions[0].method == "custom"
