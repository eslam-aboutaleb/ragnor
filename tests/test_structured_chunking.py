"""Tests for the structure-aware chunker (EX-06, issue #64).

Every test uses the real chunker with a deterministic exact tokenizer (one token
per whitespace-separated word plus one per table pipe-cell separator is not
assumed; the counter is plain ``len(text.split())`` and the chunker is held to
it). No part of the chunker is mocked.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ragnor.chunking import (
    OversizeUnitError,
    StructuredChunkingError,
    StructuredDocument,
    StructuredDocumentChunker,
    StructuredRegion,
    TableRow,
    chunker_registry,
    structured_from_pages,
)
from ragnor.chunking.markdown import MarkdownSectionChunker
from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.chunking.structured import markdown_to_structured
from ragnor.conformance import check_chunker
from ragnor.evidence.tokenizer import ApproximateTokenizer, ExactTokenizerAdapter
from ragnor.ingestion.extraction_types import (
    ExtractionCell,
    ExtractionTable,
    PageExtractionOutcome,
    PageExtractionStatus,
)
from ragnor.ingestion.versioning import build_snapshot, snapshot_matches


def words(text: str) -> int:
    return len(text.split())


EXACT = ExactTokenizerAdapter(words, name="whitespace-words")


def region(rid: str, kind: str, text: str = "", **kw):
    return StructuredRegion(rid, kind, text=text, **kw)


def table(rid: str, n_rows: int, *, page: int | None = 1, caption: str | None = None):
    return StructuredRegion(
        rid,
        "table",
        page=page,
        bbox=(0.1, 0.1, 0.9, 0.9),
        caption=caption,
        header=("Item", "Qty", "Price"),
        rows=tuple(TableRow((f"item{i}", str(i), f"{i}.00"), page=page) for i in range(n_rows)),
    )


def doc(*regions: StructuredRegion, fingerprint: str = "fp-1", access=None) -> StructuredDocument:
    return StructuredDocument("doc-1", fingerprint, tuple(regions), access or {})


def row_ids(chunks) -> list[int]:
    return [
        s["row"] for c in chunks for s in c.metadata["source_spans"] if s["kind"] == "table_row"
    ]


class TestTables:
    def test_simple_table_is_one_chunk_with_header_and_caption(self) -> None:
        d = doc(
            region("h", "heading", "Pricing", level=1),
            table("t1", 3),
            region("c1", "caption", "Table 1: Prices", caption_for="t1"),
        )
        chunks = StructuredDocumentChunker(EXACT, max_tokens=200, strict=True).chunk_document(d)
        assert len(chunks) == 1
        text = chunks[0].text
        assert "Table 1: Prices" in text
        assert "| Item | Qty | Price |" in text
        assert "| item0 | 0 | 0.00 |" in text and "| item2 | 2 | 2.00 |" in text
        assert chunks[0].metadata["kind"] == "table"
        assert chunks[0].metadata["section_path"] == ["Pricing"]
        assert "continuation" not in chunks[0].metadata

    def test_caption_before_or_after_both_bind_to_table(self) -> None:
        before = doc(region("c", "caption", "Cap", caption_for="t"), table("t", 2))
        after = doc(table("t", 2), region("c", "caption", "Cap", caption_for="t"))
        a = StructuredDocumentChunker(EXACT, max_tokens=100).chunk_document(before)
        b = StructuredDocumentChunker(EXACT, max_tokens=100).chunk_document(after)
        assert len(a) == len(b) == 1
        assert a[0].text == b[0].text
        roles = {s.get("role") for s in a[0].metadata["context_spans"]}
        assert "caption" in roles

    def test_hundred_row_table_partitions_with_repeated_header_and_caption(self) -> None:
        d = doc(
            region("h", "heading", "Inventory", level=2),
            table("t", 100, caption="Table 9: Stock"),
        )
        chunker = StructuredDocumentChunker(EXACT, max_tokens=60, strict=True)
        chunks = chunker.chunk_document(d)
        assert len(chunks) > 3
        # ordered, complete, no duplicate row attribution
        assert row_ids(chunks) == list(range(100))
        total = len(chunks)
        for index, chunk in enumerate(chunks):
            assert "Table 9: Stock" in chunk.text
            assert "| Item | Qty | Price |" in chunk.text
            assert chunk.text.startswith("Inventory")
            assert words(chunk.text) <= 60
            cont = chunk.metadata["continuation"]
            assert cont["part"] == index + 1 and cont["parts"] == total
            assert cont["continues_previous"] is (index > 0)
            assert cont["continues_next"] is (index < total - 1)
            # no row is cut: every row line is a complete 3-cell row
            body = [ln for ln in chunk.text.splitlines() if ln.startswith("| item")]
            assert all(ln.count("|") == 4 for ln in body)
            first, last = chunk.metadata["table"]["row_start"], chunk.metadata["table"]["row_end"]
            assert len(body) == last - first
        assert chunks[1].text.count("continued: part 2 of") == 1

    def test_header_context_is_not_counted_as_source_rows(self) -> None:
        d = doc(table("t", 30))
        chunks = StructuredDocumentChunker(EXACT, max_tokens=40, strict=True).chunk_document(d)
        for chunk in chunks:
            assert all(s["kind"] == "table_row" for s in chunk.metadata["source_spans"])
            assert any(s["role"] == "table_header" for s in chunk.metadata["context_spans"])

    def test_single_row_that_cannot_fit_is_never_split_silently(self) -> None:
        big = StructuredRegion(
            "t",
            "table",
            header=("A", "B"),
            rows=(TableRow(("x " * 50, "y")),),
        )
        with pytest.raises(OversizeUnitError):
            StructuredDocumentChunker(EXACT, max_tokens=20, strict=True).chunk_document(doc(big))
        loose = StructuredDocumentChunker(EXACT, max_tokens=20).chunk_document(doc(big))
        assert len(loose) == 1
        assert loose[0].metadata["exceeds_budget"] is True
        assert loose[0].text.count("x") == 50  # row intact

    def test_cell_bboxes_are_carried_per_row(self) -> None:
        t = StructuredRegion(
            "t",
            "table",
            page=2,
            header=("A", "B"),
            rows=(
                TableRow(
                    ("1", "2"),
                    page=2,
                    bbox=(0.1, 0.2, 0.9, 0.3),
                    cell_bboxes=((0.1, 0.2, 0.5, 0.3), (0.5, 0.2, 0.9, 0.3)),
                ),
            ),
        )
        (chunk,) = StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(doc(t))
        span = chunk.metadata["source_spans"][0]
        assert span["row"] == 0 and span["page"] == 2 and span["bbox"] == [0.1, 0.2, 0.9, 0.3]
        assert [c["col"] for c in span["cells"]] == [0, 1]

    def test_table_cells_escape_pipes_and_newlines(self) -> None:
        t = StructuredRegion(
            "t", "table", header=("A|B", "C"), rows=(TableRow(("x|y", "line1\nline2")),)
        )
        (chunk,) = StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(doc(t))
        assert "A\\|B" in chunk.text and "x\\|y" in chunk.text and "line1 line2" in chunk.text


class TestProvenance:
    def test_section_spanning_pages_keeps_every_region_with_its_own_citation(self) -> None:
        d = doc(
            region("h", "heading", "Results", level=1, page=1),
            region("p1", "paragraph", "First page text.", page=1, bbox=(0.1, 0.1, 0.9, 0.2)),
            region("p2", "paragraph", "Second page text.", page=2, bbox=(0.1, 0.3, 0.9, 0.4)),
            table("t", 2, page=2),
        )
        chunks = StructuredDocumentChunker(EXACT, max_tokens=100, strict=True).chunk_document(d)
        text_chunk = next(c for c in chunks if c.metadata["kind"] != "table")
        spans = text_chunk.metadata["source_spans"]
        assert [s["region_id"] for s in spans] == ["p1", "p2"]
        assert spans[0]["page"] == 1 and spans[0]["bbox"] == [0.1, 0.1, 0.9, 0.2]
        assert spans[1]["page"] == 2 and spans[1]["bbox"] == [0.1, 0.3, 0.9, 0.4]
        assert text_chunk.metadata["pages"] == [1, 2]
        table_chunk = next(c for c in chunks if c.metadata["kind"] == "table")
        # mixed page + region + cell citation: region, row index and page
        row = table_chunk.metadata["source_spans"][0]
        assert (row["region_id"], row["row"], row["page"]) == ("t", 0, 2)
        # heading is context for both, listed separately from exact sources
        assert any(s["region_id"] == "h" for s in text_chunk.metadata["context_spans"])
        assert all(s["region_id"] != "h" for s in text_chunk.metadata["source_spans"])

    def test_oversized_paragraph_split_keeps_one_exact_region(self) -> None:
        d = doc(region("p", "paragraph", " ".join(f"w{i}" for i in range(100)), page=3))
        chunks = StructuredDocumentChunker(EXACT, max_tokens=30, strict=True).chunk_document(d)
        assert len(chunks) >= 4
        got = " ".join(c.text for c in chunks).split()
        assert got == [f"w{i}" for i in range(100)]
        for index, chunk in enumerate(chunks):
            (span,) = chunk.metadata["source_spans"]
            assert span["region_id"] == "p" and span["page"] == 3
            assert span["part"] == index and span["parts"] == len(chunks)

    def test_figure_and_formula_stay_atomic_with_caption(self) -> None:
        d = doc(
            region(
                "f",
                "figure",
                "Bar chart of sales",
                page=4,
                bbox=(0, 0, 0.5, 0.5),
                assets={"crop": "s3://bucket/crops/f.png"},
            ),
            region("fc", "caption", "Figure 2", caption_for="f"),
            region("e", "formula", "E = mc^2", page=4),
        )
        chunks = StructuredDocumentChunker(EXACT, max_tokens=100).chunk_document(d)
        assert len(chunks) == 1
        text = chunks[0].text
        assert text.index("Figure 2") < text.index("Bar chart") < text.index("E = mc^2")
        ids = [s["region_id"] for s in chunks[0].metadata["source_spans"]]
        assert ids == ["f", "fc", "e"]
        assert chunks[0].metadata["storage_refs"] == {"f": {"crop": "s3://bucket/crops/f.png"}}

    def test_atomic_figure_is_not_split_across_chunks(self) -> None:
        d = doc(
            region("p", "paragraph", "word " * 20),
            region("f", "figure", "a figure description with several words"),
        )
        chunks = StructuredDocumentChunker(EXACT, max_tokens=25).chunk_document(d)
        figure_chunks = [c for c in chunks if "figure description" in c.text]
        assert len(figure_chunks) == 1
        assert "a figure description with several words" in figure_chunks[0].text

    def test_assets_are_references_not_bytes(self) -> None:
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("f", "figure", assets={"crop": b"\x89PNG"})  # type: ignore[dict-item]
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("f", "figure", assets={"crop": "x" * 5000})
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("f", "figure", assets={"blob": "s3://x"})

    def test_table_asset_refs_and_access_metadata_are_preserved(self) -> None:
        t = StructuredRegion(
            "t",
            "table",
            header=("A",),
            rows=(TableRow(("1",)),),
            assets={"html": "s3://b/t.html", "json": "s3://b/t.json"},
        )
        d = doc(t, access={"tenant_id": "acme", "acl": "finance"})
        (chunk,) = StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(d)
        assert chunk.metadata["access"] == {"tenant_id": "acme", "acl": "finance"}
        assert chunk.metadata["storage_refs"]["t"] == {
            "html": "s3://b/t.html",
            "json": "s3://b/t.json",
        }
        assert "retrieval_projection" in chunk.metadata
        assert chunk.metadata["retrieval_projection"]["header"] == ["A"]

    def test_heading_only_section_is_kept_and_nested_paths_are_exact(self) -> None:
        d = doc(
            region("h1", "heading", "Part", level=1),
            region("h2", "heading", "Chapter", level=2),
            region("p", "paragraph", "Body.", page=1),
            region("h3", "heading", "Empty", level=2),
            region("h4", "heading", "Next", level=1),
            region("q", "paragraph", "More."),
        )
        chunks = StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(d)
        paths = [c.metadata["section_path"] for c in chunks]
        assert paths == [["Part", "Chapter"], ["Part", "Empty"], ["Next"]]
        assert chunks[1].metadata["kind"] == "heading"

    def test_content_before_first_heading_has_empty_path(self) -> None:
        d = doc(region("p", "paragraph", "Preamble"), region("h", "heading", "A", level=1))
        chunks = StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(d)
        assert chunks[0].metadata["section_path"] == []
        assert chunks[0].text == "Preamble"


class TestBudget:
    def test_strict_requires_exact_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="exact tokenizer"):
            StructuredDocumentChunker(ApproximateTokenizer(), max_tokens=100, strict=True)
        with pytest.raises(ValueError, match="exact tokenizer"):
            StructuredDocumentChunker(max_tokens=100, strict=True)

    def test_approximate_mode_never_claims_exactness(self) -> None:
        chunker = StructuredDocumentChunker(max_tokens=100)
        (chunk,) = chunker.chunk_document(doc(region("p", "paragraph", "hello world")))
        assert chunk.metadata["token_count_exact"] is False
        assert chunker.config["is_exact"] is False and chunker.config["strict"] is False
        assert chunker.config["unit"] == "approximate_tokens"

    def test_strict_chunks_never_exceed_the_limit(self) -> None:
        d = doc(
            region("h", "heading", "Very long section title for budget", level=1),
            *[region(f"p{i}", "paragraph", " ".join(["tok"] * (7 + i))) for i in range(30)],
            table("t", 40, caption="A caption for the table"),
        )
        for limit in (45, 60, 90):
            chunks = StructuredDocumentChunker(EXACT, max_tokens=limit, strict=True).chunk_document(
                d
            )
            assert chunks
            assert all(words(c.text) <= limit for c in chunks)
            assert all(c.metadata["token_count"] == words(c.text) for c in chunks)
            assert all(c.metadata["token_count_exact"] for c in chunks)

    def test_budget_and_tokenizer_overrides_on_call(self) -> None:
        d = doc(region("p", "paragraph", " ".join(["a"] * 40)))
        chunker = StructuredDocumentChunker(EXACT, max_tokens=1000)
        chunks = chunker.chunk_document(d, token_budget=10)
        assert len(chunks) == 4
        assert all(c.metadata["max_tokens"] == 10 for c in chunks)
        with pytest.raises(ValueError, match="exact tokenizer"):
            StructuredDocumentChunker(EXACT, max_tokens=10, strict=True).chunk_document(
                d, tokenizer=ApproximateTokenizer()
            )

    def test_invalid_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            StructuredDocumentChunker(EXACT, max_tokens=0)

    def test_strict_unfittable_word_and_atomic_unit_raise(self) -> None:
        char_count = ExactTokenizerAdapter(len, name="chars")
        with pytest.raises(OversizeUnitError):
            StructuredDocumentChunker(char_count, max_tokens=5, strict=True).chunk_document(
                doc(region("p", "paragraph", "supercalifragilistic"))
            )
        with pytest.raises(OversizeUnitError):
            StructuredDocumentChunker(EXACT, max_tokens=3, strict=True).chunk_document(
                doc(region("f", "figure", "one two three four five"))
            )
        (loose,) = StructuredDocumentChunker(EXACT, max_tokens=3).chunk_document(
            doc(region("f", "figure", "one two three four five"))
        )
        assert loose.metadata["exceeds_budget"] is True

    def test_empty_document_has_no_chunks(self) -> None:
        assert StructuredDocumentChunker(EXACT, max_tokens=10).chunk_document(doc()) == []


class TestValidation:
    def test_bad_regions_are_rejected(self) -> None:
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("", "paragraph")
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("x", "banner")
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("x", "heading", text="t", level=0)
        with pytest.raises(StructuredChunkingError):
            StructuredRegion("x", "table")
        with pytest.raises(StructuredChunkingError):
            StructuredDocument("", "fp", ())
        with pytest.raises(StructuredChunkingError):
            doc(region("a", "paragraph"), region("a", "paragraph"))

    def test_caption_must_name_an_atomic_region(self) -> None:
        bad = doc(region("p", "paragraph", "x"), region("c", "caption", "y", caption_for="p"))
        with pytest.raises(StructuredChunkingError):
            StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(bad)
        missing = doc(region("c", "caption", "y", caption_for="nope"))
        with pytest.raises(StructuredChunkingError):
            StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(missing)
        twice = doc(
            table("t", 1),
            region("c1", "caption", "a", caption_for="t"),
            region("c2", "caption", "b", caption_for="t"),
        )
        with pytest.raises(StructuredChunkingError):
            StructuredDocumentChunker(EXACT, max_tokens=50).chunk_document(twice)

    def test_errors_do_not_leak_document_content(self) -> None:
        secret = "TOP-SECRET-CELL"
        big = StructuredRegion("t", "table", header=("A",), rows=(TableRow((secret * 30,)),))
        with pytest.raises(OversizeUnitError) as info:
            StructuredDocumentChunker(
                ExactTokenizerAdapter(len, name="chars"), max_tokens=20, strict=True
            ).chunk_document(doc(big))
        assert secret not in str(info.value)


# ---------------------------------------------------------------------------
# Identity, idempotency and re-index
# ---------------------------------------------------------------------------


class TestIdentity:
    def _doc(self, fingerprint: str = "fp-1") -> StructuredDocument:
        return doc(
            region("h", "heading", "S", level=1),
            region("p", "paragraph", "alpha beta gamma"),
            table("t", 25),
            fingerprint=fingerprint,
        )

    def test_same_input_gives_identical_ids_and_text(self) -> None:
        a = StructuredDocumentChunker(EXACT, max_tokens=30).chunk_document(self._doc())
        b = StructuredDocumentChunker(EXACT, max_tokens=30).chunk_document(self._doc())
        assert [c.id for c in a] == [c.id for c in b]
        assert [c.text for c in a] == [c.text for c in b]
        assert len({c.id for c in a}) == len(a)

    @pytest.mark.parametrize(
        "other",
        [
            {"max_tokens": 31},
            {"max_tokens": 30, "strict": True},
            {"max_tokens": 30, "tokenizer": ExactTokenizerAdapter(words, name="other-tokenizer")},
        ],
    )
    def test_option_change_changes_every_chunk_id_and_snapshot(self, other) -> None:
        base = StructuredDocumentChunker(EXACT, max_tokens=30)
        kwargs = {"max_tokens": 30, **other}
        tok = kwargs.pop("tokenizer", EXACT)
        changed = StructuredDocumentChunker(tok, **kwargs)
        ids_a = {c.id for c in base.chunk_document(self._doc())}
        ids_b = {c.id for c in changed.chunk_document(self._doc())}
        assert ids_a.isdisjoint(ids_b)
        settings_obj = type("S", (), {"embedding_model": "text-embedding-3-small"})()
        snap_a = build_snapshot("h", settings_obj, base)  # type: ignore[arg-type]
        snap_b = build_snapshot("h", settings_obj, changed)  # type: ignore[arg-type]
        assert snapshot_matches(snap_a, snap_a) is True
        assert snapshot_matches(snap_b, snap_a) is False  # forces explicit re-index

    def test_source_fingerprint_change_changes_ids(self) -> None:
        chunker = StructuredDocumentChunker(EXACT, max_tokens=30)
        a = {c.id for c in chunker.chunk_document(self._doc("fp-1"))}
        b = {c.id for c in chunker.chunk_document(self._doc("fp-2"))}
        assert a.isdisjoint(b)

    def test_structure_change_changes_only_affected_ids(self) -> None:
        chunker = StructuredDocumentChunker(EXACT, max_tokens=30)
        base = self._doc()
        edited = doc(
            region("h", "heading", "S", level=1),
            region("p", "paragraph", "alpha beta gamma"),
            table("t", 25),
            region("extra", "paragraph", "new trailing paragraph"),
        )
        ids_base = [c.id for c in chunker.chunk_document(base)]
        ids_edit = [c.id for c in chunker.chunk_document(edited)]
        # fingerprint is the same here by construction, so unchanged boundaries
        # keep identity (stable) and only the added unit is new
        assert set(ids_base) <= set(ids_edit)
        assert len(ids_edit) == len(ids_base) + 1

    def test_version_is_in_snapshot_hash_inputs(self) -> None:
        chunker = StructuredDocumentChunker(EXACT, max_tokens=30)
        assert chunker.version == "structured-v1"
        assert chunker.config["chunk_size"] == 30 and chunker.config["overlap"] == 0

    def test_ids_do_not_depend_on_access_metadata(self) -> None:
        chunker = StructuredDocumentChunker(EXACT, max_tokens=30)
        plain = doc(table("t", 2))
        tenant = doc(table("t", 2), access={"tenant_id": "a"})
        assert [c.id for c in chunker.chunk_document(plain)] == [
            c.id for c in chunker.chunk_document(tenant)
        ]


# ---------------------------------------------------------------------------
# Metamorphic / property tests
# ---------------------------------------------------------------------------

_WORD = st.text(
    alphabet=st.sampled_from(list("abcdefgh") + ["é", "ß", "世", "界", "ع", "ر", "ب", "я", "5"]),
    min_size=1,
    max_size=6,
)
_PARA = st.lists(_WORD, min_size=1, max_size=40).map(" ".join)
_ROWS = st.lists(st.tuples(_WORD, _WORD), min_size=1, max_size=40)


@st.composite
def documents(draw, *, heading_shift: int = 0):
    regions: list[StructuredRegion] = []
    n_sections = draw(st.integers(1, 4))
    paragraphs: list[str] = []
    rows: list[tuple[str, str]] = []
    for s in range(n_sections):
        level = draw(st.integers(1, 3))
        regions.append(
            StructuredRegion(f"h{s}", "heading", text=f"Sec{s}", level=level + heading_shift)
        )
        for k in range(draw(st.integers(0, 3))):
            text = draw(_PARA)
            paragraphs.append(text)
            regions.append(StructuredRegion(f"p{s}-{k}", "paragraph", text=text, page=1 + k))
        if draw(st.booleans()):
            body = draw(_ROWS)
            rows.extend((f"s{s}{a}", b) for a, b in body)
            regions.append(
                StructuredRegion(
                    f"t{s}",
                    "table",
                    header=("Key", "Val"),
                    rows=tuple(TableRow((f"s{s}{a}", b)) for a, b in body),
                )
            )
    return StructuredDocument("m", "fp", tuple(regions)), paragraphs, rows


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=documents(), limit=st.integers(25, 90))
def test_no_paragraph_words_or_table_rows_are_lost(data, limit) -> None:
    structured, paragraphs, rows = data
    chunker = StructuredDocumentChunker(EXACT, max_tokens=limit, strict=True)
    chunks = []
    try:
        chunks = chunker.chunk_document(structured)
    except OversizeUnitError:
        pytest.skip("generated unit larger than the strict budget")
    assert all(words(c.text) <= limit for c in chunks)
    # paragraphs: every word appears within the chunks attributed to that region
    for region_ in structured.regions:
        if region_.kind != "paragraph":
            continue
        pieces = [
            c.text
            for c in chunks
            if any(s["region_id"] == region_.region_id for s in c.metadata["source_spans"])
        ]
        assert pieces, "paragraph lost"
        body = " ".join(p.split("\n\n", 1)[-1] if "\n\n" in p else p for p in pieces).split()
        # strip repeated section prefix words that may follow the first newline
        assert all(w in body for w in region_.text.split())
    # table rows: each row attributed exactly once, in order, per table
    for region_ in structured.regions:
        if region_.kind != "table":
            continue
        attributed = [
            s["row"]
            for c in chunks
            for s in c.metadata["source_spans"]
            if s["region_id"] == region_.region_id and s["kind"] == "table_row"
        ]
        assert attributed == list(range(len(region_.rows)))
    # ids unique
    assert len({c.id for c in chunks}) == len(chunks)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=documents(), limit=st.integers(40, 90))
def test_heading_level_shift_keeps_content_and_paths(data, limit) -> None:
    structured, _, _ = data
    if any(r.kind == "heading" and r.level > 4 for r in structured.regions):
        return
    shifted = StructuredDocument(
        structured.document_id,
        structured.fingerprint,
        tuple(
            StructuredRegion(r.region_id, "heading", text=r.text, level=r.level + 2)
            if r.kind == "heading"
            else r
            for r in structured.regions
        ),
    )
    chunker = StructuredDocumentChunker(EXACT, max_tokens=limit)
    a, b = chunker.chunk_document(structured), chunker.chunk_document(shifted)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.metadata["section_path"] for c in a] == [c.metadata["section_path"] for c in b]


def test_reordering_regions_inside_a_section_loses_nothing() -> None:
    paras = [region(f"p{i}", "paragraph", f"unique{i} " + "filler " * 5) for i in range(6)]
    tbl = table("t", 12)
    forward = doc(region("h", "heading", "S", level=1), *paras, tbl)
    shuffled = doc(region("h", "heading", "S", level=1), tbl, *reversed(paras))
    chunker = StructuredDocumentChunker(EXACT, max_tokens=40, strict=True)

    def inventory(d):
        chunks = chunker.chunk_document(d)
        text = " ".join(c.text for c in chunks)
        return {w for w in text.split() if w.startswith("unique")}, sorted(row_ids(chunks))

    assert inventory(forward) == inventory(shuffled)
    assert inventory(forward)[1] == list(range(12))


def test_multilingual_and_nested_large_table() -> None:
    t = StructuredRegion(
        "t",
        "table",
        header=("الاسم", "名称", "Цена"),
        rows=tuple(TableRow((f"عنصر{i}", f"物{i}", f"{i},50")) for i in range(60)),
    )
    d = doc(
        region("a", "heading", "الفصل الأول", level=1),
        region("b", "heading", "第二节", level=2),
        region("c", "heading", "Подраздел", level=3),
        t,
    )
    chunks = StructuredDocumentChunker(EXACT, max_tokens=50, strict=True).chunk_document(d)
    assert row_ids(chunks) == list(range(60))
    assert all(c.metadata["section_path"] == ["الفصل الأول", "第二节", "Подраздел"] for c in chunks)
    assert all("الاسم" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# Compatibility with the existing Chunker API and registry
# ---------------------------------------------------------------------------

MD = """# Guide

Intro paragraph.

## Prices

| Item | Qty |
| --- | --- |
| apple | 3 |
| pear | 5 |

## Notes

```
# not a heading
| a | b |
```

Closing words.
"""


class TestMarkdownFallbackAndCompat:
    def test_chunk_text_api_satisfies_chunker_contract(self) -> None:
        chunker = StructuredDocumentChunker(EXACT, max_tokens=60)
        check_chunker(chunker, MD)

    def test_markdown_fallback_keeps_table_with_header(self) -> None:
        chunks = StructuredDocumentChunker(EXACT, max_tokens=60).chunk(MD)
        price = next(c for c in chunks if "apple" in c.text)
        assert "| Item | Qty |" in price.text and "| pear | 5 |" in price.text
        assert price.metadata["section_path"] == ["Guide", "Prices"]

    def test_markdown_fence_is_not_parsed(self) -> None:
        d = markdown_to_structured(MD)
        kinds = [r.kind for r in d.regions]
        assert kinds.count("table") == 1
        assert [r.text for r in d.regions if r.kind == "heading"] == ["Guide", "Prices", "Notes"]
        fenced = next(r for r in d.regions if "not a heading" in r.text)
        assert fenced.kind == "paragraph" and "| a | b |" in fenced.text

    def test_sliding_window_would_split_what_structured_keeps(self) -> None:
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(80))
        md = f"## T\n\n| K | V |\n| --- | --- |\n{rows}\n"
        old = MarkdownSectionChunker(SlidingWindowChunker(chunk_size=30, overlap=5)).chunk(md)
        new = StructuredDocumentChunker(EXACT, max_tokens=30, strict=True).chunk(md)
        assert any("| K | V |" not in c.text for c in old)  # header dropped by old chunker
        assert all("| K | V |" in c.text for c in new)  # header repeated by new one
        assert row_ids(new) == list(range(80))

    def test_registry_name(self) -> None:
        assert chunker_registry.get("structured") is StructuredDocumentChunker

    def test_existing_chunkers_unchanged(self) -> None:
        assert MarkdownSectionChunker(SlidingWindowChunker()).version == "section-aware-v1"

    def test_pages_adapter_maps_tables_with_cell_boxes(self) -> None:
        table_ = ExtractionTable(
            (0.1, 0.1, 0.9, 0.9),
            3,
            2,
            (
                ExtractionCell(0, 0, "Name", (0.1, 0.1, 0.5, 0.2)),
                ExtractionCell(0, 1, "Qty", (0.5, 0.1, 0.9, 0.2)),
                ExtractionCell(1, 0, "bolt", (0.1, 0.2, 0.5, 0.3)),
                ExtractionCell(1, 1, "4", (0.5, 0.2, 0.9, 0.3)),
                ExtractionCell(2, 0, "nut", (0.1, 0.3, 0.5, 0.4)),
                ExtractionCell(2, 1, "9", (0.5, 0.3, 0.9, 0.4)),
            ),
        )
        pages = [
            PageExtractionOutcome(
                2,
                PageExtractionStatus.EXTRACTED,
                text="Parts list.\n\nSecond block.",
                tables=(table_,),
            ),
            PageExtractionOutcome(1, PageExtractionStatus.EXTRACTED, text="Cover page."),
        ]
        d = structured_from_pages(
            pages, document_id="d", fingerprint="f", access={"tenant_id": "t"}
        )
        assert [r.region_id for r in d.regions] == [
            "p1-text0",
            "p2-text0",
            "p2-text1",
            "p2-table0",
        ]
        chunks = StructuredDocumentChunker(EXACT, max_tokens=80, strict=True).chunk_document(d)
        tbl = next(c for c in chunks if c.metadata["kind"] == "table")
        assert "| Name | Qty |" in tbl.text and "| nut | 9 |" in tbl.text
        spans = tbl.metadata["source_spans"]
        assert [s["row"] for s in spans] == [0, 1]
        assert spans[0]["page"] == 2 and spans[0]["cells"][1]["bbox"] == [0.5, 0.2, 0.9, 0.3]
        assert tbl.metadata["access"] == {"tenant_id": "t"}
