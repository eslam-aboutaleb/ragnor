"""Hand-computed unit tests for benchmark metrics (EX-02).

Cases required by the issue: perfect, missed-region, split/merge,
wrong-value and zero-ground-truth inputs with stable expected scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_extraction.manifest import (
    BBox,
    CriticalField,
    PageAnnotation,
    Region,
    Table,
    TableCell,
)
from benchmarks.pdf_extraction.metrics import (
    cell_value_accuracy,
    char_error_rate,
    field_accuracy,
    merged_cell_fidelity,
    ocr_coverage,
    reading_order_accuracy,
    region_recall,
    score_page,
    table_structure_score,
    word_error_rate,
)


def region(rid: str, text: str, order: int = 0, kind: str = "text") -> Region:
    return Region(rid, kind, BBox(0, 0, 10, 10), order, expected_text=text)


def page(regions, tables=(), fields=()) -> PageAnnotation:
    return PageAnnotation(
        document_id="d",
        file="documents/d.pdf",
        family="test",
        page=1,
        page_count=1,
        width=100,
        height=100,
        license="synthetic-generated-cc0",
        source_hash="sha256:x",
        regions=list(regions),
        tables=list(tables),
        critical_fields=list(fields),
    )


def simple_table(cells) -> Table:
    return Table("t1", 2, 2, cells)


class TestErrorRates:
    def test_perfect_extraction_is_zero(self):
        assert char_error_rate("hello world", "hello world") == 0.0
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_whitespace_normalization_is_free(self):
        assert char_error_rate("hello world", "hello\n  world") == 0.0

    def test_hand_computed_cer(self):
        # "kitten" -> "sitting": 3 substitutions/edits over 6 chars
        assert char_error_rate("kitten", "sitting") == 3 / 6

    def test_hand_computed_wer(self):
        # one substitution and one insertion over 3 words
        assert word_error_rate("the cat sat", "the dog sat down") == 2 / 3

    def test_zero_ground_truth_is_not_scored(self):
        assert char_error_rate("", "anything") is None
        assert word_error_rate("  ", "anything") is None


class TestRegions:
    def test_perfect_recall(self):
        regions = [region("r1", "alpha beta", 0), region("r2", "gamma delta", 1)]
        assert region_recall(regions, "alpha beta gamma delta") == 1.0

    def test_missed_region(self):
        regions = [region("r1", "alpha beta", 0), region("r2", "gamma delta", 1)]
        assert region_recall(regions, "alpha beta") == 0.5

    def test_zero_ground_truth_regions(self):
        assert region_recall([], "text") is None

    def test_extractor_spacing_differences_are_tolerated(self):
        regions = [region("r1", "Total due 1,250.00", 0)]
        assert region_recall(regions, "Total due\n1,250.00") == 1.0
        assert region_recall(regions, "Totaldue1,250.00") == 1.0

    def test_ocr_coverage_weights_characters(self):
        regions = [region("r1", "ab", 0), region("r2", "cdefgh", 1)]
        # recovered mass 2 of 8 characters
        assert ocr_coverage(regions, "ab") == 0.25

    def test_reading_order_pairs(self):
        regions = [
            region("r1", "first", 0),
            region("r2", "second", 1),
            region("r3", "third", 2),
        ]
        assert reading_order_accuracy(regions, "first second third") == 1.0
        assert reading_order_accuracy(regions, "third second first") == 0.0

    def test_reading_order_unscorable_without_pairs(self):
        regions = [region("r1", "only", 0), region("r2", "missing", 1)]
        assert reading_order_accuracy(regions, "only") is None


class TestTables:
    def test_identical_structure_scores_one(self):
        cells = [TableCell(0, 0, "a"), TableCell(0, 1, "b")]
        assert table_structure_score(simple_table(cells), simple_table(cells)) == 1.0

    def test_split_cell_changes_structure(self):
        expected = simple_table([TableCell(0, 0, "a", colspan=2)])
        actual = simple_table([TableCell(0, 0, "a"), TableCell(0, 1, "a")])
        score = table_structure_score(expected, actual)
        assert 0.0 < score < 1.0

    def test_merged_cell_fidelity(self):
        expected = simple_table([TableCell(0, 0, "a", colspan=2), TableCell(1, 0, "b")])
        matching = simple_table([TableCell(0, 0, "a", colspan=2), TableCell(1, 0, "b")])
        split = simple_table([TableCell(0, 0, "a"), TableCell(0, 1, "a"), TableCell(1, 0, "b")])
        assert merged_cell_fidelity(expected, matching) == 1.0
        assert merged_cell_fidelity(expected, split) == 0.0

    def test_merged_cell_fidelity_none_without_merges(self):
        plain = simple_table([TableCell(0, 0, "a")])
        assert merged_cell_fidelity(plain, plain) is None

    def test_cell_value_wrong_value(self):
        expected = simple_table([TableCell(0, 0, "100"), TableCell(0, 1, "200")])
        actual = simple_table([TableCell(0, 0, "100"), TableCell(0, 1, "999")])
        assert cell_value_accuracy(expected, actual, "") == 0.5

    def test_cell_value_without_structure_uses_text(self):
        expected = simple_table([TableCell(0, 0, "100"), TableCell(0, 1, "200")])
        assert cell_value_accuracy(expected, None, "100 200") == 1.0
        assert cell_value_accuracy(expected, None, "100") == 0.5


class TestFields:
    def test_exact_and_numeric_fields(self):
        p = page(
            [region("r1", "x", 0)],
            fields=[
                CriticalField("ref", "INV-2026-0417"),
                CriticalField("total", "1,250.00", numeric=True),
            ],
        )
        assert field_accuracy(p, "Ref INV-2026-0417 total 1250.00") == 1.0
        assert field_accuracy(p, "Ref INV-2026-0417 total 1250.01") == 0.5

    def test_no_fields_is_none(self):
        assert field_accuracy(page([region("r1", "x", 0)]), "x") is None


class TestPageScore:
    def test_silent_omission_flag(self):
        p = page([region("r1", "present", 0), region("r2", "vanished", 1)])
        score = score_page(p, "present")
        assert score.silent_omission is True
        assert score.regions_recovered == 1

    def test_empty_extraction_is_not_silent(self):
        p = page([region("r1", "scanned", 0)])
        score = score_page(p, "")
        assert score.silent_omission is False

    def test_perfect_page(self):
        p = page([region("r1", "everything here", 0)])
        score = score_page(p, "everything here")
        assert score.cer == 0.0
        assert score.region_recall == 1.0
        assert score.silent_omission is False
