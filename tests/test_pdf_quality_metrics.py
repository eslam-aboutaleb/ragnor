"""Hand-computed and adversarial checks for extraction quality, not just text length."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from benchmarks.quality_metrics import (
    TableParser,
    layout_scores,
    match_boxes,
    page_text_scores,
    table_scores,
    validate_bbox,
)
from benchmarks.quality_providers import QualityPage
from benchmarks.run_quality import aggregate, score_page, validate_manifest

ONE = "<table><tr><td>10</td></tr></table>"
TWO = "<table><tr><td>1</td><td>2</td></tr></table>"
HAS_APTED = importlib.util.find_spec("apted") is not None


@pytest.mark.skipif(
    not HAS_APTED, reason="Install benchmarks/requirements.txt for structural metrics"
)
def test_teds_perfect_wrong_value_and_swapped_cells():
    perfect = table_scores(ONE, ONE)
    assert all(value == 1 for value in perfect.values())
    wrong = table_scores(ONE, ONE.replace("10", "90"))
    assert wrong["teds_normalized"] == pytest.approx(1 - 0.5 / 3)
    assert wrong["teds_structure"] == 1
    assert wrong["cell_exact_accuracy"] == 0
    assert wrong["numeric_cell_accuracy"] == 0
    swapped = table_scores(TWO, TWO.replace("<td>1</td><td>2</td>", "<td>2</td><td>1</td>"))
    assert swapped["teds_structure"] == 1
    assert swapped["teds_normalized"] == 0.5
    assert swapped["numeric_cell_accuracy"] == 0
    # A bag-of-values metric alone would incorrectly call the swap perfect.
    assert page_text_scores("1 2", "2 1")["numeric_token_recall"] == 1


@pytest.mark.skipif(
    not HAS_APTED, reason="Install benchmarks/requirements.txt for structural metrics"
)
def test_teds_detects_merged_cell_loss():
    merged = '<table><tr><td colspan="2">1</td></tr></table>'
    score = table_scores(merged, TWO)
    assert score["teds_structure"] == 0.5
    assert score["span_accuracy"] == 0


@pytest.mark.parametrize(
    "html",
    [
        '<table><tr><td rowspan="0">x</td></tr></table>',
        '<table><tr><td rowspan="2">x</td></tr></table>',
        '<table><tr><td colspan="-1">x</td></tr></table>',
        "<table><tr><td>x",
        "<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>",
    ],
)
def test_malformed_spans_and_tables_rejected(html):
    with pytest.raises(ValueError):
        TableParser(html)


def test_layout_omission_and_pair_coverage_cannot_hide_a_lost_column():
    regions = [
        {"bbox": [i / 3, 0, (i + 1) / 3, 1], "text": value}
        for i, value in enumerate(["left", "middle", "right"])
    ]
    words = [
        {"text": "middle", "bbox": [0.4, 0.2, 0.5, 0.3]},
        {"text": "left", "bbox": [0.1, 0.2, 0.2, 0.3]},
    ]
    scores = layout_scores(regions, words)
    assert scores["empty_text_regions"] == 1
    assert scores["region_token_recall"] == pytest.approx(2 / 3)
    assert scores["reading_order_pair_accuracy"] == 0
    assert scores["reading_order_pair_coverage"] == pytest.approx(1 / 3)
    empty = layout_scores(regions, [])
    assert empty["reading_order_pair_accuracy"] is None
    assert empty["reading_order_pair_coverage"] == 0
    assert empty["region_token_recall"] == 0


def test_box_matching_deduplicates_and_enforces_overlap():
    reference = [{"bbox": [0, 0, 0.4, 0.4]}, {"bbox": [0.6, 0.6, 1, 1]}]
    actual = [{"bbox": [0, 0, 0.4, 0.4]}, {"bbox": [0, 0, 0.4, 0.4]}]
    assert len(match_boxes(reference, actual)) == 1


@pytest.mark.parametrize(
    "bbox", [[0, 0, 2, 1], [0.5, 0, 0.1, 1], [0, 0, float("nan"), 1], [0, 0, 0, 1]]
)
def test_invalid_geometry_rejected(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


def reference_page():
    return {
        "page": 1,
        "expected_text": "10",
        "regions": [
            {"id": "t1", "category": "Table", "text": "10", "bbox": [0, 0, 1, 1], "html": ONE}
        ],
    }


def test_missing_table_and_failed_page_stay_in_denominator():
    reference = reference_page()
    empty = QualityPage(1, "", "no_text", [], [])
    row = score_page(reference, empty)
    failed = score_page(reference, None, layout_supported=True, tables_supported=True)
    for result in [row, failed]:
        summary = aggregate([result])
        assert summary["mean_teds_normalized"] == 0
        assert summary["mean_cell_exact_accuracy"] == 0
        assert summary["mean_numeric_cell_accuracy"] == 0
        assert summary["table_recall_iou50"] == 0
        assert summary["mean_region_token_recall"] == 0
    assert aggregate([failed])["missing_pages"] == 1


def test_invalid_predicted_table_gets_zero_not_excluded():
    row = score_page(
        reference_page(),
        QualityPage(
            1,
            "10",
            "extracted",
            [],
            [{"bbox": [0, 0, 1, 1], "html": '<table><tr><td rowspan="0">10</td></tr></table>'}],
        ),
    )
    assert aggregate([row])["mean_teds_normalized"] == 0
    assert aggregate([row])["invalid_table_predictions"] == 1


def test_unsupported_table_capability_is_never_a_passing_score():
    summary = aggregate([score_page(reference_page(), QualityPage(1, "10", "extracted"))])
    assert summary["mean_teds_normalized"] is None
    assert summary["table_recall_iou50"] is None
    assert summary["mean_page_cer"] == 0


@pytest.mark.parametrize("defect", ["checksum", "pages", "region", "bbox", "escape"])
def test_quality_manifest_rejects_corruption(tmp_path: Path, defect):
    import hashlib

    data = b"fixture"
    (tmp_path / "one.pdf").write_bytes(data)
    page = reference_page()
    doc = {
        "document_id": "one",
        "path": "one.pdf",
        "source_hash": hashlib.sha256(data).hexdigest(),
        "family": "public",
        "pages": [page],
    }
    if defect == "checksum":
        doc["source_hash"] = "wrong"
    elif defect == "pages":
        page["page"] = 2
    elif defect == "region":
        page["regions"].append(dict(page["regions"][0]))
    elif defect == "bbox":
        page["regions"][0]["bbox"] = [0, 0, 2, 1]
    else:
        doc["path"] = "../escape.pdf"
    manifest = tmp_path / "quality.json"
    manifest.write_text(json.dumps({"schema_version": 2, "documents": [doc]}))
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_comparison_rejects_omission_wrong_value_and_lost_capability():
    from copy import deepcopy
    from benchmarks.compare_quality import compare

    summary = aggregate([score_page(reference_page(), QualityPage(1, "10", "extracted"))])
    base = {
        "corpus_sha256": "fixed",
        "configuration": {"cohort": "all"},
        "summary": summary,
        "families": {"public": dict(summary)},
        "failures": [],
    }
    assert compare(base, deepcopy(base))["passed"]
    for key, value in [
        ("mean_token_recall", 0.0),
        ("mean_numeric_token_recall", 0.0),
        ("mean_numeric_token_recall", None),
        ("mean_page_cer", 0.1),
    ]:
        candidate = deepcopy(base)
        candidate["families"]["public"][key] = value
        assert not compare(base, candidate)["passed"]
    candidate = deepcopy(base)
    candidate["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="different corpus"):
        compare(base, candidate)


def test_invalid_prediction_cannot_produce_nonfinite_gate_scores():
    from benchmarks.compare_quality import compare

    summary = {"pages": 1, "mean_page_cer": 0.0}
    base = {
        "corpus_sha256": "fixed",
        "configuration": {"cohort": "all"},
        "summary": summary,
        "families": {},
        "failures": [],
    }
    candidate = {**base, "summary": {"pages": 1, "mean_page_cer": float("nan")}}
    assert not compare(base, candidate)["passed"]


def test_no_reference_regions_are_unmeasured_not_perfect_or_failed():
    scores = layout_scores([], [])
    assert scores["region_token_recall"] is None
    assert scores["reading_order_pair_accuracy"] is None
    assert scores["reading_order_pair_coverage"] is None
