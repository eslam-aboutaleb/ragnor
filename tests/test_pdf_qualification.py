"""Hand-computed metric and fail-closed benchmark qualification contracts."""

import copy
import hashlib
import json
from decimal import Decimal

import pytest

pytest.importorskip("sympy")
pytest.importorskip("PIL")

from benchmarks.cost_accounting import cost_report
from benchmarks.coverage_metrics import ink_coverage, region_detection
from benchmarks.qualification import qualify, reference_fingerprint
from benchmarks.reference_schema import validate_prediction, validate_references
from benchmarks.semantic_metrics import formula_equivalent, numeric_value, score_fields
from benchmarks.quality_providers import QualityPage
from benchmarks.compare_quality import compare


def test_region_detection_counts_missing_and_extra_blocks():
    refs = [
        {"id": "a", "category": "Paragraph", "bbox": [0, 0, 0.4, 0.4]},
        {"id": "b", "category": "Paragraph", "bbox": [0.6, 0.6, 1, 1]},
    ]
    found = [{"bbox": [0, 0, 0.4, 0.4]}, {"bbox": [0.5, 0, 0.9, 0.4]}]
    result = region_detection(refs, found)
    assert result["region_precision"] == result["region_recall"] == 0.5
    assert result["omitted_region_ids"] == ["b"]
    assert region_detection(refs, None)["region_recall"] is None
    assert region_detection(refs, [])["region_recall"] == 0


def test_ink_coverage_uses_foreground_pixels():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 19, 19], fill="black")
    draw.rectangle([70, 70, 79, 79], fill="black")
    result = ink_coverage(image, [{"bbox": [0.1, 0.1, 0.2, 0.2]}])
    assert result["foreground_pixels"] == 200
    assert result["word_box_ink_coverage"] == 0.5
    assert ink_coverage(Image.new("RGB", (10, 10), "white"), [])["word_box_ink_coverage"] is None


@pytest.mark.parametrize("raw,value", [("١٬٢٣٤٫٥", "1234.5"), ("12.5%", ".125"), ("-2", "-2")])
def test_numeric_semantics(raw, value):
    assert numeric_value(raw) == Decimal(value)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "12,34", "1e999"])
def test_ambiguous_or_unbounded_numbers_are_rejected(raw):
    with pytest.raises(ValueError):
        numeric_value(raw)


def test_fields_distinguish_wrong_missing_and_equivalent_formulas():
    fields = [
        {"id": "n", "kind": "number", "value": "100"},
        {"id": "f", "kind": "formula", "value": "(x+1)**2"},
        {"id": "m", "kind": "chart_value", "value": "2"},
    ]
    result = score_fields(fields, {"n": "10", "f": "x*x+2*x+1"})
    assert result["field_task_accuracy"] == 1 / 3
    assert result["field_answer_coverage"] == 2 / 3
    assert len(result["field_failures"]) == 2
    assert score_fields([], None)["field_task_accuracy"] is None
    assert score_fields(fields, None)["field_task_accuracy"] == 0
    assert not formula_equivalent("x+1", "x+2")


@pytest.mark.parametrize("formula", ["__import__('os')", "x.real", "x**100", "f(x)"])
def test_formula_parser_rejects_executable_or_unbounded_syntax(formula):
    with pytest.raises(ValueError):
        formula_equivalent("x", formula)


def test_cost_has_no_invented_host_price():
    assert cost_report(3600, 1000)["total_usd_per_1000_pages"] is None
    assert (
        cost_report(3600, 1000, cpu_usd_per_hour=2, pricing_source="host quote")[
            "total_usd_per_1000_pages"
        ]
        == 2
    )
    with pytest.raises(ValueError):
        cost_report(10, 1, cpu_usd_per_hour=2)
    with pytest.raises(ValueError):
        cost_report(float("nan"), 1)


def reference():
    return {
        "document_id": "d",
        "source_hash": "source",
        "license": "CC-BY-4.0",
        "pages": [
            {
                "page": 1,
                "expected_text": "abc",
                "regions": [{"id": "a", "category": "Paragraph", "bbox": [0, 0, 1, 1]}],
                "reading_order": ["a"],
            }
        ],
    }


def test_reference_contracts_reject_wrong_order_hash_and_caption():
    doc = reference()
    validate_references(doc)
    for change in [
        {"reading_order": []},
        {"caption_links": [{"caption_id": "a", "target_id": "a"}]},
        {"critical_fields": [{"id": "f", "kind": "number", "value": "NaN"}]},
    ]:
        candidate = copy.deepcopy(doc)
        candidate["pages"][0].update(change)
        with pytest.raises(ValueError):
            validate_references(candidate)
    doc["annotation_sha256"] = "incorrect"
    with pytest.raises(ValueError):
        validate_references(doc)
    with pytest.raises(ValueError):
        validate_prediction(QualityPage(1, "x", "ok", words=[{"bbox": [0, 0, 2, 1]}]))


def report():
    return {
        "schema_version": 3,
        "corpus_sha256": "corpus",
        "configuration": {"cohort": "all"},
        "families": {"scan": {"pages": 1, "mean_page_cer": 0.1}},
        "summary": {"pages": 1, "mean_page_cer": 0.1, "mean_field_task_accuracy": 1},
        "failures": [],
        "retrieval": {
            "total_queries": 1,
            "citation_resolution_rate": 1,
            "authorization_and_revision": {"passed": True},
        },
        "cost": {"total_usd_per_1000_pages": 1},
    }


def test_comparison_rejects_lost_semantics_and_changed_schema():
    baseline = report()
    candidate = copy.deepcopy(baseline)
    candidate["summary"]["mean_field_task_accuracy"] = None
    assert not compare(baseline, candidate)["passed"]
    candidate["schema_version"] = 2
    with pytest.raises(ValueError):
        compare(baseline, candidate)


def test_review_gate_requires_real_distinct_bound_reviews_and_approved_policy():
    doc = reference()
    manifest = {"documents": [doc]}
    base = report()
    policy = {
        "approved_by": "maintainer",
        "allowed_policy_approvers": ["maintainer"],
        "approved_at": "2026-10-29",
        "allowed_reviewers": ["reviewer"],
        "minimum_reviewed_pages": 1,
        "required_families": ["scan"],
        "required_summary_metrics": ["mean_page_cer"],
        "baseline_sha256": hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
    }
    review = {
        "key": "d:1",
        "status": "approved",
        "reviewer": "reviewer",
        "reviewed_at": "2026-10-29",
        "families": ["scan"],
        "reference_sha256": reference_fingerprint(doc, doc["pages"][0]),
    }
    reviews = {"corpus_sha256": "corpus", "reviews": [review] * 30}
    result = qualify(base, manifest, policy, reviews, baseline_report=base)
    assert not result["qualified"]
    gate = next(row for row in result["checks"] if row["gate"] == "human_annotation_review")
    assert gate["detail"]["valid_reviews"] == 1
    assert not qualify(base, manifest, {}, None)["qualified"]
    reviews["reviews"][0]["reference_sha256"] = "stale"
    result = qualify(base, manifest, policy, reviews, baseline_report=base)
    assert (
        next(row for row in result["checks"] if row["gate"] == "human_annotation_review")["detail"][
            "valid_reviews"
        ]
        == 0
    )


def test_image_import_is_reproducible_and_does_not_invent_geometry(tmp_path):
    import io
    from PIL import Image
    from benchmarks.extended_corpus import _image_document

    image = Image.new("RGB", (30, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    results = []
    for name in ["first", "second"]:
        output = tmp_path / name
        output.mkdir()
        results.append(
            _image_document(
                output,
                "sample",
                buffer.getvalue(),
                "reference",
                family="scan",
                provenance={"source": "test"},
                license_name="CC0",
            )
        )
    assert results[0]["source_hash"] == results[1]["source_hash"]
    assert results[0]["pages"][0]["geometry_ground_truth"] is False
    assert results[0]["pages"][0]["regions"] == []


def test_unsafe_verifier_claim_is_counted_separately():
    from benchmarks.run_quality import score_page, aggregate

    ref = {"page": 1, "expected_text": "important", "regions": []}
    row = score_page(ref, QualityPage(1, "", "verified", verified_complete=True))
    result = aggregate([row])
    assert result["unsafe_complete_pages"] == 1
    assert result["observed_verification_false_negative_rate"] == 1


def test_table_numeric_fields_use_cell_coordinates_and_missing_cells_fail():
    from benchmarks.run_quality import score_page

    ref = {
        "page": 1,
        "expected_text": "100 200",
        "regions": [
            {
                "id": "t",
                "category": "Table",
                "bbox": [0, 0, 1, 1],
                "html": "<table><tr><td>100</td><td>200</td></tr></table>",
            }
        ],
        "critical_fields": [
            {
                "id": "t:0:0",
                "kind": "number",
                "value": "100",
                "table_region_id": "t",
                "row": 0,
                "col": 0,
            },
            {
                "id": "t:0:1",
                "kind": "number",
                "value": "200",
                "table_region_id": "t",
                "row": 0,
                "col": 1,
            },
        ],
    }
    page = QualityPage(
        1,
        "100 200",
        "ok",
        tables=[{"bbox": [0, 0, 1, 1], "html": "<table><tr><td>200</td><td>100</td></tr></table>"}],
    )
    assert score_page(ref, page)["field_task_accuracy"] == 0
    page.tables = []
    assert score_page(ref, page)["field_answer_coverage"] == 0
    page.tables = [{"bbox": [0, 0, 1, 1], "html": ref["regions"][0]["html"]}]
    assert score_page(ref, page)["field_task_accuracy"] == 1


def test_qualification_accepts_complete_host_evidence_and_rejects_stale_page():
    manifest = {"documents": []}
    reviews = {"corpus_sha256": "corpus", "reviews": []}
    for i in range(30):
        doc = reference()
        doc["document_id"] = f"d{i}"
        manifest["documents"].append(doc)
        reviews["reviews"].append(
            {
                "key": f"d{i}:1",
                "reference_sha256": reference_fingerprint(doc, doc["pages"][0]),
                "status": "approved",
                "reviewer": "test-reviewer",
                "reviewed_at": "2026-10-29",
                "families": ["scan"],
            }
        )
    base = report()
    policy = {
        "approved_by": "test-maintainer",
        "allowed_policy_approvers": ["test-maintainer"],
        "approved_at": "2026-10-29",
        "allowed_reviewers": ["test-reviewer"],
        "required_families": ["scan"],
        "required_summary_metrics": ["mean_page_cer"],
        "baseline_sha256": hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
    }
    assert qualify(base, manifest, policy, reviews, baseline_report=base)["qualified"]
    manifest["documents"][0]["pages"][0]["expected_text"] = "changed"
    assert not qualify(base, manifest, policy, reviews, baseline_report=base)["qualified"]


def test_retrieval_regression_is_a_quality_gate():
    base = report()
    base["retrieval"]["mrr"] = 1
    candidate = copy.deepcopy(base)
    candidate["retrieval"]["mrr"] = 0
    assert not compare(base, candidate)["passed"]


def test_ocr_protocol_is_explicit_even_without_tessdata_config_files(tmp_path, monkeypatch):
    from benchmarks.quality_providers import TesseractProvider, parse_tsv

    for name in ["ara", "eng"]:
        (tmp_path / f"{name}.traineddata").write_bytes(b"test model")
    monkeypatch.setattr(TesseractProvider, "executable", staticmethod(lambda: "/test/tesseract"))
    provider = TesseractProvider(language="ara+eng", tessdata_dir=str(tmp_path))
    argv = provider.command()
    assert "tessedit_create_tsv=1" in argv
    assert "tessedit_create_txt=0" in argv
    assert "tsv" not in argv
    words, blocks = parse_tsv(
        'level\ttext\tleft\ttop\twidth\theight\tblock_num\n5\t"quoted\t10\t20\t20\t10\t1\n',
        100,
        100,
    )
    assert words == [{"text": '"quoted', "bbox": [0.1, 0.2, 0.3, 0.3]}]
    assert blocks == [{"bbox": [0.1, 0.2, 0.3, 0.3]}]
    with pytest.raises(ValueError, match="TSV"):
        parse_tsv("Unexpected plain text", 100, 100)


def test_packet_never_manufactures_review_or_policy_approval(tmp_path):
    import io
    from PIL import Image
    from benchmarks.extended_corpus import _image_document
    from benchmarks.qualification import prepare_packet

    pytest.importorskip("pypdfium2")
    buffer = io.BytesIO()
    Image.new("RGB", (30, 40), "white").save(buffer, format="PNG")
    doc = _image_document(
        tmp_path,
        "sample",
        buffer.getvalue(),
        "reference",
        family="scan",
        provenance={"source": "test"},
        license_name="CC0",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 3, "documents": [doc]}))
    packet = prepare_packet(manifest, tmp_path / "review", count=1)
    assert packet["reviews"][0]["status"] == "pending"
    assert packet["reviews"][0]["reviewer"] is None
    policy = json.loads((tmp_path / "review/policy-template.json").read_text())
    assert policy["approved_by"] is None
    assert policy["allowed_reviewers"] == []


def test_region_instrumentation_preserves_pdf_text_stream_order():
    import io

    pytest.importorskip("pdfplumber")
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    from benchmarks.quality_providers import PdfplumberProvider

    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=300)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 20 200 Td (alpha) Tj ET\nBT /F1 12 Tf 200 200 Td (bravo) Tj ET\nBT /F1 12 Tf 20 180 Td (charlie) Tj ET\nBT /F1 12 Tf 200 180 Td (delta) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    result = PdfplumberProvider().extract_outcomes(buffer.getvalue())[0]
    assert result.text == "alpha bravo charlie delta"
    assert result.regions
