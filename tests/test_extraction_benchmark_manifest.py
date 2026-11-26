"""Corpus schema validation tests (EX-02): missing pages, duplicate region
ids, bbox outside page, inconsistent rowspan/colspan, invalid reading order
and checksum mismatch, plus a validity check over the shipped corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pdf_extraction.manifest import (
    Manifest,
    load_manifest,
    validate_manifest,
)

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "pdf_extraction" / "fixtures"
MANIFEST = FIXTURES / "manifest.jsonl"


def base_record(**overrides):
    record = {
        "document_id": "doc",
        "file": "documents/doc.pdf",
        "family": "plain_digital",
        "page": 1,
        "page_count": 1,
        "width": 612,
        "height": 792,
        "license": "synthetic-generated-cc0",
        "source_hash": "sha256:unused",
        "regions": [
            {
                "region_id": "r1",
                "kind": "text",
                "bbox": [10, 10, 100, 30],
                "reading_order": 0,
                "expected_text": "hello",
            }
        ],
        "tables": [],
        "critical_fields": [],
    }
    record.update(overrides)
    return record


def manifest_from(records) -> Manifest:
    pages = []
    for record in records:
        line = json.dumps(record)
        path = FIXTURES / ".test-tmp-manifest.jsonl"
        path.write_text(line, encoding="utf-8")
        try:
            pages.extend(load_manifest(path).pages)
        finally:
            path.unlink()
    return Manifest(corpus_version="test", pages=pages)


def errors_for(records, root: Path | None = None) -> list[str]:
    return validate_manifest(manifest_from(records), root or FIXTURES)


class TestShippedCorpus:
    def test_shipped_manifest_is_valid(self):
        manifest = load_manifest(MANIFEST)
        assert validate_manifest(manifest, FIXTURES) == []

    def test_shipped_corpus_size_and_families(self):
        manifest = load_manifest(MANIFEST)
        families = {p.family for p in manifest.pages}
        assert 30 <= len(manifest.pages) <= 50
        expected = {
            "plain_digital",
            "two_column",
            "three_column",
            "scan",
            "mixed",
            "rotated",
            "ruled_table_invoice",
            "borderless_table",
            "split_merged_table",
            "multilevel_headings",
            "arabic_rtl",
            "multilingual",
            "contract",
            "chart",
            "formula",
            "handwriting",
            "corrupted",
            "oversized",
        }
        assert expected <= families

    def test_no_proprietary_or_llm_generated_labels(self):
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if "corpus_version" in line:
                continue
            record = json.loads(line)
            assert record["license"] == "synthetic-generated-cc0"


class TestValidationRules:
    def test_missing_pages_detected(self):
        first = base_record(document_id="d", page=1, page_count=2)
        errors = errors_for([first], root=Path("/nonexistent"))
        assert any("missing pages [2]" in e for e in errors)

    def test_duplicate_region_ids_detected(self):
        record = base_record(regions=[base_record()["regions"][0], base_record()["regions"][0]])
        errors = errors_for([record], root=Path("/nonexistent"))
        assert any("duplicate region id r1" in e for e in errors)

    def test_bbox_outside_page_detected(self):
        record = base_record(
            regions=[
                {
                    "region_id": "r1",
                    "kind": "text",
                    "bbox": [10, 10, 700, 30],
                    "reading_order": 0,
                    "expected_text": "x",
                }
            ]
        )
        errors = errors_for([record], root=Path("/nonexistent"))
        assert any("bbox outside page" in e for e in errors)

    def test_span_exceeding_grid_detected(self):
        record = base_record(
            tables=[
                {
                    "region_id": "r1",
                    "rows": 2,
                    "cols": 2,
                    "cells": [{"row": 1, "col": 1, "text": "a", "colspan": 2}],
                }
            ]
        )
        errors = errors_for([record], root=Path("/nonexistent"))
        assert any("exceeds 2x2 grid" in e for e in errors)

    def test_overlapping_cells_detected(self):
        record = base_record(
            regions=[base_record()["regions"][0]],
            tables=[
                {
                    "region_id": "r1",
                    "rows": 2,
                    "cols": 2,
                    "cells": [
                        {"row": 0, "col": 0, "text": "a", "rowspan": 2},
                        {"row": 1, "col": 0, "text": "b"},
                    ],
                }
            ],
        )
        errors = errors_for([record], root=Path("/nonexistent"))
        assert any("cells overlap at grid position (1,0)" in e for e in errors)

    def test_invalid_reading_order_detected(self):
        record = base_record(
            regions=[
                base_record()["regions"][0],
                {
                    "region_id": "r2",
                    "kind": "text",
                    "bbox": [10, 40, 100, 60],
                    "reading_order": 5,
                    "expected_text": "y",
                },
            ]
        )
        errors = errors_for([record], root=Path("/nonexistent"))
        assert any("reading order must be a permutation" in e for e in errors)

    def test_checksum_mismatch_detected(self, tmp_path):
        payload = tmp_path / "documents"
        payload.mkdir()
        (payload / "doc.pdf").write_bytes(b"%PDF-real-bytes")
        record = base_record(source_hash="sha256:" + "0" * 64)
        errors = errors_for([record], root=tmp_path)
        assert any("checksum mismatch" in e for e in errors)

    def test_checksum_match_passes(self, tmp_path):
        payload = tmp_path / "documents"
        payload.mkdir()
        data = b"%PDF-real-bytes"
        (payload / "doc.pdf").write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        record = base_record(source_hash=f"sha256:{digest}")
        assert errors_for([record], root=tmp_path) == []

    def test_missing_fixture_file_detected(self, tmp_path):
        errors = errors_for([base_record()], root=tmp_path)
        assert any("missing fixture file" in e for e in errors)
