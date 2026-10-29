"""Structured evidence invariants without optional provider dependencies."""

from dataclasses import replace

import pytest

from ragnor.ingestion import (
    EvidenceAllowlistVerifier,
    ExtractionCell,
    ExtractionTable,
    ExtractionWord,
    PDFExtractionLimits,
    PageExtractionOutcome,
    PageExtractionStatus,
)


def test_table_serialization_preserves_spans_and_escapes_source_values():
    table = ExtractionTable(
        (0, 0, 1, 1),
        2,
        2,
        (
            ExtractionCell(0, 0, "A < B & C", (0, 0, 1, 0.5), colspan=2),
            ExtractionCell(1, 0, "10", (0, 0.5, 0.5, 1)),
            ExtractionCell(1, 1, "20", (0.5, 0.5, 1, 1)),
        ),
    )
    assert 'colspan="2"' in table.to_html()
    assert "A &lt; B &amp; C" in table.to_html()
    with pytest.raises(ValueError, match="Overlapping"):
        replace(table, cells=(*table.cells, table.cells[0]))
    with pytest.raises(ValueError, match="span exceeds"):
        replace(table, cells=(replace(table.cells[0], colspan=3),))


@pytest.mark.parametrize("box", [(0, 0, 2, 1), (0, 0, 0, 1), (0, 0, float("nan"), 1)])
def test_invalid_word_geometry_is_rejected(box):
    with pytest.raises(ValueError):
        ExtractionWord("value", box)


def test_evidence_hash_changes_for_source_model_and_geometry_but_not_acceptance():
    from typing import get_type_hints
    from ragnor.ingestion.extractors import PageExtractionOutcome as LegacyOutcome
    from ragnor.ingestion.extraction_types import (
        PageExtractionOutcome as SharedOutcome,
        PageVerifier,
    )

    assert LegacyOutcome is SharedOutcome is PageExtractionOutcome
    assert get_type_hints(PageVerifier.verify)["page"] is SharedOutcome
    original = PageExtractionOutcome(
        1,
        PageExtractionStatus.PARTIAL,
        "body",
        words=(ExtractionWord("body", (0, 0, 1, 1)),),
        diagnostics={
            "source_sha256": "a" * 64,
            "runtime": {"models_sha256": {"eng": "old"}},
            "verification": "pending",
        },
    )
    accepted = replace(
        original,
        status=PageExtractionStatus.EXTRACTED,
        reason="host_policy_accepted",
        diagnostics={**original.diagnostics, "verification": {"accepted": True}},
    )
    assert original.evidence_sha256 == accepted.evidence_sha256
    verifier = EvidenceAllowlistVerifier([original.evidence_sha256])
    assert verifier.verify(accepted)
    assert not verifier.verify(
        replace(original, diagnostics={**original.diagnostics, "source_sha256": "b" * 64})
    )
    assert not verifier.verify(
        replace(
            original,
            diagnostics={**original.diagnostics, "runtime": {"models_sha256": {"eng": "new"}}},
        )
    )
    assert not verifier.verify(replace(original, words=(ExtractionWord("body", (0, 0, 0.5, 1)),)))
    assert not EvidenceAllowlistVerifier([]).verify(original)
    with pytest.raises(ValueError):
        EvidenceAllowlistVerifier(["not a hash"])


@pytest.mark.parametrize(
    "values", [{"max_pages": True}, {"max_words": 0}, {"timeout_seconds": float("inf")}]
)
def test_limits_are_finite_positive_values(values):
    with pytest.raises(ValueError):
        PDFExtractionLimits(**values)


def test_unknown_provider_error_text_cannot_become_a_public_failure_code():
    from ragnor.ingestion import PDFExtractionError
    from ragnor.ingestion.connector_core import _error_record

    error = PDFExtractionError("PRIVATE_SOURCE_BODY")
    assert error.code == "provider_failed"
    assert "PRIVATE" not in str(error) + str(_error_record(error))


def test_large_review_allowlist_has_bounded_identity_and_preserves_membership():
    import json
    from ragnor.ingestion import PdfPlumberExtractor

    hashes = [f"{number:064x}" for number in range(2000)]
    verifier = EvidenceAllowlistVerifier(hashes)
    assert len(json.dumps(verifier.options)) < 300
    assert verifier.options == EvidenceAllowlistVerifier(reversed(hashes)).options
    assert verifier.options != EvidenceAllowlistVerifier(hashes[:-1]).options
    PdfPlumberExtractor(verifier=verifier)
