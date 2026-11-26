"""Page profiling on real PDFs and the deterministic routing matrix."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ragnor.extraction import (
    Available,
    Capability,
    PageKind,
    PageProfile,
    Route,
    RoutePolicy,
    plan_document,
    route_page,
    summarize,
)
from ragnor.extraction.registry import ProviderSpec

pytest.importorskip("pypdf")
from ragnor.ingestion.pdf_profiler import PdfPageProfiler  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "benchmarks/pdf_extraction/fixtures/documents"
PROFILER = PdfPageProfiler()


def _profile(name: str) -> list[PageProfile]:
    return PROFILER.profile_pages((DOCS / name).read_bytes())


def _pdf(content: bytes, *, rotate: int = 0) -> bytes:
    """Minimal one-page PDF with Helvetica; real bytes, parsed by pypdf."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate "
        + str(rotate).encode()
        + b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


TEXT = b"BT /F1 12 Tf 72 700 Td (Quarterly revenue grew twelve percent) Tj ET"
SIDEWAYS = b"BT /F1 12 Tf 0 1 -1 0 300 100 Tm (Quarterly revenue grew twelve percent) Tj ET"
UPSIDE = b"BT /F1 12 Tf -1 0 0 -1 500 700 Tm (Quarterly revenue grew twelve percent) Tj ET"

ALL = Available(local=frozenset(Capability))
TEXT_ONLY = Available(local=frozenset({Capability.TEXT}))
NOTHING = Available()


# -- profiling real PDFs -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kinds"),
    [
        ("plain-digital-report.pdf", ["digital", "digital"]),
        ("scanned-contract.pdf", ["scan", "scan"]),
        ("handwriting-note.pdf", ["scan"]),
        ("mixed-memo.pdf", ["mixed", "scan"]),
        ("chart-report.pdf", ["mixed"]),
        ("rotated-notice.pdf", ["digital"]),
    ],
)
def test_fixture_pages_are_classified(name, kinds):
    assert [p.kind.value for p in _profile(name)] == kinds


def test_columns_are_detected_on_real_layouts():
    assert [p.text_columns for p in _profile("three-column-brochure.pdf")] == [3, 3]
    assert [p.text_columns for p in _profile("two-column-newsletter.pdf")] == [2, 2]
    assert [p.text_columns for p in _profile("plain-digital-report.pdf")] == [1, 1]


def test_blank_page_is_verified_blank():
    (page,) = PROFILER.profile_pages(_pdf(b""))
    assert page.kind is PageKind.BLANK
    assert page.char_count == 0


def test_upright_text_is_not_sideways():
    (page,) = PROFILER.profile_pages(_pdf(TEXT))
    assert (page.kind, page.rotation, page.visual_orientation) == (PageKind.DIGITAL, 0, 0)
    assert not page.sideways


def test_sideways_text_on_unrotated_page_is_detected():
    (page,) = PROFILER.profile_pages(_pdf(SIDEWAYS))
    assert page.text_orientation == 90
    assert page.sideways
    assert page.visual_orientation == 90


def test_upside_down_text_is_detected():
    (page,) = PROFILER.profile_pages(_pdf(UPSIDE))
    assert page.visual_orientation == 180


def test_rotate_flag_that_cancels_text_angle_reads_upright():
    # Text drawn at 90 degrees on a page with /Rotate 90 displays upright.
    (page,) = PROFILER.profile_pages(_pdf(SIDEWAYS, rotate=90))
    assert (page.rotation, page.text_orientation) == (90, 90)
    assert not page.sideways


def test_rotate_flag_on_upright_text_is_sideways_for_the_viewer():
    (page,) = PROFILER.profile_pages(_pdf(TEXT, rotate=270))
    assert page.rotation == 270
    assert page.visual_orientation == 90


def test_corrupt_and_oversized_inputs_raise_rather_than_return_nothing():
    with pytest.raises(Exception):  # noqa: B017, PT011 - pypdf raises its own parse errors
        PROFILER.profile_pages((DOCS / "corrupted-report.pdf").read_bytes())
    with pytest.raises(ValueError, match="max_pages"):
        PdfPageProfiler(max_pages=1000).profile_pages((DOCS / "oversized-blank.pdf").read_bytes())
    with pytest.raises(ValueError, match="max_bytes"):
        PdfPageProfiler(max_bytes=10).profile_pages(_pdf(TEXT))


def test_document_profile_matches_the_page_profiler_port():
    profile = asyncio.run(PROFILER.profile((DOCS / "mixed-memo.pdf").read_bytes()))
    assert profile.media_type == "application/pdf"
    assert profile.page_count == 2
    assert profile.has_text_layer is True
    assert profile.source_sha256


# -- routing matrix ----------------------------------------------------------


def _p(kind, **kw):
    return PageProfile(1, kind, **kw)


@pytest.mark.parametrize(
    ("profile", "available", "route"),
    [
        (_p(PageKind.DIGITAL), ALL, Route.NATIVE),
        (_p(PageKind.DIGITAL), TEXT_ONLY, Route.NATIVE),
        (_p(PageKind.DIGITAL, text_columns=3), ALL, Route.LAYOUT),
        (_p(PageKind.DIGITAL, text_columns=3), TEXT_ONLY, Route.NATIVE),
        (_p(PageKind.SCAN), ALL, Route.OCR),
        (_p(PageKind.SCAN), TEXT_ONLY, Route.UNROUTABLE),
        (_p(PageKind.MIXED), ALL, Route.LAYOUT),
        (
            _p(PageKind.MIXED),
            Available(local=frozenset({Capability.TEXT, Capability.OCR})),
            Route.OCR,
        ),
        (_p(PageKind.MIXED), TEXT_ONLY, Route.NATIVE),
        (_p(PageKind.BLANK), NOTHING, Route.SKIP_BLANK),
        (_p(PageKind.UNKNOWN), ALL, Route.OCR),
        (_p(PageKind.UNKNOWN), TEXT_ONLY, Route.UNROUTABLE),
        (_p(PageKind.DIGITAL), NOTHING, Route.UNROUTABLE),
    ],
)
def test_routing_matrix(profile, available, route):
    assert route_page(profile, available).route is route


def test_unroutable_names_the_missing_capability_and_never_skips():
    d = route_page(_p(PageKind.SCAN), TEXT_ONLY)
    assert d.route is Route.UNROUTABLE
    assert d.needs == (Capability.OCR,)
    assert d.capability is None


def test_remote_providers_need_policy_opt_in():
    remote = Available(local=frozenset({Capability.TEXT}), remote=frozenset({Capability.OCR}))
    blocked = route_page(_p(PageKind.SCAN), remote)
    assert blocked.route is Route.UNROUTABLE
    assert "remote_provider_blocked_by_policy" in blocked.notes
    allowed = route_page(_p(PageKind.SCAN), remote, RoutePolicy(allow_remote=True))
    assert allowed.route is Route.OCR


def test_local_provider_beats_remote_when_both_exist():
    specs = [
        ProviderSpec(
            "hosted",
            lambda o: None,
            frozenset({"application/pdf"}),
            frozenset({Capability.OCR}),
            remote=True,
        ),
        ProviderSpec(
            "local", lambda o: None, frozenset({"application/pdf"}), frozenset({Capability.OCR})
        ),
    ]
    avail = Available.from_specs(specs)
    assert Capability.OCR in avail.local
    assert route_page(_p(PageKind.SCAN), avail).route is Route.OCR


def test_sideways_pages_are_flagged_for_bbox_transform():
    d = route_page(_p(PageKind.DIGITAL, text_orientation=90), ALL)
    assert d.route is Route.NATIVE
    assert "bbox_requires_rotation_transform" in d.notes


def test_policy_can_disable_column_layout_routing():
    d = route_page(
        _p(PageKind.DIGITAL, text_columns=3), ALL, RoutePolicy(prefer_layout_for_columns=False)
    )
    assert d.route is Route.NATIVE


def test_routing_is_deterministic_and_json_serialisable():
    pages = _profile("mixed-memo.pdf") + _profile("three-column-brochure.pdf")
    first = plan_document(pages, ALL)
    assert first == plan_document(pages, ALL)
    json.dumps([d.to_dict() for d in first])
    json.dumps([p.to_dict() for p in pages])


def test_every_page_gets_exactly_one_decision():
    pages = _profile("scanned-contract.pdf") + _profile("plain-digital-report.pdf")
    decisions = plan_document(pages, TEXT_ONLY)
    assert len(decisions) == len(pages)
    summary = summarize(decisions)
    assert summary["unroutable_pages"] == [1, 2]
    assert summary["routes"] == {"unroutable": 2, "native": 2}


def test_invalid_profile_values_are_rejected():
    with pytest.raises(ValueError, match="rotation"):
        PageProfile(1, PageKind.DIGITAL, rotation=45)
    with pytest.raises(ValueError, match="Page numbers"):
        PageProfile(0, PageKind.DIGITAL)
