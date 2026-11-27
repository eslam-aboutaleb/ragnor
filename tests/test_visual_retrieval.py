# ruff: noqa: PLR0913, PLR0917, S603
"""EX-10: opt-in page-image retrieval (core, in-memory index and Qdrant adapter)."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from benchmarks.visual_retrieval.fake_model import DIM, MODEL, FakeVisualProvider, make_page, png  # noqa: E402

from ragnor.ingestion.extraction_types import PageExtractionOutcome, PageExtractionStatus  # noqa: E402
from ragnor.retrieval.visual import (  # noqa: E402
    ImageLimits,
    InMemoryVisualIndex,
    ScoreCalibrator,
    SourceReference,
    TextPageHit,
    VisualEvidence,
    VisualIndexer,
    VisualModel,
    VisualPage,
    VisualRetrievalError,
    VisualRetriever,
    VisualRoutingPolicy,
    derive_interpretation,
    fuse_results,
    image_dimensions,
    maxsim,
    validate_multivector,
)

DRAWING_A = png({(0, 0): 3, (1, 2): 2, (3, 3): 1})
DRAWING_B = png({(2, 1): 3, (0, 3): 2})
DRAWING_C = png({(1, 1): 1})


class Auth:
    def __init__(self, deny=(), boom=False):
        self.deny, self.boom = set(deny), boom

    def is_authorized(self, tenant, principal, page):
        if self.boom:
            raise RuntimeError("authz backend down")
        return page.source.document_id not in self.deny and principal != "mallory"


class Artifacts:
    def __init__(self, blobs):
        self.blobs = blobs

    def read(self, ref):
        return self.blobs[ref]


def build(index=None, tenant="t1", provider=None):
    provider = provider or FakeVisualProvider()
    index = index or InMemoryVisualIndex(MODEL)
    pages = [
        (make_page(tenant, "dwg-a", 1, DRAWING_A), DRAWING_A),
        (make_page(tenant, "dwg-b", 1, DRAWING_B), DRAWING_B),
        (make_page(tenant, "dwg-c", 1, DRAWING_C), DRAWING_C),
    ]
    VisualIndexer(provider, index).index(tenant, pages)
    return provider, index, pages


def retriever(provider, index, **kw):
    return VisualRetriever(provider, index, kw.pop("auth", Auth()), **kw)


@pytest.fixture(params=["memory", "qdrant"])
def index(request):
    if request.param == "memory":
        return InMemoryVisualIndex(MODEL)
    qdrant = pytest.importorskip("qdrant_client")
    from ragnor.stores.qdrant_visual import QdrantVisualIndex

    return QdrantVisualIndex(qdrant.QdrantClient(":memory:"), MODEL)


# --------------------------------------------------------------- retrieval wins


def test_textless_raster_drawing_returns_page_image_evidence_with_source_and_page(index):
    provider, index, _ = build(index)
    found = retriever(provider, index).search("t1", "alice", "c6b2 c0b3", limit=3)
    assert found[0].page_id == "dwg-a-p1"
    assert found[0].source == SourceReference("src", "dwg-a", "v1")
    assert (found[0].page, found[0].artifact_ref) == (1, "private://t1/dwg-a/1")
    record = found[0].to_dict()
    assert record["kind"] == "visual_page" and record["derived"] is False
    assert not {"text", "quote", "snippet"} & record.keys()  # no invented text


def test_tenant_filter_prevents_cross_tenant_leakage(index):
    provider, index, _ = build(index, tenant="t1")
    other = [(make_page("t2", "secret", 1, DRAWING_B), DRAWING_B)]
    VisualIndexer(provider, index).index("t2", other)
    hits_t1 = retriever(provider, index).search("t1", "alice", "c9b3", limit=10)
    assert "secret-p1" not in {h.page_id for h in hits_t1}
    hits_t2 = retriever(provider, index).search("t2", "bob", "c9b3", limit=10)
    assert [h.page_id for h in hits_t2] == ["secret-p1"]


def test_revocation_removes_pages_immediately(index):
    provider, index, _ = build(index)
    assert index.revoke_document("t1", "dwg-a") == 1
    ids = {h.page_id for h in retriever(provider, index).search("t1", "alice", "c6b2", limit=10)}
    assert "dwg-a-p1" not in ids and "dwg-b-p1" in ids


def test_authorization_is_enforced_and_fails_closed():
    provider, index, _ = build()
    assert (
        retriever(provider, index, auth=Auth(deny={"dwg-a"}))
        .search("t1", "alice", "c6b2")[0]
        .page_id
        != "dwg-a-p1"
    )
    assert retriever(provider, index).search("t1", "mallory", "c6b2") == []
    assert retriever(provider, index, auth=Auth(boom=True)).search("t1", "alice", "c6b2") == []


def test_replay_and_reindex_do_not_duplicate_or_leave_stale_vectors(index):
    provider, index, pages = build(index)
    VisualIndexer(provider, index).index("t1", pages)  # replay
    hits = retriever(provider, index).search("t1", "alice", "c6b2", limit=10)
    assert [h.page_id for h in hits].count("dwg-a-p1") == 1
    assert len(hits) == 3
    new = png({(3, 0): 3})
    VisualIndexer(provider, index).index(
        "t1", [(make_page("t1", "dwg-a", 1, new, revision="v2"), new)]
    )
    hits = retriever(provider, index).search("t1", "alice", "c12b3", limit=10)
    assert hits[0].page_id == "dwg-a-p1" and hits[0].source.revision == "v2"
    assert len([h for h in hits if h.page_id == "dwg-a-p1"]) == 1


# ------------------------------------------------------------ model compatibility


def test_model_mismatch_fails_closed_everywhere(index):
    other = VisualModel("fake-colpali", "r2", DIM)
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(FakeVisualProvider(other), index)
    assert err.value.code == "model_mismatch"
    with pytest.raises(VisualRetrievalError):
        index.begin("t1", other)
    with pytest.raises(VisualRetrievalError):
        index.search("t1", [[1.0] * DIM], other, 3)
    with pytest.raises(VisualRetrievalError):
        VisualRetriever(FakeVisualProvider(other), index, Auth())


def test_indexer_reports_reindex_needed_when_model_version_changes():
    indexer = VisualIndexer(FakeVisualProvider(), InMemoryVisualIndex(MODEL))
    assert indexer.needs_reindex(VisualModel("fake-colpali", "r0", DIM))
    assert not indexer.needs_reindex(MODEL)


def test_qdrant_collection_is_per_model_and_attach_verifies_model():
    qdrant = pytest.importorskip("qdrant_client")
    from qdrant_client import models

    from ragnor.stores.qdrant_visual import _SENTINEL, QdrantVisualIndex

    client = qdrant.QdrantClient(":memory:")
    first = QdrantVisualIndex(client, MODEL)
    second = QdrantVisualIndex(client, VisualModel("fake-colpali", "r2", DIM))
    assert first.collection != second.collection
    # Same collection reattached with the same model works.
    assert QdrantVisualIndex(client, MODEL).collection == first.collection
    # A collection whose sentinel disagrees with the requested model is rejected.
    client.set_payload(first.collection, {"revision": "tampered"}, points=[str(_SENTINEL)])
    with pytest.raises(VisualRetrievalError) as err:
        QdrantVisualIndex(client, MODEL)
    assert err.value.code == "model_mismatch"
    # A collection with the wrong vector size / no multivector config is rejected.
    bad = QdrantVisualIndex.__new__(QdrantVisualIndex)
    bad.model, bad._client, bad.collection = MODEL, client, "plain"
    client.create_collection(
        "plain", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    )
    with pytest.raises(VisualRetrievalError):
        bad._verify_collection()


def test_qdrant_int8_quantization_and_invalid_option():
    qdrant = pytest.importorskip("qdrant_client")
    from ragnor.stores.qdrant_visual import QdrantVisualIndex

    client = qdrant.QdrantClient(":memory:")
    idx = QdrantVisualIndex(client, MODEL, quantization="int8", collection_prefix="q8")
    build(idx)
    with pytest.raises(ValueError):
        QdrantVisualIndex(client, MODEL, quantization="pq")


def test_vector_shape_validation():
    for bad in (
        [],
        [[1.0] * (DIM - 1)],
        [[0.0] * DIM],
        [[float("nan")] * DIM],
        [[True] * DIM],
        [["a"] * DIM],
    ):
        with pytest.raises(VisualRetrievalError) as err:
            validate_multivector(bad, MODEL)
        assert err.value.code == "invalid_vectors"
    with pytest.raises(VisualRetrievalError):
        validate_multivector([[1.0] * DIM] * 3, MODEL, max_patches=2)
    with pytest.raises(ValueError):
        VisualModel("", "r", 4)
    with pytest.raises(ValueError):
        VisualModel("m", "r", 0)


def test_qdrant_and_memory_scores_agree():
    qdrant = pytest.importorskip("qdrant_client")
    from ragnor.stores.qdrant_visual import QdrantVisualIndex

    memory = InMemoryVisualIndex(MODEL)
    remote = QdrantVisualIndex(qdrant.QdrantClient(":memory:"), MODEL)
    provider = FakeVisualProvider()
    build(memory, provider=provider)
    build(remote, provider=provider)
    query = provider.embed_query("c6b2 c0b3 c11b1")
    a = memory.search("t1", query, MODEL, 3)
    b = remote.search("t1", query, MODEL, 3)
    assert [h.page_id for h in a] == [h.page_id for h in b]
    for left, right in zip(a, b, strict=True):
        assert left.score == pytest.approx(right.score, abs=1e-4)
    assert maxsim([[1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]) == 1.0


# ----------------------------------------------------- failures, staging, resources


def test_failed_qdrant_write_rolls_back_staged_generation():
    qdrant = pytest.importorskip("qdrant_client")
    from ragnor.stores.qdrant_visual import QdrantVisualIndex

    client = qdrant.QdrantClient(":memory:")
    idx = QdrantVisualIndex(client, MODEL)
    provider = FakeVisualProvider()
    build(idx, provider=provider)
    real_upsert, state = client.upsert, {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("connection reset to 10.0.0.5 secret-token")
        return real_upsert(*args, **kwargs)

    client.upsert = flaky
    pages = [(make_page("t1", f"new{i}", 1, DRAWING_C), DRAWING_C) for i in range(3)]
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(provider, idx).index("t1", pages)
    assert err.value.code == "index_failed" and "secret" not in str(err.value)
    client.upsert = real_upsert
    ids = {h.page_id for h in retriever(provider, idx).search("t1", "a", "c5b0", limit=20)}
    assert not any(i.startswith("new") for i in ids)  # nothing from the failed snapshot is visible
    assert client.count(idx.collection, exact=True).count == 1 + 3  # sentinel + original pages


def test_failed_commit_rolls_back(index):
    provider, index, pages = build(index)
    real = index.commit

    def boom(generation):
        raise VisualRetrievalError("index_failed")

    index.commit = boom
    extra = [(make_page("t1", "x", 1, DRAWING_C), DRAWING_C)]
    with pytest.raises(VisualRetrievalError):
        VisualIndexer(provider, index).index("t1", extra)
    index.commit = real
    assert "x-p1" not in {
        h.page_id for h in retriever(provider, index).search("t1", "a", "c5b0", limit=20)
    }


def test_cancellation_before_commit_publishes_nothing(index):
    provider = FakeVisualProvider()

    class Flag:
        cancelled = False

    flag = Flag()
    real = provider.embed_pages

    def embed_and_cancel(images):
        flag.cancelled = True
        return real(images)

    provider.embed_pages = embed_and_cancel
    pages = [(make_page("t1", f"d{i}", 1, DRAWING_A), DRAWING_A) for i in range(3)]
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(provider, index, batch_size=1).index("t1", pages, cancel=flag)
    assert err.value.code == "cancelled"
    assert retriever(provider, index).search("t1", "a", "c0b3", limit=10) == []


def test_pre_cancelled_and_oom_and_provider_error_are_mapped(index):
    class Done:
        cancelled = True

    pages = [(make_page("t1", "d", 1, DRAWING_A), DRAWING_A)]
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(FakeVisualProvider(), index).index("t1", pages, cancel=Done())
    assert err.value.code == "cancelled"
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(FakeVisualProvider(fail=MemoryError()), index).index("t1", pages)
    assert err.value.code == "resource_limit"
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(FakeVisualProvider(fail=RuntimeError("api key sk-abc")), index).index(
            "t1", pages
        )
    assert err.value.code == "provider_failed" and "sk-" not in str(err.value)
    assert retriever(FakeVisualProvider(), index).search("t1", "a", "c0b3") == []


def test_provider_must_return_valid_vectors(index):
    class Short(FakeVisualProvider):
        def embed_pages(self, images):
            return []

    class Wrong(FakeVisualProvider):
        def embed_pages(self, images):
            return [[[1.0, 2.0]] for _ in images]

    pages = [(make_page("t1", "d", 1, DRAWING_A), DRAWING_A)]
    for provider in (Short(), Wrong()):
        with pytest.raises(VisualRetrievalError) as err:
            VisualIndexer(provider, index).index("t1", pages)
        assert err.value.code == "invalid_vectors"


def test_query_provider_failure_is_content_free():
    provider, index, _ = build()

    class Bad(FakeVisualProvider):
        def embed_query(self, text):
            raise RuntimeError(f"leak {text}")

    with pytest.raises(VisualRetrievalError) as err:
        retriever(Bad(), index).search("t1", "a", "c0b3")
    assert err.value.code == "provider_failed" and "leak" not in str(err.value)


def test_remote_providers_require_explicit_opt_in():
    index = InMemoryVisualIndex(MODEL)
    remote = FakeVisualProvider(remote=True)
    with pytest.raises(VisualRetrievalError) as err:
        VisualIndexer(remote, index)
    assert err.value.code == "remote_not_allowed"
    with pytest.raises(VisualRetrievalError):
        VisualRetriever(remote, index, Auth())
    VisualIndexer(remote, index, allow_remote=True)
    VisualRetriever(remote, index, Auth(), allow_remote=True)


def test_indexer_rejects_tenant_hash_and_dimension_mismatch_before_staging():
    provider, index = FakeVisualProvider(), InMemoryVisualIndex(MODEL)
    indexer = VisualIndexer(provider, index)
    page = make_page("t1", "d", 1, DRAWING_A)
    with pytest.raises(VisualRetrievalError):
        indexer.index("other-tenant", [(page, DRAWING_A)])
    with pytest.raises(VisualRetrievalError):
        indexer.index("t1", [(page, DRAWING_B)])  # bytes differ from the immutable hash
    bigger = png({(0, 0): 1}, size=64)
    mismatched = make_page("t1", "d", 1, DRAWING_A)
    forged = VisualPage(
        **{**vars(mismatched), "content_sha256": hashlib.sha256(bigger).hexdigest()}
    )
    with pytest.raises(VisualRetrievalError):
        indexer.index("t1", [(forged, bigger)])
    assert provider.calls == 0
    with pytest.raises(ValueError):
        VisualIndexer(provider, index, batch_size=0)


def test_stage_rejects_unknown_generation_and_foreign_tenant():
    index = InMemoryVisualIndex(MODEL)
    page = make_page("t1", "d", 1, DRAWING_A)
    vec = FakeVisualProvider().embed_pages([DRAWING_A])[0]
    with pytest.raises(VisualRetrievalError) as err:
        index.stage("nope", page, vec, MODEL)
    assert err.value.code == "unknown_generation"
    gen = index.begin("t2", MODEL)
    with pytest.raises(VisualRetrievalError):
        index.stage(gen, page, vec, MODEL)
    with pytest.raises(VisualRetrievalError):
        index.commit("nope")
    index.rollback("nope")
    assert index.vector_count() == 0


# ------------------------------------------------------------------ image safety


def test_image_limits_and_malformed_images():
    assert image_dimensions(DRAWING_A) == (32, 32)
    with pytest.raises(VisualRetrievalError) as err:
        image_dimensions(DRAWING_A, ImageLimits(max_bytes=10))
    assert err.value.code == "image_too_large"
    with pytest.raises(VisualRetrievalError) as err:
        image_dimensions(DRAWING_A, ImageLimits(max_pixels=100))
    assert err.value.code == "image_too_large"
    with pytest.raises(VisualRetrievalError):
        image_dimensions(DRAWING_A, ImageLimits(max_side=16))
    for junk in (
        b"",
        b"not an image",
        b"\x89PNG\r\n\x1a\n" + b"x" * 4,
        DRAWING_A[:20],
        b"\xff\xd8\xff",
    ):
        with pytest.raises(VisualRetrievalError) as err:
            image_dimensions(junk)
        assert err.value.code == "invalid_image"
    # Decompression bomb: a tiny PNG header declaring a huge canvas is refused unseen.
    bomb = DRAWING_A[:16] + (60000).to_bytes(4, "big") + (60000).to_bytes(4, "big") + DRAWING_A[24:]
    with pytest.raises(VisualRetrievalError) as err:
        image_dimensions(bomb)
    assert err.value.code == "image_too_large"


def test_jpeg_header_is_parsed_without_decoding():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48, 20), "white").save(buf, "JPEG")
    assert image_dimensions(buf.getvalue()) == (48, 20)
    truncated = buf.getvalue()[:30]
    with pytest.raises(VisualRetrievalError):
        image_dimensions(truncated)
    with pytest.raises(VisualRetrievalError):
        image_dimensions(b"\xff\xd8\x00\x00")
    with pytest.raises(VisualRetrievalError):
        image_dimensions(b"\xff\xd8\xff\xc0\x00\x01")


@pytest.mark.parametrize(
    "ref",
    [
        "https://cdn.example/p.png",
        "HTTP://x/y",
        "data:image/png;base64,AAAA",
        "file:///etc/passwd",
        "//cdn/x",
        "",
    ],
)
def test_public_urls_are_not_valid_artifact_references(ref):
    with pytest.raises(ValueError):
        make_page("t1", "d", 1, DRAWING_A, ref=ref or None) if ref else VisualPage(
            "t1", "p", SourceReference("s", "d", "v"), 1, "", 1, 1, "0" * 64
        )


def test_page_model_validation_and_coordinate_transforms():
    page = make_page("t1", "d", 1, DRAWING_A)
    assert page.to_pixels((0.25, 0.5, 0.75, 1.0)) == (8, 16, 24, 32)
    assert page.to_normalized((8, 16, 24, 32)) == (0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError):
        page.to_pixels((0.5, 0.5, 0.4, 0.9))
    with pytest.raises(ValueError):
        page.to_normalized((0, 0, 99, 5))
    base = vars(page)
    for change in ({"page": 0}, {"width": 0}, {"content_sha256": "zz"}, {"tenant": ""}):
        with pytest.raises(ValueError):
            VisualPage(**{**base, **change})
    assert page.key == ("t1", "d", 1)


def test_fetch_image_enforces_authorization_hash_and_format():
    provider, index, pages = build()
    page, data = pages[0]
    store = Artifacts({page.artifact_ref: data})
    ok = retriever(provider, index, artifacts=store)
    assert ok.fetch_image("t1", "alice", page) == data
    with pytest.raises(VisualRetrievalError) as err:
        ok.fetch_image("t1", "mallory", page)
    assert err.value.code == "unauthorized"
    with pytest.raises(VisualRetrievalError) as err:
        ok.fetch_image("t2", "alice", page)  # cross-tenant request
    assert err.value.code == "unauthorized"
    swapped = retriever(provider, index, artifacts=Artifacts({page.artifact_ref: DRAWING_B}))
    with pytest.raises(VisualRetrievalError) as err:
        swapped.fetch_image("t1", "alice", page)
    assert err.value.code == "invalid_image"
    corrupt = retriever(provider, index, artifacts=Artifacts({page.artifact_ref: b"junk"}))
    with pytest.raises(VisualRetrievalError):
        corrupt.fetch_image("t1", "alice", page)
    with pytest.raises(VisualRetrievalError):
        retriever(provider, index).fetch_image("t1", "alice", page)  # no artifact store


# --------------------------------------------------------------- routing, fusion


def test_policy_routes_families_and_unstructurable_pages_and_keeps_text():
    policy = VisualRoutingPolicy(families=frozenset({"cad-drawings"}))
    assert policy.route("cad-drawings", "extracted", True).reason == "designated_family"
    assert policy.route("cad-drawings", "extracted", True).keep_text
    unstructurable = policy.route(None, PageExtractionStatus.NEEDS_OCR.value, False)
    assert unstructurable.visual and not unstructurable.keep_text
    plain = policy.route("invoice", PageExtractionStatus.EXTRACTED.value, True)
    assert not plain.visual and plain.keep_text
    outcome = PageExtractionOutcome(1, PageExtractionStatus.PARTIAL, text="some")
    assert policy.route(None, outcome.status.value, bool(outcome.text)).visual


def _ev(doc, page, score, revision="v1", tenant="t1"):
    return VisualEvidence(
        tenant,
        SourceReference("s", doc, revision),
        page,
        f"{doc}-p{page}",
        f"private://{doc}",
        score,
    )


def _tx(doc, page, score, revision="v1", tenant="t1", hit="h"):
    return TextPageHit(
        f"{hit}-{doc}-{page}", tenant, SourceReference("s", doc, revision), page, score
    )


def test_fusion_dedupes_pages_with_both_representations_and_calibrates():
    vis_cal = ScoreCalibrator([0.5, 1.0, 1.5, 2.0])
    txt_cal = ScoreCalibrator([0.01, 0.02, 0.03, 0.04])
    fused = fuse_results(
        [_ev("a", 1, 2.0), _ev("b", 1, 0.5), _ev("a", 1, 1.0)],
        [_tx("a", 1, 0.02), _tx("c", 2, 0.04)],
        visual_calibrator=vis_cal,
        text_calibrator=txt_cal,
    )
    keys = [(r.source.document_id, r.page) for r in fused]
    assert len(keys) == len(set(keys)) == 3
    top = fused[0]
    assert top.representations == ("text", "visual") and top.evidence.text_hit_id == top.text_hit_id
    assert top.score == 1.0
    assert fuse_results([], [], visual_calibrator=vis_cal, text_calibrator=txt_cal) == []
    assert (
        len(
            fuse_results(
                [_ev("a", 1, 2.0)],
                [_tx("c", 2, 0.04)],
                visual_calibrator=vis_cal,
                text_calibrator=txt_cal,
                limit=1,
            )
        )
        == 1
    )


def test_fusion_drops_stale_revisions_and_text_only_results_have_no_evidence():
    cal = ScoreCalibrator([1.0])
    fused = fuse_results(
        [_ev("a", 1, 1.0, revision="v1")],
        [_tx("a", 1, 1.0, revision="v2"), _tx("z", 1, 1.0)],
        visual_calibrator=cal,
        text_calibrator=cal,
        current_revisions={"a": "v2"},
    )
    assert [(r.source.document_id, r.source.revision) for r in fused] == [("a", "v2"), ("z", "v1")]
    assert all(r.evidence is None for r in fused)  # stale visual evidence was dropped
    assert fused[0].representations == ("text",)


def test_calibrator_is_monotonic_and_validated():
    cal = ScoreCalibrator([3.0, 1.0, 2.0, float("nan")])
    assert [cal(x) for x in (0.0, 1.0, 2.5, 9.0)] == [0.0, 1 / 3, 2 / 3, 1.0]
    with pytest.raises(ValueError):
        ScoreCalibrator([float("nan")])


def test_derived_interpretation_is_flagged_and_failures_are_content_free():
    ev = _ev("a", 1, 1.0)
    out = derive_interpretation(ev, lambda e: "looks like a pump assembly")
    assert out["derived"] is True and out["kind"] == "derived_interpretation"
    assert out["derived_from"]["page_id"] == "a-p1" and ev.derived is False

    def boom(_):
        raise RuntimeError("secret")

    with pytest.raises(VisualRetrievalError) as err:
        derive_interpretation(ev, boom)
    assert "secret" not in str(err.value)
    assert VisualRetrievalError("made-up").code == "provider_failed"


# -------------------------------------------------------------- optional-extra rules


def test_core_visual_module_imports_without_optional_extras():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys\n"
        "for name in ('qdrant_client','PIL','numpy','litellm','pypdfium2'): sys.modules[name] = None\n"
        "import ragnor.retrieval.visual as v\n"
        "assert v.InMemoryVisualIndex\n"
    )
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )  # noqa: S603
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    not os.environ.get("RAGNOR_VISUAL_MODEL_SMOKE"),
    reason="optional real-model smoke: set RAGNOR_VISUAL_MODEL_SMOKE and install colpali-engine",
)
def test_real_visual_model_smoke():  # pragma: no cover - needs model weights
    pytest.importorskip("colpali_engine")
    pytest.skip("Host must wire a ColPali provider; no weights ship with Ragnor")
