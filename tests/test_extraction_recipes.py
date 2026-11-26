"""EX-09: versioned recipes, scoped stores, guarded fast path and learning."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from ragnor.ingestion.extraction_types import (
    EvidenceAllowlistVerifier,
    PageExtractionOutcome,
    PageExtractionStatus,
    extraction_evidence_hash,
)
from ragnor.ingestion.recipes import (
    CancellationToken,
    CountCoverageVerifier,
    DocumentFingerprint,
    ExtractedRegion,
    ExtractionCancelled,
    ExtractionRecipe,
    FamilyClassifier,
    FamilyCriteria,
    InMemoryRecipeStore,
    JsonFileRecipeStore,
    QualityThresholds,
    RecipeAnchor,
    RecipeError,
    RecipeFastPath,
    RecipeLearner,
    RecipeLearningRefused,
    RecipeProvenance,
    RecipeRegion,
    RecipeStatus,
    RecipeTelemetry,
    RecipeValidity,
    SQLiteRecipeStore,
)

VALIDITY = RecipeValidity.for_options(
    {"dpi": 200},
    extractor="tesseract",
    extractor_version="5",
    model="none",
    layout_schema="3",
    rotation=0,
    template_version="2026-01",
)
ANCHORS = (
    RecipeAnchor("ACME INVOICE", (0.1, 0.05, 0.4, 0.1)),
    RecipeAnchor("Total", (0.6, 0.8, 0.7, 0.85)),
)
CRITERIA = FamilyCriteria("acme-invoice", (("vendor", "acme"),), ANCHORS, page_count=1)
REGIONS = (
    RecipeRegion("header", 1, (0.1, 0.05, 0.9, 0.2), "tesseract"),
    RecipeRegion("lines", 1, (0.1, 0.3, 0.9, 0.7), "table-provider", kind="table"),
)


def make_recipe(tenant="t1", criteria=CRITERIA, validity=VALIDITY, **kw) -> ExtractionRecipe:
    return ExtractionRecipe(
        tenant=tenant,
        criteria=criteria,
        regions=REGIONS,
        validity=validity,
        provenance=RecipeProvenance("tester", "2026-01-01T00:00:00Z", "a" * 64, "v", "1"),
        expected_counts=(("lines", 2, 4),),
        **kw,
    )


def fingerprint(vendor="acme", words=None, page_count=1, rotation=0) -> DocumentFingerprint:
    words = (
        words
        if words is not None
        else (("ACME INVOICE", (0.1, 0.05, 0.4, 0.1)), ("Total", (0.6, 0.8, 0.7, 0.85)))
    )
    return DocumentFingerprint({"vendor": vendor}, tuple(words), page_count, rotation)


class Extractor:
    """Deterministic region extractor over dict documents."""

    def __init__(self):
        self.calls = []

    def extract_region(self, document, region):
        self.calls.append(region.name)
        value = document[region.name]
        if isinstance(value, Exception):
            raise value
        return value


def doc(lines=3, header="Invoice 7", drop=None):
    data = {
        "header": ExtractedRegion("header", header, 1),
        "lines": ExtractedRegion("lines", "a|b", lines, (lines, 2)),
    }
    if drop:
        data.pop(drop)
    return data


class Env:
    def __init__(self, store=None, engine=VALIDITY, classifier=None):
        self.store = store or InMemoryRecipeStore()
        self.extractor = Extractor()
        self.full_calls = 0
        self.classifier = classifier or FamilyClassifier([CRITERIA])

        def full(document):
            self.full_calls += 1
            return [v for v in document.values() if isinstance(v, ExtractedRegion)]

        self.path = RecipeFastPath(
            classifier=self.classifier,
            store=self.store,
            region_extractor=self.extractor,
            verifier=CountCoverageVerifier({"lines": (3, 2)}),
            full_extractor=full,
            engine_validity=engine,
            full_cost_calls=5,
        )


# ---------------------------------------------------------------- recipe model


def test_recipe_json_round_trip_is_lossless_and_hash_stable():
    recipe = make_recipe()
    again = ExtractionRecipe.from_json(recipe.to_json())
    assert again == recipe
    assert again.content_sha256 == recipe.content_sha256
    assert recipe.extraction_fingerprint == again.extraction_fingerprint


def test_fingerprint_changes_with_every_validity_input():
    base = make_recipe().extraction_fingerprint
    for change in (
        {"extractor": "other"},
        {"extractor_version": "6"},
        {"model": "vlm-2"},
        {"layout_schema": "4"},
        {"template_version": "2027"},
        {"options_sha256": "0" * 64},
    ):
        assert make_recipe(validity=replace(VALIDITY, **change)).extraction_fingerprint != base
    rotated = make_recipe(
        criteria=replace(CRITERIA, rotation=90), validity=replace(VALIDITY, rotation=90)
    )
    assert rotated.extraction_fingerprint != base


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("regions"),
        lambda d: d.update(status="bogus"),
        lambda d: d.update(schema_version=99),
        lambda d: d["regions"][0].update(bbox=[0, 0, 5, 5]),
        lambda d: d["validity"].pop("model"),
        lambda d: d.update(revision=0),
    ],
)
def test_corrupt_recipe_dicts_fail_closed(mutate):
    data = json.loads(make_recipe().to_json())
    mutate(data)
    with pytest.raises(RecipeError) as err:
        ExtractionRecipe.from_dict(data)
    assert err.value.code == "corrupt_recipe"


@pytest.mark.parametrize("text", ["not json", "[]", "null", '{"a":'])
def test_corrupt_recipe_json_fails_closed(text):
    with pytest.raises(RecipeError):
        ExtractionRecipe.from_json(text)


def test_model_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        FamilyCriteria("x")
    with pytest.raises(ValueError):
        FamilyCriteria("x", (("a", "b"),), rotation=45)
    with pytest.raises(ValueError):
        FamilyCriteria("x", (("a", "b"),), page_count=0)
    with pytest.raises(ValueError):
        RecipeAnchor("t", (0, 0, 1, 1), tolerance=0.9)
    with pytest.raises(ValueError):
        RecipeRegion("r", 1, (0, 0, 1, 1), "p", kind="audio")
    with pytest.raises(ValueError):
        RecipeRegion("r", 0, (0, 0, 1, 1), "p")
    with pytest.raises(ValueError):
        RecipeRegion("r", 1, (0, 0, 1, 1), "p", dpi=10)
    with pytest.raises(ValueError):
        RecipeRegion("r", 1, (0, 0, 1, 1), "p", margin=0.9)
    with pytest.raises(ValueError):
        QualityThresholds(min_region_coverage=0)
    with pytest.raises(ValueError):
        QualityThresholds(min_text_chars=-1)
    with pytest.raises(ValueError):
        make_recipe(artifact_policy="public-urls")
    with pytest.raises(ValueError):
        replace(make_recipe(), regions=())
    with pytest.raises(ValueError):
        replace(make_recipe(), regions=REGIONS + REGIONS[:1])
    with pytest.raises(ValueError):
        replace(make_recipe(), expected_counts=(("nope", 0, 1),))
    with pytest.raises(ValueError):
        replace(make_recipe(), expected_counts=(("lines", 5, 1),))
    with pytest.raises(ValueError):
        replace(make_recipe(), validity=replace(VALIDITY, rotation=90))
    with pytest.raises(ValueError):
        replace(make_recipe(), tenant="")


# ------------------------------------------------------------------- stores


def _sqlite(tmp_path):
    return SQLiteRecipeStore(tmp_path / "r.db")


@pytest.fixture(params=["memory", "json", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryRecipeStore()
    if request.param == "json":
        return JsonFileRecipeStore(tmp_path / "recipes")
    return _sqlite(tmp_path)


def test_store_contract_revisions_are_atomic_and_audited(store):
    recipe = make_recipe()
    assert store.get("t1", "acme-invoice") is None
    assert store.put_if_absent(recipe) is True
    assert store.put_if_absent(recipe) is False
    nxt = replace(recipe, revision=2, serializer="markdown")
    assert store.compare_and_set(1, nxt) is True
    assert store.compare_and_set(1, replace(recipe, revision=2)) is False
    assert store.get("t1", "acme-invoice").serializer == "markdown"
    assert store.deactivate("t1", "acme-invoice", 2, "retired") is True
    assert store.deactivate("t1", "acme-invoice", 2, "again") is False
    assert store.deactivate("t1", "missing", 1, "x") is False
    latest = store.get("t1", "acme-invoice")
    assert (latest.revision, latest.status, latest.status_reason) == (
        3,
        RecipeStatus.DEACTIVATED,
        "retired",
    )
    history = store.list_revisions("t1", "acme-invoice")
    assert [r.revision for r in history] == [1, 2, 3]
    with pytest.raises(ValueError):
        store.put_if_absent(nxt)
    with pytest.raises(ValueError):
        store.compare_and_set(5, nxt)


def test_stores_isolate_tenants(store):
    assert store.put_if_absent(make_recipe("tenant-a"))
    assert store.get("tenant-b", "acme-invoice") is None
    assert store.put_if_absent(replace(make_recipe("tenant-b"), serializer="markdown"))
    assert store.get("tenant-a", "acme-invoice").serializer == "plain-text"
    assert store.get("tenant-b", "acme-invoice").serializer == "markdown"


def test_concurrent_writers_use_cas(store):
    store.put_if_absent(make_recipe())
    base = make_recipe()
    wins = []

    def writer(i):
        wins.append(store.compare_and_set(1, replace(base, revision=2, serializer=f"s{i}")))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1
    assert len(store.list_revisions("t1", "acme-invoice")) == 2


def test_json_store_hostile_ids_cannot_escape_root(tmp_path):
    store = JsonFileRecipeStore(tmp_path / "root")
    hostile = replace(make_recipe("../../evil"), criteria=replace(CRITERIA, family_id="../../../x"))
    assert store.put_if_absent(hostile)
    produced = {p.resolve() for p in tmp_path.rglob("*") if p.is_file()}
    assert all((tmp_path / "root").resolve() in p.parents for p in produced)


def test_json_store_corruption_is_detected(tmp_path):
    store = JsonFileRecipeStore(tmp_path)
    store.put_if_absent(make_recipe())
    path = next(tmp_path.rglob("rev-00000001.json"))
    path.write_text("{broken")
    with pytest.raises(RecipeError):
        store.get("t1", "acme-invoice")
    swapped = make_recipe("other")
    path.write_text(swapped.to_json())
    with pytest.raises(RecipeError):
        store.get("t1", "acme-invoice")
    path.write_text(replace(make_recipe(), revision=7).to_json())
    with pytest.raises(RecipeError):
        store.get("t1", "acme-invoice")
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(RecipeError):
        store.list_revisions("t1", "acme-invoice")


def test_json_store_ignores_stray_files(tmp_path):
    store = JsonFileRecipeStore(tmp_path)
    store.put_if_absent(make_recipe())
    directory = next(tmp_path.rglob("rev-00000001.json")).parent
    (directory / "rev-bad.json").write_text("{}")
    assert store.get("t1", "acme-invoice").revision == 1
    assert JsonFileRecipeStore(tmp_path / "empty").get("t", "f") is None


def test_sqlite_store_persists_and_closes(tmp_path):
    first = _sqlite(tmp_path)
    first.put_if_absent(make_recipe())
    first.close()
    assert _sqlite(tmp_path).get("t1", "acme-invoice").revision == 1


# --------------------------------------------------------------- classification


def test_classifier_requires_strong_unambiguous_match():
    cls = FamilyClassifier([CRITERIA])
    assert cls.classify(fingerprint()).family_id == "acme-invoice"
    for bad in (
        fingerprint(vendor="other"),
        fingerprint(page_count=2),
        fingerprint(rotation=90),
        fingerprint(words=()),
        fingerprint(words=[("ACME INVOICE", (0.1, 0.05, 0.4, 0.1))]),  # one anchor moved
        fingerprint(
            words=[("ACME INVOICE", (0.5, 0.5, 0.9, 0.9)), ("Total", (0.6, 0.8, 0.7, 0.85))]
        ),
    ):
        decision = cls.classify(bad)
        assert decision.family_id is None
        assert decision.reason == "no_strong_match"


def test_classifier_collision_returns_no_family_and_normalizes_text():
    twin = replace(CRITERIA, family_id="acme-invoice-b")
    cls = FamilyClassifier([CRITERIA, twin])
    decision = cls.classify(fingerprint())
    assert (decision.family_id, decision.reason) == (None, "ambiguous_match")
    shouted = fingerprint(
        words=[("acme   invoice", (0.1, 0.05, 0.4, 0.1)), ("TOTAL", (0.6, 0.8, 0.7, 0.85))]
    )
    assert FamilyClassifier([CRITERIA]).classify(shouted).family_id == "acme-invoice"


def test_classifier_metadata_only_family_and_config_validation():
    meta = FamilyCriteria("meta", (("vendor", "acme"),))
    assert FamilyClassifier([meta]).classify(fingerprint(words=())).family_id == "meta"
    assert FamilyClassifier([]).classify(fingerprint()).family_id is None
    with pytest.raises(ValueError):
        FamilyClassifier([meta, meta])
    with pytest.raises(ValueError):
        FamilyClassifier([meta], min_score=0)


# ------------------------------------------------------------------- fast path


def test_recipe_hit_preserves_values_and_skips_full_detection():
    env = Env()
    env.store.put_if_absent(make_recipe())
    result = env.path.process("t1", doc(header="Invoice 8"), fingerprint())
    assert result.source == "recipe"
    assert result.regions[0].text == "Invoice 8"
    assert result.provider_calls == 2
    assert env.full_calls == 0
    assert result.extraction_fingerprint == make_recipe().extraction_fingerprint
    snap = env.path.telemetry.snapshot("t1")
    assert snap["recipe_hit"] == 1 and snap["calls_saved"] == 3
    assert env.path.telemetry.rates("t1")["cache_hit_rate"] == 1.0


def test_no_family_or_no_recipe_runs_full_detection():
    env = Env()
    assert env.path.process("t1", doc(), fingerprint(vendor="other")).fallback_reason == "no_family"
    result = env.path.process("t1", doc(), fingerprint())
    assert (result.source, result.fallback_reason) == ("full", "no_recipe")
    assert env.extractor.calls == []


def test_ambiguous_family_runs_full_detection():
    twin = replace(CRITERIA, family_id="b")
    env = Env(classifier=FamilyClassifier([CRITERIA, twin]))
    env.store.put_if_absent(make_recipe())
    assert env.path.process("t1", doc(), fingerprint()).fallback_reason == "ambiguous_match"
    assert env.extractor.calls == []


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (doc(drop="lines"), "region_coverage"),  # table moved away: missing region
        (doc(lines=9), "count_out_of_range:lines"),  # added table rows
        (doc(header="  "), "empty_region:header"),
    ],
)
def test_verification_runs_on_hit_and_template_drift_never_publishes(document, reason):
    env = Env()
    env.store.put_if_absent(make_recipe())
    if "lines" not in document:
        document["lines"] = Exception("region missing")  # extractor cannot find the crop
    result = env.path.process("t1", document, fingerprint())
    assert result.source == "full"
    assert result.fallback_reason == "verification_failed"
    assert env.full_calls == 1  # exactly one fallback run, no loop
    stored = env.store.get("t1", "acme-invoice")
    assert stored.status is RecipeStatus.QUARANTINED
    snap = env.path.telemetry.snapshot("t1")
    assert snap["false_hit"] == 1 and snap["fallback"] == 1
    # Quarantined recipe is not used again.
    again = env.path.process("t1", doc(), fingerprint())
    assert again.fallback_reason == "recipe_quarantined"
    assert env.full_calls == 2
    assert reason  # parametrized documentation of the drift kind


def test_verifier_reports_specific_reasons():
    verifier = CountCoverageVerifier({"lines": (3, 2)})
    recipe = make_recipe()
    report = verifier.verify(recipe, [ExtractedRegion("header", "x", 1)])
    assert not report.ok and "region_coverage" in report.reasons
    report = verifier.verify(
        recipe,
        [
            ExtractedRegion("header", "x", 1),
            ExtractedRegion("lines", "y", 9, (9, 2)),
            ExtractedRegion("extra", "z", 1),
        ],
    )
    assert {"count_out_of_range:lines", "shape_mismatch:lines", "unexpected_region"} <= set(
        report.reasons
    )
    dup = verifier.verify(recipe, [ExtractedRegion("header", "x", 1)] * 2)
    assert "duplicate_region" in dup.reasons
    lenient = replace(recipe, thresholds=QualityThresholds(min_region_coverage=0.5))
    assert verifier.verify(lenient, [ExtractedRegion("header", "x", 1)]).ok


def test_provider_exception_fails_closed_without_leaking_message():
    env = Env()
    env.store.put_if_absent(make_recipe())
    bad = doc()
    bad["lines"] = RuntimeError("secret customer text 123")
    result = env.path.process("t1", bad, fingerprint())
    assert result.source == "full" and result.fallback_reason == "verification_failed"
    assert "secret" not in repr(result.fallback_reason)


def test_stale_engine_fingerprint_invalidates_recipe_and_falls_back():
    env = Env(engine=replace(VALIDITY, extractor_version="6"))
    env.store.put_if_absent(make_recipe())
    result = env.path.process("t1", doc(), fingerprint())
    assert result.fallback_reason == "stale_validity"
    assert env.store.get("t1", "acme-invoice").status is RecipeStatus.DEACTIVATED
    assert env.path.telemetry.snapshot()["stale_invalidation"] == 1
    assert env.extractor.calls == []


def test_tenants_do_not_share_recipes():
    env = Env()
    env.store.put_if_absent(make_recipe("tenant-a"))
    assert env.path.process("tenant-b", doc(), fingerprint()).fallback_reason == "no_recipe"
    assert env.path.process("tenant-a", doc(), fingerprint()).source == "recipe"


def test_foreign_tenant_recipe_from_store_is_rejected():
    class LeakyStore(InMemoryRecipeStore):
        def get(self, tenant, family_id):
            return make_recipe("someone-else")

    env = Env(store=LeakyStore())
    assert env.path.process("t1", doc(), fingerprint()).fallback_reason == "no_recipe"


def test_cancellation_before_and_during_processing():
    env = Env()
    env.store.put_if_absent(make_recipe())
    token = CancellationToken()
    token.cancel()
    with pytest.raises(ExtractionCancelled):
        env.path.process("t1", doc(), fingerprint(), cancel=token)
    live = CancellationToken()

    class Cancelling(Extractor):
        def extract_region(self, document, region):
            live.cancel()
            return super().extract_region(document, region)

    env.path._regions = Cancelling()
    with pytest.raises(ExtractionCancelled):
        env.path.process("t1", doc(), fingerprint(), cancel=live)
    assert env.store.get("t1", "acme-invoice").active  # cancel never quarantines
    assert env.full_calls == 0


def test_cancellation_stops_fallback_too():
    env = Env()
    token = CancellationToken()
    env.full_calls = 0

    class Full:
        def __call__(self, document):
            token.cancel()
            return []

    env.path._full = Full()
    token2 = CancellationToken()
    token2.cancel()
    with pytest.raises(ExtractionCancelled):
        env.path.process("t1", doc(), fingerprint(vendor="zzz"), cancel=token2)


def test_verifier_is_mandatory_at_construction():
    with pytest.raises(ValueError, match="verifier"):
        RecipeFastPath(
            classifier=FamilyClassifier([CRITERIA]),
            store=InMemoryRecipeStore(),
            region_extractor=Extractor(),
            verifier=None,
            full_extractor=lambda d: [],
            engine_validity=VALIDITY,
        )


def test_telemetry_rates_with_no_traffic_and_global_totals():
    telemetry = RecipeTelemetry()
    assert telemetry.rates()["false_hit_rate"] == 0.0
    telemetry.record("a", "documents")
    telemetry.record("b", "documents", 2)
    assert telemetry.snapshot()["documents"] == 3
    assert telemetry.snapshot("zzz") == {}


# --------------------------------------------------------------------- learning


def page(text="ok", provider="tesseract", status=PageExtractionStatus.EXTRACTED):
    return PageExtractionOutcome(1, status, text=text, provider=provider)


def learner(store, pages, *, approve=True, confirm=True, required=True):
    verifier = EvidenceAllowlistVerifier(extraction_evidence_hash(p) for p in pages)
    return RecipeLearner(
        store,
        verifier,
        sample_confirmer=lambda recipe: confirm,
        approval_required=required,
        approver=(lambda recipe: approve) if required else None,
        clock=lambda: 0,
    )


def test_learning_stores_only_after_gate_sample_and_approval():
    store = InMemoryRecipeStore()
    good = [page()]
    recipe = learner(store, good).learn(make_recipe(), good)
    assert recipe.revision == 1 and recipe.provenance.approval == "host-approved"
    assert recipe.provenance.verifier_id == "evidence-allowlist"
    assert recipe.provenance.created_at == "1970-01-01T00:00:00Z"
    # Relearning appends revision 2 rather than overwriting.
    again = learner(store, good, required=False).learn(make_recipe(), good)
    assert again.revision == 2 and again.provenance.approval == "automatic"
    assert len(store.list_revisions("t1", "acme-invoice")) == 2


@pytest.mark.parametrize(
    ("kwargs", "pages", "code"),
    [
        ({}, [], "no_verified_run"),
        ({}, "unverified", "unverified_run"),  # VLM guess the host never approved
        ({}, "incomplete", "unverified_run"),
        ({"confirm": False}, "good", "sample_not_confirmed"),
        ({"approve": False}, "good", "not_approved"),
    ],
)
def test_learning_refuses_unverified_or_unapproved_runs(kwargs, pages, code):
    store = InMemoryRecipeStore()
    good = [page()]
    named = {
        "good": good,
        "unverified": [page("hallucinated", provider="vlm")],
        "incomplete": [page(status=PageExtractionStatus.NEEDS_OCR)],
    }
    run = named[pages] if isinstance(pages, str) else pages
    allow = good if pages != "incomplete" else run
    with pytest.raises(RecipeLearningRefused) as err:
        learner(store, allow, **kwargs).learn(make_recipe(), run)
    assert err.value.code == code
    assert store.get("t1", "acme-invoice") is None


def test_learning_detects_concurrent_update_and_needs_approver():
    store = InMemoryRecipeStore()
    good = [page()]
    lrn = learner(store, good)
    store.put_if_absent(make_recipe())
    real_get = store.get
    stale = real_get("t1", "acme-invoice")
    store.get = lambda tenant, family: stale
    store.compare_and_set(1, replace(stale, revision=2))
    with pytest.raises(RecipeLearningRefused) as err:
        lrn.learn(make_recipe(), good)
    assert err.value.code == "concurrent_update"
    with pytest.raises(ValueError, match="approver"):
        RecipeLearner(store, EvidenceAllowlistVerifier([]), sample_confirmer=lambda r: True)


def test_learned_recipe_round_trips_through_fast_path():
    env = Env()
    good = [page()]
    learner(env.store, good, required=False).learn(make_recipe(), good)
    assert env.path.process("t1", doc(), fingerprint()).source == "recipe"


def test_quarantine_threshold_tolerates_isolated_drift_and_resets_on_success():
    env = Env()
    env.path._quarantine_after = 2
    env.store.put_if_absent(make_recipe())
    assert (
        env.path.process("t1", doc(lines=9), fingerprint()).fallback_reason == "verification_failed"
    )
    assert env.store.get("t1", "acme-invoice").active  # one odd document does not kill the family
    assert (
        env.path.process("t1", doc(), fingerprint()).source == "recipe"
    )  # success resets the count
    env.path.process("t1", doc(lines=9), fingerprint())
    assert env.store.get("t1", "acme-invoice").active
    env.path.process("t1", doc(lines=9), fingerprint())
    assert env.store.get("t1", "acme-invoice").status is RecipeStatus.QUARANTINED
    with pytest.raises(ValueError):
        RecipeFastPath(
            classifier=env.classifier,
            store=env.store,
            region_extractor=env.extractor,
            verifier=CountCoverageVerifier(),
            full_extractor=lambda d: [],
            engine_validity=VALIDITY,
            quarantine_after=0,
        )
