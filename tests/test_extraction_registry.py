"""Registry: explicit registration, selection, fallback, timeout, cancellation."""

from __future__ import annotations

import asyncio

import pytest

from ragnor.extraction import (
    Capability,
    ExtractionConfigError,
    ExtractionError,
    ExtractionStatus,
    Mode,
    ProviderRegistry,
    StructuredExtractor,
)
from tests.extraction_doubles import FakeProvider

PDF = "application/pdf"
STRUCT = [Capability.TEXT, Capability.STRUCTURE, Capability.TABLES]


def make(registry=None, pid="fake", behavior="ok", **kw):
    registry = registry or ProviderRegistry()
    made: list[FakeProvider] = []

    def factory(opts):
        provider = FakeProvider(opts, behavior=behavior)
        provider.id = pid
        made.append(provider)
        return provider

    kw.setdefault("version", "1.0")
    registry.register(pid, factory, formats=[PDF], capabilities=kw.pop("caps", STRUCT), **kw)
    return registry, made


def test_register_validates_identity_and_rejects_duplicates():
    registry, _ = make()
    with pytest.raises(ExtractionConfigError) as err:
        make(registry)
    assert err.value.code == "duplicate_provider"
    for kwargs in (
        {"pid": "Bad Id"},
        {"pid": "x", "caps": ["telepathy"]},
        {"pid": "y", "caps": []},
        {"pid": "z", "version": ""},
    ):
        with pytest.raises(ExtractionConfigError, match="invalid_provider"):
            make(ProviderRegistry(), **kwargs)
    with pytest.raises(ExtractionConfigError, match="invalid_provider"):
        registry.register("w", "not callable", formats=[PDF], capabilities=[Capability.TEXT])
    with pytest.raises(ExtractionConfigError, match="invalid_provider"):
        registry.register("v", lambda o: None, formats=[], capabilities=[Capability.TEXT])
    assert registry.ids() == ["fake"]
    registry.unregister("fake")
    with pytest.raises(ExtractionConfigError, match="unknown_provider"):
        registry.unregister("fake")


def test_selection_is_deterministic_and_fails_closed():
    registry = ProviderRegistry()
    make(registry, "b-struct", priority=10)
    make(registry, "a-struct", priority=10)
    make(registry, "plain", caps=[Capability.TEXT], priority=1)
    assert [s.provider_id for s in registry.select(media_type=PDF)] == [
        "a-struct",
        "b-struct",
        "plain",
    ]
    assert [s.provider_id for s in registry.select(media_type=PDF, mode="text")] == [
        "plain",
        "a-struct",
        "b-struct",
    ]
    assert [s.provider_id for s in registry.select(media_type=PDF, mode=Mode.STRUCTURED)] == [
        "a-struct",
        "b-struct",
    ]
    assert [s.provider_id for s in registry.select(media_type=PDF, providers=["plain"])] == [
        "plain"
    ]
    with pytest.raises(ExtractionConfigError, match="unsupported_format"):
        registry.select(media_type="image/png")
    with pytest.raises(ExtractionConfigError, match="incompatible_capabilities"):
        registry.select(media_type=PDF, require=[Capability.VLM])
    with pytest.raises(ExtractionConfigError, match="invalid_mode"):
        registry.select(media_type=PDF, mode="magic")
    with pytest.raises(ExtractionConfigError, match="unknown_provider"):
        registry.select(media_type=PDF, providers=["ghost"])
    text_only = ProviderRegistry()
    make(text_only, "plain", caps=[Capability.TEXT])
    with pytest.raises(ExtractionConfigError, match="incompatible_capabilities"):
        text_only.select(media_type=PDF, mode="structured")


def test_remote_providers_need_explicit_host_enablement():
    registry, made = make(remote=True)
    with pytest.raises(ExtractionConfigError, match="remote_not_enabled"):
        registry.select(media_type=PDF)
    with pytest.raises(ExtractionConfigError, match="remote_not_enabled"):
        asyncio.run(registry.extract(b"%PDF", revision="r", name="a.pdf"))
    assert made == []  # never instantiated, so no content could be sent
    registry.enable_remote("fake")
    assert registry.select(media_type=PDF)
    registry.disable_remote("fake")
    with pytest.raises(ExtractionConfigError):
        registry.select(media_type=PDF)
    with pytest.raises(ExtractionConfigError, match="unknown_provider"):
        registry.enable_remote("ghost")


def test_options_are_validated_before_the_factory_runs():
    def validate(opts):
        if "dpi" in opts and not 50 <= opts["dpi"] <= 600:
            raise ValueError("bad dpi")

    registry, made = make(validate_options=validate)
    with pytest.raises(ExtractionConfigError, match="invalid_options"):
        registry.create("fake", {"dpi": 5})
    assert made == []
    assert registry.create("fake", {"dpi": 300}).options == {"dpi": 300}
    with pytest.raises(ExtractionConfigError, match="unknown_provider"):
        registry.create("ghost")


def test_provider_must_match_registered_identity():
    registry = ProviderRegistry()
    registry.register(
        "fake",
        FakeProvider,
        formats=[PDF],
        capabilities=[Capability.TEXT],
        version="2",
    )
    with pytest.raises(ExtractionConfigError, match="invalid_provider"):
        registry.create("fake")  # instance reports version 1.0, spec says 2
    registry2 = ProviderRegistry()
    registry2.register("fake", lambda o: object(), formats=[PDF], capabilities=[Capability.TEXT])
    with pytest.raises(ExtractionConfigError, match="invalid_provider"):
        registry2.create("fake")


async def test_extract_success_records_attempt_and_closes_provider():
    registry, made = make()
    result = await registry.extract(b"%PDF-1", revision="rev", name="a.pdf")
    assert result.complete and result.attempts[0].status == "succeeded"
    assert result.attempts[0].provider_version == "1.0"
    assert made[0].closed == 1


async def test_unsupported_format_is_rejected_before_any_provider_runs():
    registry, made = make()
    with pytest.raises(ExtractionConfigError, match="unsupported_format"):
        await registry.extract(b"x", revision="r", name="a.docx")
    assert made == []
    with pytest.raises(ExtractionConfigError, match="invalid_options"):
        await registry.extract(b"x", revision="r", name="a.pdf", timeout=0)


async def test_fallback_after_error_timeout_invalid_and_failed_results():
    registry = ProviderRegistry()
    for pid, behavior, prio in (
        ("p1-boom", "boom", 1),
        ("p2-hang", "hang", 2),
        ("p3-garbage", "garbage", 3),
        ("p4-failed", "failed", 4),
        ("p5-ok", "ok", 5),
    ):
        make(registry, pid, behavior, priority=prio)
    result = await registry.extract(b"x", revision="r", name="a.pdf", timeout=0.05)
    assert [a.status for a in result.attempts] == [
        "failed",
        "timed_out",
        "failed",
        "failed",
        "succeeded",
    ]
    assert [a.reason for a in result.attempts][:4] == [
        "provider_error",
        "timeout",
        "invalid_output",
        "provider_error",
    ]


async def test_all_failures_raise_sanitized_error_without_provider_text():
    registry, _ = make(behavior="boom")
    with pytest.raises(ExtractionError) as err:
        await registry.extract(b"x", revision="r", name="a.pdf")
    assert err.value.code == "fallback_exhausted"
    assert "secret-api-key" not in str(err.value) + repr(err.value.attempts)
    assert err.value.attempts[0].status == "failed"


async def test_partial_extraction_is_fail_closed_unless_accepted():
    registry, _ = make(behavior="partial")
    with pytest.raises(ExtractionError) as err:
        await registry.extract(b"x", revision="r", name="a.pdf")
    assert err.value.code == "partial_only"
    result = await registry.extract(b"x", revision="r", name="a.pdf", accept_partial=True)
    assert result.status is ExtractionStatus.PARTIAL
    assert result.attempts[-1].reason == "partial_output"


async def test_partial_best_candidate_kept_when_later_provider_fails():
    registry = ProviderRegistry()
    make(registry, "a-partial", "partial", priority=1)
    make(registry, "b-boom", "boom", priority=2)
    result = await registry.extract(b"x", revision="r", name="a.pdf", accept_partial=True)
    assert result.status is ExtractionStatus.PARTIAL
    assert [a.status for a in result.attempts] == ["partial", "failed"]


async def test_cancellation_closes_the_provider_and_propagates():
    registry, made = make(behavior="hang")
    task = asyncio.create_task(registry.extract(b"x", revision="r", name="a.pdf", timeout=30))
    for _ in range(100):
        if made and made[0].started.is_set():
            break
        await asyncio.sleep(0.01)
    assert made[0].started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert made[0].closed == 1


async def test_provider_without_aclose_and_failing_aclose_are_tolerated():
    registry = ProviderRegistry()

    class Bare:
        id = "bare"
        version = "1"
        options = {}

        async def extract(self, data, *, name="", revision):
            from tests.extraction_doubles import sample_result

            return sample_result(revision)

    class BadClose(Bare):
        id = "bad-close"

        async def aclose(self):
            raise RuntimeError("close failed")

    registry.register(
        "bare",
        lambda o: Bare(),
        formats=[PDF],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
        priority=1,
    )
    registry.register(
        "bad-close",
        lambda o: BadClose(),
        formats=[PDF],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
        priority=2,
    )
    assert (await registry.extract(b"x", revision="r", name="a.pdf")).complete
    assert (
        await registry.extract(b"x", revision="r", name="a.pdf", providers=["bad-close"])
    ).complete


async def test_structured_extractor_opt_in_text_and_documents():
    registry, _ = make()
    extractor = StructuredExtractor(registry, mode="structured", timeout=5)
    assert extractor.id == "structured"
    result = await extractor.aextract_result(b"%PDF", name="a.pdf")
    assert result.region("r-table") is not None
    docs = await extractor.aextract_documents(b"%PDF", source_name="a.pdf", external_id="e1")
    assert [d.metadata["region_id"] for d in docs] == ["r-head", "r-body", "r-table"]
    assert all(d.extractor == "structured" and d.external_id == "e1" for d in docs)


def test_structured_extractor_sync_views_and_loop_guard():
    registry, _ = make()
    extractor = StructuredExtractor(
        registry, providers=["fake"], provider_options={"fake": {"dpi": 200}}
    )
    pages = extractor.extract_pages(b"%PDF", name="a.pdf")
    assert pages[0][0] == 1 and "Results" in pages[0][1]
    assert "Body text" in extractor.extract(b"%PDF", name="a.pdf")
    with pytest.raises(ExtractionConfigError):
        StructuredExtractor(registry, providers=["ghost"])
    with pytest.raises(ExtractionConfigError):
        StructuredExtractor(registry, provider_options={"ghost": {}})

    async def inside_loop():
        extractor.extract_result(b"%PDF", name="a.pdf")

    with pytest.raises(ExtractionConfigError):
        asyncio.run(inside_loop())
