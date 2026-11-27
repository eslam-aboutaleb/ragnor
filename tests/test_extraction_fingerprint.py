"""Fingerprints: deterministic reindex on provider change, no secret leakage."""

from __future__ import annotations

import json

from ragnor.chunking.sliding_window import SlidingWindowChunker
from ragnor.embeddings.dimensions import register_dimension
from ragnor.extraction import (
    Capability,
    ProviderRegistry,
    StructuredExtractor,
    extractor_fingerprint,
    fingerprint_hash,
    provider_fingerprint,
    safe_provider_options,
)
from ragnor.ingestion import build_snapshot, snapshot_matches
from ragnor.ingestion.versioning import compute_source_hash
from tests.extraction_doubles import FakeProvider

register_dimension("ragnor-extraction-test-model", 4)


class _Settings:
    embedding_model = "ragnor-extraction-test-model"


def registry(version="1.0", **_):
    reg = ProviderRegistry()
    reg.register(
        "fake",
        FakeProvider,
        formats=["application/pdf"],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
        version=version,
    )
    return reg


def snapshot(extractor):
    return build_snapshot("a" * 64, _Settings(), SlidingWindowChunker(), extractor=extractor)


def test_unchanged_configuration_is_idempotent_and_changes_trigger_reindex():
    base = snapshot(
        StructuredExtractor(registry(), provider_options={"fake": {"recipe": "v1", "model": "m1"}})
    )
    same = snapshot(
        StructuredExtractor(registry(), provider_options={"fake": {"recipe": "v1", "model": "m1"}})
    )
    assert snapshot_matches(same, base)
    for extractor in (
        StructuredExtractor(
            registry(version="2.0"), provider_options={"fake": {"recipe": "v1", "model": "m1"}}
        ),
        StructuredExtractor(registry(), provider_options={"fake": {"recipe": "v2", "model": "m1"}}),
        StructuredExtractor(registry(), provider_options={"fake": {"recipe": "v1", "model": "m2"}}),
        StructuredExtractor(
            registry(), provider_options={"fake": {"recipe": "v1", "model": "m1"}}, mode="auto"
        ),
        StructuredExtractor(
            registry(),
            provider_options={"fake": {"recipe": "v1", "model": "m1"}},
            accept_partial=True,
        ),
    ):
        assert not snapshot_matches(snapshot(extractor), base)


def test_option_ordering_does_not_change_the_fingerprint():
    a = extractor_fingerprint(
        StructuredExtractor(registry(), provider_options={"fake": {"a": 1, "b": 2}})
    )
    b = extractor_fingerprint(
        StructuredExtractor(registry(), provider_options={"fake": {"b": 2, "a": 1}})
    )
    assert a == b and len(a) == 64


def test_secrets_prompts_bytes_and_url_credentials_never_reach_snapshots():
    options = {
        "model": "gpt-x",
        "api_key": "sk-live-SECRET",
        "prompt": "You are a confidential extractor",
        "system_message": "do things",
        "raw": b"%PDF-1.7 raw bytes",
        "endpoint": "https://user:hunter2@api.example.com:8443/v1?key=abc#frag",
        "long_text": "x" * 1000,
        "nested": {"password": "pw", "authorization": "Bearer abc", "ok": 3},
        "headers": {"X-Key": "k"},
        "list": [1, b"bytes", "s"],
        "max_tokens": 512,
    }
    reg = registry()
    extractor = StructuredExtractor(reg, provider_options={"fake": options})
    blob = json.dumps(snapshot(extractor).extractor_options) + extractor_fingerprint(extractor)
    for leaked in (
        "SECRET",
        "confidential",
        "raw bytes",
        "hunter2",
        "abc",
        "do things",
        "xxxx",
        "Bearer",
    ):
        assert leaked not in blob
    safe = safe_provider_options(options)
    assert safe["model"] == "gpt-x" and safe["max_tokens"] == 512  # real tunables stay
    assert safe["endpoint"] == "https://api.example.com:8443/v1"
    assert safe["nested"] == {"ok": 3}
    assert safe["list"] == [1, "s"]
    assert "long_text" not in safe and "headers" not in safe


def test_provider_fingerprint_and_hash_helpers():
    fp = provider_fingerprint(FakeProvider({"k": 1, "api_key": "z"}))
    assert fp == {"id": "fake", "version": "1.0", "options": {"k": 1}}
    assert fingerprint_hash(fp) == fingerprint_hash(dict(reversed(list(fp.items()))))
    assert safe_provider_options(None) == {}
    assert (
        safe_provider_options(
            {"f": float("nan"), "deep": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}}
        )
        is not None
    )
    assert extractor_fingerprint(object())  # plain objects are fingerprintable


def test_extraction_identity_feeds_the_manifest_source_hash():
    from ragnor.extraction import to_raw_documents
    from tests.extraction_doubles import sample_result

    docs = to_raw_documents(sample_result("rev-a"), source_name="a.pdf")
    other = to_raw_documents(sample_result("rev-b"), source_name="a.pdf")
    assert compute_source_hash(docs) != compute_source_hash(other)
    assert compute_source_hash(docs) == compute_source_hash(
        to_raw_documents(sample_result("rev-a"), source_name="a.pdf")
    )
