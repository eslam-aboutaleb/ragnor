"""Benchmark gates (deterministic, offline)."""

import pytest

pytest.importorskip("PIL")

from benchmarks.visual_retrieval.runner import run_benchmark as visual_benchmark  # noqa: E402


def test_visual_benchmark_wins_on_text_free_pages_and_reports_costs():
    report = visual_benchmark(pages=24)
    assert report["text_only"]["recall_at_5"] == 0.0
    assert report["visual"]["recall_at_5"] >= 0.95 and report["fused"]["recall_at_5"] >= 0.95
    assert report["visual"]["mrr"] > 0.8
    assert report["provider_is_fake"] is True
    assert report["index_memory_bytes_estimate"] > 0 and report["patches_per_page"] > 1
    assert report["false_positive_rate"] == 0.0
    extra = visual_benchmark(pages=10, user_queries=[("c0b0", "doc0-p1")])
    assert extra["queries"] == 11
