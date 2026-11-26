"""Benchmark gates (deterministic, offline)."""


from benchmarks.recipe_family.runner import run_benchmark as recipe_benchmark


def test_recipe_family_benchmark_saves_calls_without_accuracy_regression():
    report = recipe_benchmark(latency=0.0)
    assert report["no_accuracy_regression"] is True
    assert report["recipe"]["accuracy"] == 1.0 and report["no_recipe"]["accuracy"] == 1.0
    assert report["provider_call_reduction"] > 0.3
    assert report["drifted_documents"] > 0
    # Every drifted document was caught by verification and fell back; none published.
    assert report["recipe"]["telemetry"]["false_hit"] == report["drifted_documents"]
    assert report["recipe"]["telemetry"]["fallback"] == report["drifted_documents"]
    assert report["recipe"]["rates"]["cache_hit_rate"] > 0.9
