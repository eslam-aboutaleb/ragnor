"""Publishing a benchmark must not publish its source data or raw artifacts."""

from scripts.check_benchmark_payloads import forbidden_payloads


def test_corpus_and_raw_reports_are_rejected_while_code_and_summary_are_allowed():
    forbidden = [
        "benchmarks/data/private.pdf",
        "benchmarks/fixtures/labels.json",
        "benchmarks/results.parquet",
        "docs/qualification/raw-run.json",
        "benchmarks/screenshot.PNG",
    ]
    allowed = [
        "benchmarks/run_quality.py",
        "benchmarks/requirements.txt",
        "docs/qualification/pdf-quality-results.md",
        "tests/test_pdf_quality_metrics.py",
    ]
    assert forbidden_payloads(forbidden + allowed) == forbidden
