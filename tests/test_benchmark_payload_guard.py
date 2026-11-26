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
        "docs/extraction-benchmarks.md",
        "tests/test_pdf_quality_metrics.py",
        "benchmarks/pdf_extraction/fixtures/documents/plain-digital-report.pdf",
        "benchmarks/pdf_extraction/fixtures/manifest.jsonl",
    ]
    assert forbidden_payloads(forbidden + allowed) == forbidden


def test_review_packets_and_transcripts_cannot_escape_data_guard():
    paths = [
        "benchmarks/review/index.html",
        "benchmarks/labels.tsv",
        "benchmarks/archive.tar",
        "benchmarks/page.tiff",
        "docs/qualification/page.reference.txt",
        "benchmarks/transcript.txt",
    ]
    assert forbidden_payloads(paths) == paths


def test_repository_index_has_no_forbidden_payloads():
    import subprocess

    tracked = (
        subprocess.run(  # noqa: S603 -- fixed read-only Git call
            ["git", "ls-files", "-z"],  # noqa: S607
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    assert forbidden_payloads([name for name in tracked if name]) == []
