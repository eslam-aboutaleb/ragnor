"""Ground-truth PDF extraction benchmark suite (issue #60 / EX-02).

A versioned, synthetic, redistributable corpus of challenging pages with
hand-checked annotations, a pluggable provider runner, quality/cost metrics
and CI non-regression gates. No PDF contents are ever printed or persisted
in reports: only ids, counts, hashes and metric values.
"""

from benchmarks.pdf_extraction.manifest import (
    SCHEMA_VERSION,
    Manifest,
    PageAnnotation,
    load_manifest,
)
from benchmarks.pdf_extraction.metrics import (
    char_error_rate,
    score_page,
    word_error_rate,
)
from benchmarks.pdf_extraction.providers import ExtractionProvider, ProviderPage
from benchmarks.pdf_extraction.runner import run_benchmark

__all__ = [
    "ExtractionProvider",
    "Manifest",
    "PageAnnotation",
    "ProviderPage",
    "SCHEMA_VERSION",
    "char_error_rate",
    "load_manifest",
    "run_benchmark",
    "score_page",
    "word_error_rate",
]
