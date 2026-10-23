"""Reproducible tokenizer accuracy and chunking performance profile.

The values are observations printed by CI, not asserted universal speed or
accuracy guarantees. No raw customer text is read or transmitted.
"""

from __future__ import annotations

import json
from functools import partial
import time
import tracemalloc

from ragnor.chunking import SlidingWindowChunker, TokenAwareChunker
from ragnor.evidence.tiktoken_adapter import TiktokenTokenizer
from ragnor.evidence.tokenizer import ApproximateTokenizer

SAMPLES = {
    "english": "Customer refund policy covers international shipping and returns. " * 4,
    "arabic": "سياسة استرداد المنتجات تحمي العملاء عند الشراء عبر الإنترنت. " * 4,
    "cjk": "我们提供透明的退货政策并且需要完整引用来源文件。 " * 4,
    "json": '{"refund": true, "count": 13, "items": ["a", "b", "c"]}\n' * 4,
    "code": 'async def load():\n    return {"scope": "tenant", "valid": True}\n' * 4,
    "unicode": "👩🏻‍💻 👨🏽‍🚀 🇪🇬 é é 🧪 " * 10,
}


def _profile(callable_fn):
    tracemalloc.start()
    start = time.perf_counter()
    produced = callable_fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return len(produced), round(elapsed * 1000, 3), peak


def main() -> None:
    exact = TiktokenTokenizer(encoding_name="cl100k_base")
    estimate = ApproximateTokenizer()
    values = []
    for kind, text in SAMPLES.items():
        exact_count = exact.count_tokens(text)
        approx = estimate.count_tokens(text)
        exact_chunker = TokenAwareChunker(exact, max_tokens=48, overlap_tokens=4)
        legacy_chunker = SlidingWindowChunker(chunk_size=40, overlap=6)
        exact_parts, exact_ms, exact_mem = _profile(partial(exact_chunker.chunk, text))
        word_parts, word_ms, word_mem = _profile(partial(legacy_chunker.chunk, text))
        row = {
            "sample": kind,
            "characters": len(text),
            "model": exact.name,
            "exact_tokens": exact_count,
            "estimated_tokens": approx,
            "relative_error": round(abs(exact_count - approx) / max(exact_count, 1), 5),
            "exact_chunks": exact_parts,
            "word_chunks": word_parts,
            "exact_duration_ms": exact_ms,
            "word_duration_ms": word_ms,
            "exact_peak_bytes": exact_mem,
            "word_peak_bytes": word_mem,
        }
        assert exact_parts > 0
        assert word_parts > 0
        values.append(row)
    print(json.dumps(values, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
