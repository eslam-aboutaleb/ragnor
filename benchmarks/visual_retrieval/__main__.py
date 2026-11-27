"""``python -m benchmarks.visual_retrieval`` prints the JSON report."""

from __future__ import annotations

import json
import sys

from benchmarks.visual_retrieval.runner import run_benchmark

if __name__ == "__main__":
    report = run_benchmark()
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
