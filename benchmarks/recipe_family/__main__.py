"""``python -m benchmarks.recipe_family`` prints the JSON report."""

from __future__ import annotations

import json
import sys

from benchmarks.recipe_family.runner import run_benchmark

if __name__ == "__main__":
    json.dump(run_benchmark(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
