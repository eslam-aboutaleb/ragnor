"""Reject corpus, annotation and raw report payloads in the Git index."""

from __future__ import annotations

import subprocess
from pathlib import PurePosixPath


def forbidden_payloads(paths: list[str]) -> list[str]:
    result = []
    for name in paths:
        path = PurePosixPath(name)
        if name.startswith(("benchmarks/data/", "benchmarks/fixtures/")) or (
            name.startswith(("benchmarks/", "docs/qualification/"))
            and path.suffix.lower()
            in {".pdf", ".png", ".jpg", ".jpeg", ".parquet", ".json", ".jsonl", ".zip"}
        ):
            result.append(name)
    return result


def main() -> int:
    indexed = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607 -- host Git, fixed read-only arguments
        check=True,
        capture_output=True,
    ).stdout
    rejected = forbidden_payloads([name.decode() for name in indexed.split(b"\0") if name])
    if rejected:
        print("Benchmark corpus/annotation/raw-report files must stay outside Git:")
        print("\n".join(rejected))
        return 1
    print("No benchmark data payloads in the Git index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
