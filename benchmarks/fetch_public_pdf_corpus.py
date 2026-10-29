"""Bounded checksum downloads and upstream text conversion for local benchmarks."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def reference_text(elements: list[dict]) -> str:
    """Use upstream references, never the extractor under evaluation as labels."""
    parts = []
    for element in elements:
        content = element["content"]
        if content.get("text"):
            parts.append(content["text"])
        elif content.get("html"):
            parser = _HTMLText()
            parser.feed(content["html"])
            parts.append(" ".join(parser.parts))
    return " ".join(" ".join(parts).split())


def _download(url: str, path: Path, checksum: str, limit: int) -> bytes:
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:  # noqa: S310 -- pinned HTTPS source only
            data = response.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("Public corpus size/checksum validation failed")
    # Only publish a local file after validating the full downloaded object.
    path.write_bytes(data)
    return data
