"""Encoding shared by evidence citations and source lookup tools."""

from __future__ import annotations

import re
from urllib.parse import quote, unquote

_SPAN = re.compile(r"^(.*)\[(\d+):(\d+|None)\]$")


def source_reference(
    source_id: str,
    document_id: str,
    revision: str,
    start: int | None = None,
    end: int | None = None,
) -> str:
    base = f"{quote(source_id, safe='')}@{quote(revision, safe='')}:{quote(document_id, safe='')}"
    return f"{base}[{start}:{end}]" if start is not None else base


def parse_source_reference(ref: str) -> tuple[str, str, str | None, int | None, int | None]:
    start = end = None
    match = _SPAN.fullmatch(ref)
    if match:
        ref, raw_start, raw_end = match.groups()
        start = int(raw_start)
        end = None if raw_end == "None" else int(raw_end)
    revision = None
    if "@" in ref and ":" in ref:
        source, rest = ref.split("@", 1)
        revision, document = rest.split(":", 1)
    elif ":" in ref:
        source, document = ref.split(":", 1)
    else:
        source = document = ref
    return unquote(source), unquote(document), unquote(revision) if revision else None, start, end
