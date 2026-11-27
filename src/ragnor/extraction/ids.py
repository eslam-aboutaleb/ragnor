"""Deterministic region identifiers."""

from __future__ import annotations

import hashlib


def deterministic_region_id(revision: str, page: int, key: str) -> str:
    """Stable id from document revision, physical page and a provider region key.

    The same revision/page/key always yields the same id, across processes and
    runs, so citations stay resolvable after a re-extraction of unchanged bytes.
    A new revision yields new ids.
    """
    if not revision or type(page) is not int or page < 1 or not isinstance(key, str) or not key:
        raise ValueError("Region ids need a revision, a positive page and a key")
    digest = hashlib.sha256(f"{revision}\x1f{page}\x1f{key}".encode()).hexdigest()
    return f"rg-{digest[:24]}"
