"""Shared secret-stripping utility for canonical hash computation.

This module exists solely to break the circular import between
:mod:`ragnor.embeddings.profile` and :mod:`ragnor.ingestion.versioning`.
Both modules need :func:`safe_config` but neither can import the other
at module load time.

Do not import from :mod:`ragnor.embeddings` or :mod:`ragnor.ingestion`
from this module.
"""

from __future__ import annotations

import math
from typing import Any

#: Keys (lowercased) that are stripped from dicts before hashing.
_SECRET_KEYS = {
    "token",
    "headers",
    "cookies",
    "secret",
    "password",
    "credential",
    "authorization",
    "api_key",
    "apikey",
}

_SECRET_SUFFIXES = ("_token",)
_SECRET_SUBSTRINGS = ("secret", "password", "credential", "authorization", "api_key", "apikey")


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in _SECRET_KEYS:
        return True
    if any(key_lower.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return True
    return any(part in key_lower for part in _SECRET_SUBSTRINGS)


def safe_config(value: Any) -> Any:
    """Copy JSON configuration only, omitting secrets and arbitrary objects.

    Never use repr/default=str: clients and credentials may expose secrets there.

    This is a standalone copy of the same function in
    :mod:`ragnor.ingestion.versioning`; keep them in sync.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            key: safe_config(item)
            for key, item in value.items()
            if isinstance(key, str)
            and not _is_secret_key(key)
            and isinstance(item, (str, bool, int, float, dict, list, tuple, type(None)))
        }
    if isinstance(value, (list, tuple)):
        return [
            safe_config(item)
            for item in value
            if isinstance(item, (str, bool, int, float, dict, list, tuple, type(None)))
        ]
    return None
