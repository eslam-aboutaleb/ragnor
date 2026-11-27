"""Safe, deterministic provider fingerprints for snapshots and reindexing.

A fingerprint records identity, version and *non-sensitive* options. Prompts,
credentials, URLs with userinfo, long free text and raw bytes are dropped so
they cannot leak into snapshot metadata, manifests or logs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ragnor.embeddings._safe_config import safe_config

_BLOCKED = re.compile(
    r"prompt|instruction|system_?message|content|bytes|payload|document|body|header|cookie"
    r"|authoriz|credential|password|secret|api_?key|private",
    re.IGNORECASE,
)
_MAX_STRING = 256
_MAX_DEPTH = 6
_MAX_ITEMS = 64


def _scrub_string(value: str) -> str | None:
    if len(value) > _MAX_STRING:
        return None
    if "://" in value:
        parts = urlsplit(value)
        host = parts.hostname or ""
        return urlunsplit(
            (parts.scheme, host + (f":{parts.port}" if parts.port else ""), parts.path, "", "")
        )
    return value


def _clean(value: Any, depth: int) -> Any:  # noqa: PLR0911
    if depth > _MAX_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, Mapping):
        out = {}
        for key in sorted(k for k in value if isinstance(k, str)):
            if _BLOCKED.search(key):
                continue
            cleaned = _clean(value[key], depth + 1)
            if cleaned is not None or value[key] is None:
                out[key] = cleaned
            if len(out) >= _MAX_ITEMS:
                break
        return out
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in list(value)[:_MAX_ITEMS] if not isinstance(v, bytes)]
    return None


def safe_provider_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a bounded JSON copy of ``options`` that is safe to persist."""
    return _clean(safe_config(dict(options or {})), 0) or {}


def provider_fingerprint(provider: Any) -> dict[str, Any]:
    """Identity, version and safe options of a provider instance."""
    return {
        "id": str(getattr(provider, "id", type(provider).__name__)),
        "version": str(getattr(provider, "version", "0")),
        "options": safe_provider_options(getattr(provider, "options", {})),
    }


def fingerprint_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def extractor_fingerprint(extractor: Any) -> str:
    """Hash usable as ``sync.manifest`` ``extractor_fingerprint`` for any extractor."""
    return fingerprint_hash(
        {
            "id": str(getattr(extractor, "id", type(extractor).__name__)),
            "version": str(getattr(extractor, "version", "0")),
            "options": safe_provider_options(getattr(extractor, "options", {})),
        }
    )
