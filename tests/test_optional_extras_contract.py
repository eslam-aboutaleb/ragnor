"""Optional dependency metadata must not advertise non-functional integrations."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_recursive_all_extra_includes_each_supported_adapter_and_no_reserved_extra() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    declared = set(extras)
    assert extras["local-embeddings"] == [], "reserved extra must remain explicitly empty"
    assert len(extras["all"]) == 1
    match = re.fullmatch(r"ragnor\[([\w,-]+)\]", extras["all"][0])
    assert match is not None
    actual = set(match.group(1).split(","))
    supported = declared - {"all", "test", "local-embeddings"}
    assert actual == supported
    assert all(extras[name] for name in supported)


def test_extra_coverage_matrix_is_documented() -> None:
    doc = (ROOT / "docs" / "sdk" / "optional-extras.md").read_text(encoding="utf-8")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    for name in extras:
        assert f"`{name}`" in doc, f"undocumented optional extra: {name}"
    assert "reserved" in doc.lower()
