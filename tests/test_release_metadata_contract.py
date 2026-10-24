"""Release metadata must agree with the actual license and install extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_and_distribution_license_claim_agree() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    license_id = data["project"]["license"]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    licensing_doc = (ROOT / "docs" / "licensing.md").read_text(encoding="utf-8")
    assert license_id == "Elastic-2.0"
    assert "Elastic License 2.0" in license_text
    assert "Elastic License 2.0" in readme.split("## License", maxsplit=1)[-1]
    assert "Elastic-2.0" in licensing_doc
    assert "released under the MIT License" not in readme


def test_zero_dependency_and_reserved_extra_contract() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["dependencies"] == []
    extras = project["optional-dependencies"]
    assert extras["local-embeddings"] == []
    assert "tokenizers-tiktoken" in extras
    assert all(isinstance(name, str) for name in extras)
    assert "ragnor" in data["tool"]["setuptools"]["package-data"]
    assert "py.typed" in data["tool"]["setuptools"]["package-data"]["ragnor"]
