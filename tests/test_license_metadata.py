"""Source and distribution licensing must describe the actual shipped terms."""

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_license_is_mit_and_not_an_older_license() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    license_text = (ROOT / "LICENSE").read_text()
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == (
        "c28facedc5bfb2d238a3dac6375f057bb5f76eed029978babe1a810fffe8068c"
    )
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Eslam Aboutaleb" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text
    assert "Copyright (c) 2026 Eslam Aboutaleb" in (ROOT / "NOTICE").read_text()
    current_readme = (ROOT / "README.md").read_text().split("## License", 1)[1]
    assert "MIT License" in current_readme
    assert "`MIT`" in current_readme
    assert "PolyForm" not in current_readme
    assert "Elastic" not in current_readme
    licensing = (ROOT / "docs/licensing.md").read_text()
    assert "SPDX `MIT`" in licensing


def test_no_active_file_still_declares_elastic() -> None:
    reference = tomllib.loads((ROOT / "external-adapters/reference/pyproject.toml").read_text())
    assert reference["project"]["license"] == "MIT"
    assert "Elastic-2.0" not in (ROOT / "scripts/check_distribution_license.py").read_text()
