"""Source and distribution licensing must describe the actual shipped terms."""

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_active_license_is_elastic_and_not_an_older_license() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Elastic-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    license_text = (ROOT / "LICENSE").read_text()
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == (
        "48255018b41fc0e965b1115af7e6779bc218bb8a6747d561da800d5022622aa2"
    )
    assert license_text.startswith("Elastic License 2.0\n")
    assert "## Limitations" in license_text
    assert "## Termination" in license_text
    assert "## No Liability" in license_text
    assert "Copyright (c) 2026 Eslam Aboutaleb" in (ROOT / "NOTICE").read_text()
    current_readme = (ROOT / "README.md").read_text().split("## License", 1)[1]
    assert "Elastic License 2.0" in current_readme
    assert "Elastic-2.0" in current_readme
    assert "PolyForm" not in current_readme
    assert "MIT License" not in current_readme
    assert "Elastic-2.0" in (ROOT / "docs/licensing.md").read_text()
