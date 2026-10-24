"""Regression tests for PR template compliance and action-safe validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "check_pr_description.py"
spec = importlib.util.spec_from_file_location("check_pr_description", MODULE)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


VALID = """## Description
Replace the token counter via the injected port. Fixes #27.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] CI/CD or tooling enhancement

## Behavior and Compatibility
Legacy chunkers retain their previous source ordering.

## Verification Evidence
`pytest -q tests/test_token_aware_chunker.py`: passed, includes Unicode and strict failure.

## Independent Review
Pending: independent review is required for storage concurrency.

## Checklist
- [x] My code adheres to the project's coding style guidelines
- [x] Changed runtime contracts and compatibility requirements are documented above
- [x] Behavioral tests exercise the changed component boundaries
- [x] Verification commands, results, and skipped checks are recorded above
- [x] I have updated documentation where appropriate
"""


def test_complete_schema_is_accepted() -> None:
    assert checker.validate(VALID) == []


def test_missing_sections_and_placeholder_only_content_fail() -> None:
    errors = checker.validate("## Description\n<!-- fill me -->\n")
    assert any("Type of Change" in error for error in errors)
    assert any("Description" in error for error in errors)


def test_verification_must_name_a_specific_test_and_result_or_gap() -> None:
    invalid = VALID.replace(
        "`pytest -q tests/test_token_aware_chunker.py`: passed, includes Unicode and strict failure.",
        "CI checked.",
    )
    assert any("Verification Evidence" in error for error in checker.validate(invalid))


def test_type_requires_exactly_one_checked_option() -> None:
    invalid = VALID.replace(
        "- [ ] CI/CD or tooling enhancement", "- [x] CI/CD or tooling enhancement"
    )
    assert any("Type of Change" in error for error in checker.validate(invalid))


def test_pending_independent_review_is_explicitly_accepted() -> None:
    assert (
        checker.validate(
            VALID.replace(
                "Pending: independent review is required for storage concurrency.",
                "N/A: docs-only change.",
            )
        )
        == []
    )


def test_security_review_cannot_be_invisible() -> None:
    invalid = VALID.replace(
        "Pending: independent review is required for storage concurrency.", "TBD"
    )
    assert any("Independent Review" in error for error in checker.validate(invalid))


def test_missing_checklist_is_not_accepted() -> None:
    assert any(
        "Checklist" in error
        for error in checker.validate(
            VALID.replace("- [x] I have updated documentation where appropriate", "")
        )
    )


def test_cli_validates_actual_human_event_and_rejects_empty_body(tmp_path) -> None:
    import json
    import subprocess
    import sys

    event = tmp_path / "event.json"
    for body, expected in ((VALID, 0), ("", 1)):
        event.write_text(
            json.dumps({"pull_request": {"user": {"login": "maintainer"}, "body": body}})
        )
        result = subprocess.run(  # noqa: S603 - fixed interpreter and local checker
            [sys.executable, str(MODULE), "--event", str(event)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected, result.stdout


def test_cli_exempts_automation_event(tmp_path) -> None:
    import json
    import subprocess
    import sys

    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"pull_request": {"user": {"login": "dependabot[bot]"}, "body": ""}})
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and local checker
        [sys.executable, str(MODULE), "--event", str(event)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Automation-authored" in result.stdout
