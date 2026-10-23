"""Validate Ragnor's pull-request contract from local Markdown or GitHub event JSON.

Exits nonzero for missing sections, untracked verification, empty independent
review or malformed checklists. Never executes untrusted PR body content.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SECTIONS = (
    "Description",
    "Type of Change",
    "Behavior and Compatibility",
    "Verification Evidence",
    "Independent Review",
    "Checklist",
)
TYPES = (
    "Bug fix (non-breaking change which fixes an issue)",
    "New feature (non-breaking change which adds functionality)",
    (
        "Breaking change (fix or feature that would cause existing "
        "functionality to not work as expected)"
    ),
    "Documentation update",
    "Performance improvement",
    "CI/CD or tooling enhancement",
)
CHECKLIST = (
    "My code adheres to the project's coding style guidelines",
    "Changed runtime contracts and compatibility requirements are documented above",
    "Behavioral tests exercise the changed component boundaries",
    "Verification commands, results, and skipped checks are recorded above",
    "I have updated documentation where appropriate",
)
HEADER = re.compile(r"^## ([^\n]+)\s*$", re.MULTILINE)
ITEM = re.compile(r"^\s*-\s*\[([ xX])\]\s*(.*?)\s*$", re.MULTILINE)
EVIDENCE = re.compile(
    r"(tests?/[\w./-]+|\bpytest\b|\bruff\b|\bmypy\b|actions/runs/\d+|"
    r"\bmake test\b|\bnpm test\b)",
    re.IGNORECASE,
)
OUTCOME = re.compile(
    r"\b(pass(?:ed)?|fail(?:ed)?|skipp?ed|pending|not run|unverified|"
    r"not applicable)\b|\bN/A\b",
    re.IGNORECASE,
)


def _substantial(value: str) -> bool:
    stripped = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    return len(stripped.strip()) >= 12 and stripped.strip().upper() not in {"TBD", "TODO", "N/A"}


def validate(body: str) -> list[str]:
    """Return human-readable contract violations, never mutate the PR."""
    errors: list[str] = []
    headers = list(HEADER.finditer(body))
    pieces: dict[str, str] = {}
    for idx, match in enumerate(headers):
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(body)
        pieces[match.group(1).strip()] = body[match.end() : end].strip()

    for section in SECTIONS:
        if section not in pieces:
            errors.append(f"Missing section: ## {section}")
        elif not _substantial(pieces[section]):
            errors.append(f"{section} must contain an explicit non-placeholder explanation")

    if "Type of Change" in pieces:
        checked = [
            label for state, label in ITEM.findall(pieces["Type of Change"])
            if state.lower() == "x" and label in TYPES
        ]
        if len(checked) != 1:
            errors.append("Type of Change must select exactly one official template option")
        for value in TYPES:
            if not any(label == value for _, label in ITEM.findall(pieces["Type of Change"])):
                errors.append(f"Type of Change missing template option: {value}")

    verification = pieces.get("Verification Evidence", "")
    if verification and (
        not EVIDENCE.search(verification) or not OUTCOME.search(verification)
    ):
        errors.append(
            "Verification Evidence must name a concrete test or CI run and "
            "give its result or an explicit pending/unverified status"
        )

    independent = pieces.get("Independent Review", "")
    if independent and not _substantial(independent):
        errors.append("Independent Review must name approval, pending review or justified N/A")
    elif independent and independent.strip().upper() in {"PENDING", "NONE"}:
        errors.append("Independent Review must explain who/what is pending or why N/A applies")

    items = [text for _, text in ITEM.findall(pieces.get("Checklist", ""))]
    for item in CHECKLIST:
        if not any(candidate.startswith(item) for candidate in items):
            errors.append(f"Checklist missing template item: {item}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=Path, help="GitHub pull_request event payload")
    source.add_argument("--body-file", type=Path, help="Local markdown body")
    args = parser.parse_args()

    if args.event:
        event = json.loads(args.event.read_text(encoding="utf-8"))
        pull_request = event.get("pull_request") or {}
        user = (pull_request.get("user") or {}).get("login", "")
        if user.endswith("[bot]"):
            print("Automation-authored PR: template check is not required.")
            return 0
        body = pull_request.get("body") or ""
    else:
        body = args.body_file.read_text(encoding="utf-8")

    errors = validate(body)
    for error in errors:
        print(f"::error::{error}")
    if errors:
        return 1
    print("PR description meets the repository template contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
