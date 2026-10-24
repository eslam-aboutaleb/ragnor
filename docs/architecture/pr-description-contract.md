# PR description contract

This guard uses the repository's existing `.github/pull_request_template.md`
as the source of truth. Human-authored PRs to `main` must fill all six
template sections, check exactly one change type, show a concrete test command
or GitHub Actions run and result, and disclose approval or an explicitly
pending independent review. All five checklist rows must be present, but
unchecked items are allowed while work is incomplete.

The check reads the read-only `pull_request` event JSON on GitHub Actions
and never interpolates PR text into a shell command. Automated bot PRs
are exempt to preserve Dependabot operation. It has no write permissions.

Local authoring:

```bash
python scripts/check_pr_description.py --body-file /path/to/pr-body.md
python -m pytest -q tests/test_pr_description_contract.py
```

This schema gate is **not** a substitute for tests or independent review.
It checks that evidence is described, not that the described feature is
functionally correct; full CI and approval rules must still run.
