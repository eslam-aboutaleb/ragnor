"""Atomic, verification-gated publication of an extraction revision."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ragnor.ingestion.verification.report import VerificationReport


@dataclass(frozen=True)
class Revision:
    revision_id: str
    pages: tuple[tuple[int, str], ...]  # (page, verification status)


class RevisionStore(Protocol):
    def active(self, document_id: str) -> Revision | None:
        raise NotImplementedError

    def commit(self, document_id: str, revision: Revision) -> None:
        """Atomically replace the active revision, or raise leaving it unchanged."""
        raise NotImplementedError


class InMemoryRevisionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, Revision] = {}

    def active(self, document_id: str) -> Revision | None:
        with self._lock:
            return self._active.get(document_id)

    def commit(self, document_id: str, revision: Revision) -> None:
        with self._lock:
            self._active[document_id] = revision


@dataclass(frozen=True)
class PublishPolicy:
    strict: bool = True
    required_checks: frozenset[str] = frozenset({"completeness"})


@dataclass(frozen=True)
class PublishDecision:
    published: bool
    reason: str
    revision_id: str
    blocked_pages: tuple[int, ...] = ()


def publish_verified(  # noqa: PLR0913
    store: RevisionStore,
    document_id: str,
    revision_id: str,
    reports: Sequence[VerificationReport],
    *,
    expected_pages: int,
    policy: PublishPolicy = PublishPolicy(),
) -> PublishDecision:
    """Publish a whole revision or nothing. Strict mode never replaces the active
    revision unless every page verified and every required check actually ran."""
    by_page = {r.page: r for r in reports}
    blocked: list[int] = []
    reason = ""
    if len(by_page) != len(reports) or set(by_page) != set(range(1, expected_pages + 1)):
        return PublishDecision(False, "incomplete_reports", revision_id)
    if len({r.settings_fingerprint for r in reports}) > 1:
        return PublishDecision(False, "mixed_verifier_settings", revision_id)
    for page in sorted(by_page):
        report = by_page[page]
        if not policy.required_checks <= set(report.checks_run):
            blocked.append(page)
            reason = "required_check_skipped"
        elif report.status == "failed" or (policy.strict and report.status != "verified"):
            blocked.append(page)
            reason = reason or f"page_{report.status}"
    if blocked:
        return PublishDecision(False, reason, revision_id, tuple(blocked))
    revision = Revision(revision_id, tuple((p, by_page[p].status) for p in sorted(by_page)))
    try:
        store.commit(document_id, revision)
    except Exception:  # noqa: BLE001 - previous revision stays active
        return PublishDecision(False, "commit_failed", revision_id)
    return PublishDecision(True, "published", revision_id)
