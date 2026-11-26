"""Page verification pipeline: cheap checks everywhere, render-back only where needed."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ragnor.ingestion.extraction_types import (
    BBox,
    ExtractionTable,
    PageExtractionOutcome,
)
from ragnor.ingestion.verification.checks import (
    ArithmeticRelation,
    Check,
    CheckUnavailableError,
    ExpectedPage,
    VerificationContext,
    default_checks,
)
from ragnor.ingestion.verification.report import Finding, VerificationReport, VerifierSettings

logger = logging.getLogger("ragnor.ingestion.verification")


@dataclass(frozen=True)
class RenderBackResult:
    similarity: float
    artifact_ref: str


class RenderBackVerifier(Protocol):
    """Expensive optional image comparison, run only for targeted failing regions."""

    def compare(self, page: int, region_id: str, bbox: BBox | None) -> RenderBackResult:
        raise NotImplementedError


class PageVerificationPipeline:
    def __init__(
        self,
        settings: VerifierSettings = VerifierSettings(),
        checks: Sequence[Check] | None = None,
        *,
        render_back: RenderBackVerifier | None = None,
        render_back_min_similarity: float = 0.9,
        render_back_limit: int = 5,
    ) -> None:
        self.settings = settings
        self.checks = tuple(default_checks() if checks is None else checks)
        ids = [c.check_id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate check IDs")
        self._render_back = render_back
        self._min_similarity = render_back_min_similarity
        self._limit = render_back_limit

    def verify(
        self,
        outcome: PageExtractionOutcome,
        *,
        expected: ExpectedPage | None = None,
        alternate_tables: tuple[ExtractionTable, ...] | None = None,
        relations: tuple[ArithmeticRelation, ...] = (),
    ) -> VerificationReport:
        ctx = VerificationContext(outcome, expected, alternate_tables, relations, self.settings)
        page = outcome.page
        findings: list[Finding] = []
        ran: list[str] = []
        for check in self.checks:
            try:
                result = check.run(ctx)
            except CheckUnavailableError:
                if check.check_id in self.settings.mandatory:
                    findings.append(
                        Finding(
                            page,
                            f"page-{page}",
                            check.check_id,
                            "error",
                            "required input unavailable",
                            "check executed",
                            "pipeline",
                            "needs_review",
                        )
                    )
                continue
            except Exception:  # noqa: BLE001 - a crashing check must never pass silently
                logger.warning("verification check failed page=%s check=%s", page, check.check_id)
                findings.append(
                    Finding(
                        page,
                        f"page-{page}",
                        check.check_id,
                        "error",
                        "check raised",
                        "check executed",
                        "pipeline",
                        "needs_review",
                    )
                )
                continue
            ran.append(check.check_id)
            findings.extend(result)
        checked = self._render_back_findings(page, findings)
        findings.extend(checked[0])
        return VerificationReport(
            page,
            self._status(findings),
            tuple(findings),
            tuple(ran),
            self.settings.version,
            self.settings.fingerprint,
            outcome.status.value,
            checked[1],
        )

    def _render_back_findings(
        self, page: int, findings: list[Finding]
    ) -> tuple[list[Finding], tuple[str, ...]]:
        if self._render_back is None:
            return [], ()
        targets = list(
            dict.fromkeys(
                f.region_id
                for f in findings
                if f.severity == "error" and f.disposition == "repair_candidate"
            )
        )[: self._limit]
        extra: list[Finding] = []
        for region_id in targets:
            try:
                result = self._render_back.compare(page, region_id, None)
                ok = result.similarity >= self._min_similarity
                extra.append(
                    Finding(
                        page,
                        region_id,
                        "render_back",
                        "info" if ok else "error",
                        f"similarity {result.similarity:.2f}",
                        f"at least {self._min_similarity:.2f}",
                        result.artifact_ref,
                        "none" if ok else "needs_review",
                    )
                )
            except Exception:  # noqa: BLE001
                logger.warning("render-back failed page=%s", page)
                extra.append(
                    Finding(
                        page,
                        region_id,
                        "render_back",
                        "error",
                        "render-back raised",
                        "comparison",
                        "pipeline",
                        "needs_review",
                    )
                )
        return extra, tuple(targets)

    def _status(self, findings: Sequence[Finding]) -> str:
        errors = [f for f in findings if f.severity == "error"]
        if any(f.disposition == "needs_review" for f in errors):
            return "needs_review"
        if errors:
            return "failed"
        if self.settings.fail_on_warning and any(f.severity == "warning" for f in findings):
            return "needs_review"
        return "verified"


class PipelinePageVerifier:
    """``PageVerifier`` adapter so the existing extractor gate runs the pipeline.

    Pages are accepted only when the report is ``verified``. A crash in the
    pipeline propagates, which the extractor turns into ``verification_failed``
    rather than a silent accept.
    """

    id = "page-verification-pipeline"

    def __init__(
        self,
        pipeline: PageVerificationPipeline,
        context_for: Callable[[PageExtractionOutcome], Mapping[str, Any]] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._context_for = context_for
        self._lock = threading.Lock()
        self._reports: dict[int, VerificationReport] = {}

    @property
    def version(self) -> str:
        return self._pipeline.settings.version

    @property
    def options(self) -> dict[str, Any]:
        return {
            "settings_sha256": self._pipeline.settings.fingerprint,
            "checks": [c.check_id for c in self._pipeline.checks],
        }

    def verify(self, page: PageExtractionOutcome) -> bool:
        context = dict(self._context_for(page)) if self._context_for else {}
        report = self._pipeline.verify(page, **context)
        with self._lock:
            self._reports[page.page] = report
        return report.status == "verified"

    def report(self, page: int) -> VerificationReport | None:
        with self._lock:
            return self._reports.get(page)
