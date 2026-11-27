"""Cost-aware, opt-in page inventory planning (EX-07).

Clean pages never reach the provider. Remote providers need an explicit
cloud-upload opt-in. Every provider failure ends in a typed, content-free
fallback: the deterministic baseline layout, or ``needs_review``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Protocol

from ragnor.ingestion.layout.inventory import (
    BudgetLedger,
    InventoryBudget,
    InventoryError,
    InventoryLimits,
    PageInventory,
    PageInventoryProvider,
)
from ragnor.ingestion.layout.reconcile import (
    LayoutPlan,
    ReconcilePolicy,
    baseline_plan,
    reconcile,
)
from ragnor.ingestion.layout.regions import PageImage, ProposedLayout
from ragnor.ingestion.layout.suspicion import SuspicionPolicy, SuspicionReport, assess

logger = logging.getLogger("ragnor.ingestion.layout")


class PageImageSource(Protocol):
    def render(self, page: int, *, max_dpi: int, max_pixels: int) -> PageImage:
        raise NotImplementedError


@dataclass(frozen=True)
class LayoutInventoryPolicy:
    """Everything defaults to off: no provider call without ``enabled=True``."""

    enabled: bool = False
    allow_remote_upload: bool = False
    on_failure: str = "baseline"  # "baseline" keeps the parser layout, else "needs_review"
    max_concurrency: int = 2
    suspicion: SuspicionPolicy = SuspicionPolicy()
    reconcile: ReconcilePolicy = ReconcilePolicy()
    budget: InventoryBudget = InventoryBudget()
    limits: InventoryLimits = InventoryLimits()

    def __post_init__(self) -> None:
        if self.on_failure not in {"baseline", "needs_review"}:
            raise ValueError("on_failure must be 'baseline' or 'needs_review'")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")


@dataclass(frozen=True)
class PageLayoutResult:
    plan: LayoutPlan
    suspicion: SuspicionReport
    provider_calls: int
    failure_code: str = ""
    skipped_reason: str = ""


class LayoutInventoryPlanner:
    def __init__(
        self,
        provider: PageInventoryProvider | None,
        images: PageImageSource | None,
        policy: LayoutInventoryPolicy = LayoutInventoryPolicy(),
    ) -> None:
        self._provider = provider
        self._images = images
        self.policy = policy

    def new_ledger(self) -> BudgetLedger:
        return BudgetLedger(self.policy.budget)

    def plan_page(
        self,
        layout: ProposedLayout,
        *,
        ledger: BudgetLedger | None = None,
        cancel: threading.Event | None = None,
    ) -> PageLayoutResult:
        policy = self.policy
        ledger = ledger or self.new_ledger()
        report = assess(layout, policy.suspicion)
        codes = report.codes
        skip = ""
        if not policy.enabled or self._provider is None or self._images is None:
            skip = "inventory_disabled"
        elif not report.suspicious:
            skip = "clean_page"
        if skip:
            plan = baseline_plan(
                layout,
                status="unchanged",
                reason=skip,
                suspicion_codes=codes,
                crop_margin=policy.reconcile.crop_margin,
            )
            return PageLayoutResult(plan, report, 0, skipped_reason=skip)
        calls_before = ledger.calls
        try:
            inventory = self._fetch(layout, ledger, cancel)
        except InventoryError as exc:
            logger.warning("layout inventory failed page=%s code=%s", layout.page, exc.code)
            plan = baseline_plan(
                layout,
                status="baseline" if policy.on_failure == "baseline" else "needs_review",
                reason="inventory_failed",
                suspicion_codes=codes,
                failure_code=exc.code,
                crop_margin=policy.reconcile.crop_margin,
            )
            return PageLayoutResult(plan, report, ledger.calls - calls_before, exc.code)
        plan = reconcile(layout, inventory, policy=policy.reconcile, suspicion_codes=codes)
        logger.info(
            "layout inventory page=%s status=%s reasons=%s",
            layout.page,
            plan.status,
            ",".join(codes),
        )
        return PageLayoutResult(plan, report, ledger.calls - calls_before)

    def plan_pages(
        self,
        layouts: Sequence[ProposedLayout],
        *,
        ledger: BudgetLedger | None = None,
        cancel: threading.Event | None = None,
    ) -> list[PageLayoutResult]:
        """Plan pages with bounded concurrency; results keep input order."""
        ledger = ledger or self.new_ledger()
        with ThreadPoolExecutor(max_workers=self.policy.max_concurrency) as pool:
            futures = [
                pool.submit(self.plan_page, layout, ledger=ledger, cancel=cancel)
                for layout in layouts
            ]
            return [f.result() for f in futures]

    def _fetch(
        self, layout: ProposedLayout, ledger: BudgetLedger, cancel: threading.Event | None
    ) -> PageInventory:
        provider, images, policy = self._provider, self._images, self.policy
        assert provider is not None and images is not None  # noqa: S101 - narrowed by caller
        if provider.remote and not policy.allow_remote_upload:
            raise InventoryError("external_upload_denied")
        _check_cancel(cancel)
        budget = policy.budget
        image = images.render(layout.page, max_dpi=budget.max_dpi, max_pixels=budget.max_pixels)
        if image.dpi > budget.max_dpi or image.pixels > budget.max_pixels:
            raise InventoryError("image_limit")
        last = "provider_failed"
        for _attempt in range(budget.max_retries + 1):
            _check_cancel(cancel)
            if not ledger.reserve(provider.cost_per_call):
                raise InventoryError("budget_exceeded")
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(provider.inventory, image, layout, budget)
            try:
                raw = future.result(timeout=budget.timeout_seconds)
            except FutureTimeout:
                last = "timeout"
                continue
            except Exception:  # noqa: BLE001 - provider errors are never surfaced verbatim
                last = "provider_failed"
                continue
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            return PageInventory.from_json(
                raw,
                limits=policy.limits,
                expected_fingerprint=layout.page_fingerprint or None,
                image=image,
                rotation=image.rotation,
            )
        raise InventoryError(last)


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise InventoryError("cancelled")
