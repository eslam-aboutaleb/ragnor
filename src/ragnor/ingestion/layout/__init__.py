"""Opt-in, confidence-triggered page layout inventory and reconciliation (EX-07)."""

from ragnor.ingestion.layout.inventory import (
    BudgetLedger,
    InventoryBudget,
    InventoryError,
    InventoryLimits,
    InventoryRegion,
    InventoryTrace,
    PageInventory,
    PageInventoryProvider,
)
from ragnor.ingestion.layout.planner import (
    LayoutInventoryPlanner,
    LayoutInventoryPolicy,
    PageImageSource,
    PageLayoutResult,
)
from ragnor.ingestion.layout.reconcile import (
    LayoutPlan,
    PlanInvariantError,
    PlanRegion,
    ReconcilePolicy,
    apply_plan,
    baseline_plan,
    reconcile,
)
from ragnor.ingestion.layout.regions import (
    LayoutRegion,
    PageImage,
    ProposedLayout,
    proposed_from_outcome,
    stable_region_id,
)
from ragnor.ingestion.layout.suspicion import Signal, SuspicionPolicy, SuspicionReport, assess

__all__ = [
    "BudgetLedger",
    "InventoryBudget",
    "InventoryError",
    "InventoryLimits",
    "InventoryRegion",
    "InventoryTrace",
    "LayoutInventoryPlanner",
    "LayoutInventoryPolicy",
    "LayoutPlan",
    "LayoutRegion",
    "PageImage",
    "PageImageSource",
    "PageInventory",
    "PageInventoryProvider",
    "PageLayoutResult",
    "PlanInvariantError",
    "PlanRegion",
    "ProposedLayout",
    "ReconcilePolicy",
    "Signal",
    "SuspicionPolicy",
    "SuspicionReport",
    "apply_plan",
    "assess",
    "baseline_plan",
    "proposed_from_outcome",
    "reconcile",
    "stable_region_id",
]
