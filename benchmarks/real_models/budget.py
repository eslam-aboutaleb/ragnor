"""Hard spend cap shared by every suite in one run."""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    """The next call could push spend past the cap; nothing was sent."""


@dataclass
class CostMeter:
    max_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    by_suite: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.max_usd > 0:
            raise ValueError("max_usd must be positive")

    def reserve(self, estimate_usd: float) -> None:
        """Refuse a call whose estimated cost would pass the cap."""
        if self.spent_usd + max(0.0, estimate_usd) > self.max_usd:
            raise BudgetExceededError(
                f"cost cap {self.max_usd:.2f} USD reached after {self.calls} calls "
                f"({self.spent_usd:.4f} USD spent)"
            )

    def charge(self, suite: str, usd: float) -> None:
        self.calls += 1
        self.spent_usd += max(0.0, usd)
        self.by_suite[suite] = self.by_suite.get(suite, 0.0) + max(0.0, usd)

    def to_dict(self) -> dict[str, object]:
        return {
            "cap_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "by_suite_usd": {k: round(v, 6) for k, v in self.by_suite.items()},
        }
