"""Per-page profiles and deterministic, cost-aware routing (zero dependencies).

A :class:`PageProfile` holds cheap, engine-free facts about one page. The
:func:`route_page` function maps a profile plus the capabilities the host has
registered onto a :class:`RouteDecision`. Routing is pure: same inputs, same
decision. It never calls a model, never touches the network, and never drops
a page: when no configured provider can read content, the decision is
``unroutable`` with the missing capability named, so the caller can fail or
queue the page rather than ingest nothing.

Cost is a fixed ordering (``text`` < ``layout`` < ``ocr`` < ``vlm``), and a
remote provider is only chosen when the policy allows it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ragnor.extraction.registry import Capability


class PageKind(StrEnum):
    DIGITAL = "digital"  # selectable text, nothing visual that text can miss
    SCAN = "scan"  # visual content, no text layer
    MIXED = "mixed"  # text layer plus images, drawings or annotations
    BLANK = "blank"  # no content operations at all
    UNKNOWN = "unknown"  # the profiler could not read the page


class Route(StrEnum):
    NATIVE = "native"  # text layer is enough
    LAYOUT = "layout"  # layout/table-aware engine
    OCR = "ocr"  # rasterise then OCR
    SKIP_BLANK = "skip_blank"  # verified blank, recorded not dropped
    UNROUTABLE = "unroutable"  # nothing configured can read it


@dataclass(frozen=True)
class PageProfile:
    page: int
    kind: PageKind
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0  # /Rotate in degrees, normalised to 0/90/180/270
    text_orientation: int = 0  # dominant on-page text direction, same units
    char_count: int = 0
    has_images: bool = False
    has_drawings: bool = False
    has_annotations: bool = False
    text_columns: int = 1  # x-clusters of text lines, a cheap layout hint
    reason: str = ""

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("Page numbers start at 1")
        for name in ("rotation", "text_orientation"):
            if getattr(self, name) not in (0, 90, 180, 270):
                raise ValueError(f"{name} must be 0, 90, 180 or 270")

    @property
    def visual_orientation(self) -> int:
        """Text direction as a viewer sees it. /Rotate turns the page clockwise."""
        return (self.text_orientation - self.rotation) % 360

    @property
    def sideways(self) -> bool:
        return self.visual_orientation != 0

    def to_dict(self) -> dict[str, Any]:
        out = {k: (v.value if isinstance(v, StrEnum) else v) for k, v in vars(self).items()}
        out["visual_orientation"] = self.visual_orientation
        return out


@dataclass(frozen=True)
class RoutePolicy:
    """Host policy. Defaults are the cheapest local behaviour."""

    allow_remote: bool = False
    prefer_layout_for_columns: bool = True
    min_columns_for_layout: int = 2


@dataclass(frozen=True)
class RouteDecision:
    page: int
    route: Route
    capability: Capability | None
    reason: str
    needs: tuple[Capability, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "route": self.route.value,
            "capability": self.capability.value if self.capability else None,
            "reason": self.reason,
            "needs": [c.value for c in self.needs],
            "notes": list(self.notes),
        }


#: Providers available to the router: capability -> is the cheapest one remote?
#: ``local`` lists capabilities with a local provider; ``remote`` those that
#: exist only as hosted services.
@dataclass(frozen=True)
class Available:
    local: frozenset[Capability] = frozenset()
    remote: frozenset[Capability] = frozenset()

    @classmethod
    def from_specs(cls, specs: Iterable[Any]) -> Available:
        local: set[Capability] = set()
        remote: set[Capability] = set()
        for spec in specs:
            (remote if spec.remote else local).update(spec.capabilities)
        return cls(frozenset(local), frozenset(remote - local))

    def usable(self, cap: Capability, policy: RoutePolicy) -> bool:
        return cap in self.local or (policy.allow_remote and cap in self.remote)


_OCR = (Route.OCR, Capability.OCR)
_LAYOUT = (Route.LAYOUT, Capability.LAYOUT)
_NATIVE = (Route.NATIVE, Capability.TEXT)


def _preferences(
    profile: PageProfile, policy: RoutePolicy
) -> tuple[tuple[tuple[Route, Capability], ...], str]:
    """Ordered (route, capability) choices, cheapest safe one first."""
    kind = profile.kind
    if kind is PageKind.SCAN:
        return (_OCR,), "no_text_layer"
    if kind is PageKind.UNKNOWN:
        # Unreadable by the profiler: most capable local path, never a skip.
        return (_OCR, _LAYOUT), "profile_failed"
    if kind is PageKind.MIXED:
        # Images on a text page may hold content the text layer lacks, so
        # native text is the last resort, not the default.
        return (_LAYOUT, _OCR, _NATIVE), "mixed_content"
    if policy.prefer_layout_for_columns and profile.text_columns >= policy.min_columns_for_layout:
        return (_LAYOUT, _NATIVE), "multi_column_text"
    return (_NATIVE, _LAYOUT, _OCR), "text_layer"


def route_page(
    profile: PageProfile, available: Available, policy: RoutePolicy | None = None
) -> RouteDecision:
    """Pick the cheapest route that cannot lose content on this page."""
    policy = policy or RoutePolicy()
    n = profile.page
    if profile.kind is PageKind.BLANK:
        return RouteDecision(n, Route.SKIP_BLANK, None, "verified_blank")
    notes: list[str] = []
    if profile.sideways:
        notes.append(f"text_orientation_{profile.visual_orientation}")
        if profile.visual_orientation in (90, 270):
            # Reading order is fine, but boxes need the rotated frame.
            notes.append("bbox_requires_rotation_transform")
    order, reason = _preferences(profile, policy)
    for index, (route, cap) in enumerate(order):
        if available.usable(cap, policy):
            if index:
                notes.append(f"preferred_provider_unavailable:{order[0][1].value}")
            return RouteDecision(n, route, cap, reason, notes=tuple(notes))
    if not policy.allow_remote and any(cap in available.remote for _, cap in order):
        notes.append("remote_provider_blocked_by_policy")
    return RouteDecision(
        n, Route.UNROUTABLE, None, reason, needs=(order[0][1],), notes=tuple(notes)
    )


def plan_document(
    profiles: Iterable[PageProfile],
    available: Available,
    policy: RoutePolicy | None = None,
) -> list[RouteDecision]:
    """One decision per input page, in page order; length always matches."""
    return [route_page(p, available, policy) for p in profiles]


def summarize(decisions: Iterable[RouteDecision]) -> Mapping[str, Any]:
    counts: dict[str, int] = {}
    unroutable: list[int] = []
    for d in decisions:
        counts[d.route.value] = counts.get(d.route.value, 0) + 1
        if d.route is Route.UNROUTABLE:
            unroutable.append(d.page)
    return {"routes": counts, "unroutable_pages": unroutable}
