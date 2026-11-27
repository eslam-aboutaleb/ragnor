"""Strict ``PageInventory`` schema, provider port and budget ledger (EX-07).

The inventory states only *where* objects are and what type they are. It never
carries table or cell values, so a model cannot introduce content through it.
Parsing is strict: any violation raises :class:`InventoryError` with a fixed,
content-free code.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ragnor.ingestion.extraction_types import BBox, checked_bbox
from ragnor.ingestion.layout.regions import (
    REGION_TYPES,
    LayoutRegion,
    PageImage,
    ProposedLayout,
    iou,
    to_canonical,
)

SCHEMA_VERSION = 1
_TOP_KEYS = frozenset(
    {"schema_version", "page_fingerprint", "image", "regions", "counts", "uncertainty", "trace"}
)
_REGION_KEYS = frozenset({"id", "type", "bbox", "reading_order", "caption_target"})


class InventoryError(RuntimeError):
    """Fixed-code failure; messages never contain provider output or page content."""

    codes = frozenset(
        {
            "invalid_schema",
            "output_limit",
            "object_limit",
            "fingerprint_mismatch",
            "image_mismatch",
            "conflicting_regions",
            "external_upload_denied",
            "budget_exceeded",
            "timeout",
            "cancelled",
            "provider_failed",
            "image_limit",
        }
    )

    def __init__(self, code: str):
        self.code = code if code in self.codes else "provider_failed"
        super().__init__(f"Page inventory failed: {self.code}")


@dataclass(frozen=True)
class InventoryLimits:
    max_objects: int = 200
    max_output_bytes: int = 256_000

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.max_objects, self.max_output_bytes)):
            raise ValueError("Inventory limits must be positive integers")


@dataclass(frozen=True)
class InventoryRegion:
    region_id: str
    kind: str
    bbox: BBox
    reading_order: int | None = None
    caption_target: str | None = None


@dataclass(frozen=True)
class InventoryTrace:
    provider: str
    model: str = ""
    request_id: str = ""


@dataclass(frozen=True)
class PageInventory:
    page_fingerprint: str
    image_width: int
    image_height: int
    regions: tuple[InventoryRegion, ...]
    uncertainty: float
    trace: InventoryTrace

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for region in self.regions:
            result[region.kind] = result.get(region.kind, 0) + 1
        return dict(sorted(result.items()))

    @classmethod
    def from_json(
        cls,
        raw: str | bytes | Mapping[str, Any],
        *,
        limits: InventoryLimits = InventoryLimits(),
        expected_fingerprint: str | None = None,
        image: PageImage | None = None,
        rotation: int = 0,
    ) -> PageInventory:
        """Validate untrusted provider output. Boxes are mapped to the upright frame."""
        if isinstance(raw, str | bytes):
            if len(raw) > limits.max_output_bytes:
                raise InventoryError("output_limit")
            try:
                data = json.loads(raw)
            except (ValueError, RecursionError):
                raise InventoryError("invalid_schema") from None
        else:
            data = dict(raw)
        if not isinstance(data, dict) or set(data) - _TOP_KEYS or data.get("schema_version") != 1:
            raise InventoryError("invalid_schema")
        regions_raw = data.get("regions")
        if not isinstance(regions_raw, list):
            raise InventoryError("invalid_schema")
        if len(regions_raw) > limits.max_objects:
            raise InventoryError("object_limit")
        fingerprint = data.get("page_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise InventoryError("invalid_schema")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise InventoryError("fingerprint_mismatch")
        img = data.get("image")
        if (
            not isinstance(img, dict)
            or set(img) != {"width", "height"}
            or any(type(img[k]) is not int or img[k] < 1 for k in img)
        ):
            raise InventoryError("invalid_schema")
        if image is not None and (img["width"], img["height"]) != (image.width, image.height):
            raise InventoryError("image_mismatch")
        uncertainty = data.get("uncertainty")
        if (
            isinstance(uncertainty, bool)
            or not isinstance(uncertainty, int | float)
            or not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise InventoryError("invalid_schema")
        trace_raw = data.get("trace", {})
        if not isinstance(trace_raw, dict) or any(
            not isinstance(v, str) or len(v) > 200 for v in trace_raw.values()
        ):
            raise InventoryError("invalid_schema")
        regions = tuple(_parse_region(item, rotation) for item in regions_raw)
        ids = [r.region_id for r in regions]
        if len(set(ids)) != len(ids):
            raise InventoryError("conflicting_regions")
        known = set(ids)
        if any(r.caption_target is not None and r.caption_target not in known for r in regions):
            raise InventoryError("invalid_schema")
        orders = [r.reading_order for r in regions if r.reading_order is not None]
        if len(set(orders)) != len(orders):
            raise InventoryError("conflicting_regions")
        for i, first in enumerate(regions):
            for second in regions[i + 1 :]:
                if first.kind == second.kind and iou(first.bbox, second.bbox) > 0.5:
                    raise InventoryError("conflicting_regions")
        inventory = cls(
            fingerprint,
            img["width"],
            img["height"],
            regions,
            float(uncertainty),
            InventoryTrace(
                str(trace_raw.get("provider", "")),
                str(trace_raw.get("model", "")),
                str(trace_raw.get("request_id", "")),
            ),
        )
        declared = data.get("counts")
        if declared is not None and declared != inventory.counts():
            raise InventoryError("invalid_schema")
        return inventory

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "page_fingerprint": self.page_fingerprint,
                "image": {"width": self.image_width, "height": self.image_height},
                "regions": [
                    {
                        "id": r.region_id,
                        "type": r.kind,
                        "bbox": list(r.bbox),
                        "reading_order": r.reading_order,
                        "caption_target": r.caption_target,
                    }
                    for r in self.regions
                ],
                "counts": self.counts(),
                "uncertainty": self.uncertainty,
                "trace": {
                    "provider": self.trace.provider,
                    "model": self.trace.model,
                    "request_id": self.trace.request_id,
                },
            },
            sort_keys=True,
        )


def _parse_region(item: object, rotation: int) -> InventoryRegion:
    if (
        not isinstance(item, dict)
        or set(item) - _REGION_KEYS
        or not {"id", "type", "bbox"} <= set(item)
    ):
        raise InventoryError("invalid_schema")
    region_id, kind, bbox = item["id"], item["type"], item["bbox"]
    if not isinstance(region_id, str) or not region_id or len(region_id) > 100:
        raise InventoryError("invalid_schema")
    if kind not in REGION_TYPES:
        raise InventoryError("invalid_schema")
    if (
        not isinstance(bbox, list | tuple)
        or len(bbox) != 4
        or any(isinstance(v, bool) or not isinstance(v, int | float) for v in bbox)
    ):
        raise InventoryError("invalid_schema")
    order = item.get("reading_order")
    if order is not None and (type(order) is not int or order < 0):
        raise InventoryError("invalid_schema")
    target = item.get("caption_target")
    if target is not None and not isinstance(target, str):
        raise InventoryError("invalid_schema")
    try:
        checked_bbox(tuple(float(v) for v in bbox))  # type: ignore[arg-type]
        canonical = to_canonical(tuple(float(v) for v in bbox), rotation)  # type: ignore[arg-type]
    except ValueError:
        raise InventoryError("invalid_schema") from None
    return InventoryRegion(region_id, kind, canonical, order, target)


@dataclass(frozen=True)
class InventoryBudget:
    """Per-document limits. ``max_cost`` is in provider-defined cost units."""

    max_calls: int = 5
    max_cost: float = 1.0
    timeout_seconds: float = 30.0
    max_retries: int = 1
    max_dpi: int = 150
    max_pixels: int = 8_000_000

    def __post_init__(self) -> None:
        if any(
            type(v) is not int or v < lo
            for v, lo in (
                (self.max_calls, 0),
                (self.max_retries, 0),
                (self.max_dpi, 36),
                (self.max_pixels, 1),
            )
        ):
            raise ValueError("Invalid inventory budget")
        if not math.isfinite(self.max_cost) or self.max_cost < 0:
            raise ValueError("Inventory cost budget must be finite and non-negative")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Inventory timeout must be positive and finite")


class BudgetLedger:
    """Thread-safe call and cost accounting shared by all pages of one document."""

    def __init__(self, budget: InventoryBudget) -> None:
        self.budget = budget
        self._lock = threading.Lock()
        self.calls = 0
        self.cost = 0.0
        self.refusals = 0

    def reserve(self, cost: float) -> bool:
        with self._lock:
            if self.calls + 1 > self.budget.max_calls or self.cost + cost > self.budget.max_cost:
                self.refusals += 1
                return False
            self.calls += 1
            self.cost += cost
            return True


class PageInventoryProvider(Protocol):
    """Host-injected VLM port. ``remote`` providers need explicit cloud opt-in.

    ``inventory`` receives the rendered page and the proposed layout (geometry
    only) and returns raw JSON (or a mapping) matching :class:`PageInventory`.
    It must not be asked for, and the schema cannot carry, table or cell values.
    """

    id: str
    version: str
    remote: bool
    cost_per_call: float

    def inventory(
        self, page_image: PageImage, proposed_layout: ProposedLayout, budget: InventoryBudget
    ) -> str | bytes | Mapping[str, Any]:
        raise NotImplementedError


__all__ = [
    "BudgetLedger",
    "InventoryBudget",
    "InventoryError",
    "InventoryLimits",
    "InventoryRegion",
    "InventoryTrace",
    "LayoutRegion",
    "PageInventory",
    "PageInventoryProvider",
]
