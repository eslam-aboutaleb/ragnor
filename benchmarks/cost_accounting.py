"""Measured local usage with explicit pricing provenance; no invented host rate."""

from __future__ import annotations

import math


def cost_report(
    cpu_seconds: float | None,
    pages: int,
    *,
    cpu_usd_per_hour: float | None = None,
    pricing_source: str | None = None,
    external_api_usd: float = 0.0,
) -> dict:
    if pages < 1:
        raise ValueError("Cost accounting needs measured pages")
    for value in [cpu_seconds, cpu_usd_per_hour, external_api_usd]:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid usage or price")
    if cpu_usd_per_hour is not None and not pricing_source:
        raise ValueError("A host-provided price needs pricing provenance")
    units = cpu_seconds / 3600 * 1000 / pages if cpu_seconds is not None else None
    compute = (
        units * cpu_usd_per_hour if units is not None and cpu_usd_per_hour is not None else None
    )
    external = external_api_usd * 1000 / pages
    return {
        "scope": (
            "local provider CPU and observed external service charges; excludes idle/capital costs"
        ),
        "cpu_seconds": cpu_seconds,
        "cpu_hours_per_1000_pages": units,
        "cpu_usd_per_hour": cpu_usd_per_hour,
        "pricing_source": pricing_source,
        "external_api_usd_per_1000_pages": external,
        "compute_usd_per_1000_pages": compute,
        "total_usd_per_1000_pages": compute + external if compute is not None else None,
    }
