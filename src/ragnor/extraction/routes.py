"""Per-region-type provider routes (tables, formulas, figures).

A route table maps a region kind to a registered provider id implementing the
``RegionExtractor`` port. Base layout comes from one provider; each routed
region is then refined by its route. A VLM route is used only when the host
configured it (and enabled it, if remote); absent configuration the region
stays as the base provider produced it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ragnor.extraction.registry import ProviderRegistry
from ragnor.extraction.types import ExtractionResult, Region, RegionKind


async def apply_routes(
    registry: ProviderRegistry,
    result: ExtractionResult,
    data: bytes,
    routes: Mapping[RegionKind | str, str],
    *,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> ExtractionResult:
    """Refine regions of routed kinds; fail closed if a route is unusable.

    Providers are created once per route, used, and closed. A failing route
    keeps the base region and records nothing about provider messages.
    """
    table = {RegionKind(k): pid for k, pid in routes.items()}
    for pid in set(table.values()):
        spec = registry.get(pid)  # unknown ids raise before any content is sent
        if spec.remote and pid not in registry._remote_enabled:  # noqa: SLF001
            from ragnor.extraction.errors import ExtractionConfigError  # noqa: PLC0415

            raise ExtractionConfigError("remote_not_enabled")
    providers = {pid: registry.create(pid, (options or {}).get(pid)) for pid in set(table.values())}
    try:
        pages = []
        for page in result.pages:
            regions = []
            for region in page.regions:
                pid = table.get(region.kind)
                refined: Region = region
                if pid is not None:
                    try:
                        candidate = await providers[pid].extract_region(data, region)
                        if (
                            isinstance(candidate, Region)
                            and candidate.region_id == region.region_id
                        ):
                            refined = candidate
                    except Exception:  # noqa: BLE001, S110 -- keep base region, no provider text
                        pass
                regions.append(refined)
            pages.append(replace(page, regions=tuple(regions)))
        return replace(result, pages=tuple(pages))
    finally:
        for provider in providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()
