"""Independently authored fake provider used by the structured-extraction tests.

It deliberately imports only the *public* ``ragnor.extraction`` API, the way a
third-party package would.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ragnor.extraction import (
    Cell,
    ExtractionResult,
    ExtractionStatus,
    HeaderScope,
    Page,
    Region,
    RegionKind,
    Table,
)


def sample_result(revision: str = "rev1", *, status=ExtractionStatus.COMPLETE) -> ExtractionResult:
    page = Page(1, 612.0, 792.0)
    box = page.box  # noqa: E731
    table = Table(
        3,
        3,
        (
            Cell(0, 0, "Item", header_scope=HeaderScope.COL, bbox=box(50, 300, 200, 320)),
            Cell(0, 1, "Q1-Q2", header_scope=HeaderScope.COL, colspan=2),
            Cell(1, 0, "Revenue", rowspan=2),
            Cell(1, 1, "1,234.50"),
            Cell(1, 2, "(12%)"),
            Cell(2, 1, "<script>x</script>"),
            Cell(2, 2, None),
        ),
    )
    regions = (
        Region(
            "r-head",
            RegionKind.HEADING,
            1,
            box(50, 50, 560, 80),
            0,
            text="Results",
            section_path=("Results",),
            method="fake",
        ),
        Region(
            "r-body",
            RegionKind.TEXT,
            1,
            box(50, 90, 560, 250),
            1,
            text="Body text",
            section_path=("Results",),
            method="fake",
            confidence=0.9,
        ),
        Region(
            "r-table", RegionKind.TABLE, 1, box(50, 300, 560, 400), 2, table=table, method="fake"
        ),
    )
    return ExtractionResult(
        revision,
        (Page(1, 612.0, 792.0, regions),),
        status=status,
    )


class FakeProvider:
    id = "fake"
    version = "1.0"

    def __init__(self, options: dict[str, Any] | None = None, *, behavior: str = "ok") -> None:
        self.options = dict(options or {})
        self.behavior = behavior
        self.closed = 0
        self.started = asyncio.Event()

    async def extract(self, data: bytes, *, name: str = "", revision: str):
        self.started.set()
        if self.behavior == "hang":
            await asyncio.sleep(3600)
        if self.behavior == "boom":
            raise RuntimeError("secret-api-key-123 leaked in provider message")
        if self.behavior == "garbage":
            return {"not": "a result"}
        if self.behavior == "partial":
            return sample_result(revision, status=ExtractionStatus.PARTIAL)
        if self.behavior == "failed":
            return ExtractionResult(revision, (), status=ExtractionStatus.FAILED)
        return sample_result(revision)

    async def aclose(self) -> None:
        self.closed += 1
