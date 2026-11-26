"""A third-party extraction provider written only against ``ragnor.extraction``.

Run with ``python examples/custom_extraction_provider.py``. A real provider
package would expose ``register(registry)`` through the
``ragnor.extraction_providers`` entry-point group; the host then calls
``registry.load_entry_points()`` explicitly.
"""

from __future__ import annotations

import asyncio

from ragnor.extraction import (
    Capability,
    ExtractionResult,
    Page,
    ProviderRegistry,
    Region,
    RegionKind,
    result_to_text,
)


class CustomProvider:
    id = "custom-lines"
    version = "1.0"

    def __init__(self, options: dict) -> None:
        self.options = dict(options)

    async def extract(self, data: bytes, *, name: str = "", revision: str) -> ExtractionResult:
        blank = Page(1, 612.0, 792.0)
        regions = tuple(
            Region(
                f"line-{i}",
                RegionKind.TEXT,
                1,
                blank.box(50, 50 + 20 * i, 560, 65 + 20 * i),
                i,
                text=line,
                method="custom-lines",
            )
            for i, line in enumerate(data.decode().splitlines())
        )
        return ExtractionResult(revision, (Page(1, 612.0, 792.0, regions),))

    async def aclose(self) -> None:  # release engines, sockets, temp files
        return None


def register(registry: ProviderRegistry) -> None:
    registry.register(
        "custom-lines",
        CustomProvider,
        formats=["text/plain"],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
        version="1.0",
    )


async def main() -> None:
    registry = ProviderRegistry()
    register(registry)
    result = await registry.extract(b"first\nsecond", revision="demo", name="demo.txt")
    print(result_to_text(result))


if __name__ == "__main__":
    asyncio.run(main())
