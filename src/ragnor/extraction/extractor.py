"""Opt-in structured extractor compatible with the legacy ``Extractor`` seam.

``PDFExtractor.extract`` / ``extract_pages`` are unchanged. A host that wants
structured output constructs a :class:`StructuredExtractor` over a registry it
configured; nothing is selected or sent anywhere by default.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from ragnor.extraction.adapters import to_raw_documents
from ragnor.extraction.errors import ExtractionConfigError
from ragnor.extraction.fingerprint import safe_provider_options
from ragnor.extraction.registry import Capability, Mode, ProviderRegistry
from ragnor.extraction.serializers import region_text
from ragnor.extraction.types import SCHEMA_VERSION, ExtractionResult
from ragnor.ingestion.records import RawDocument


class StructuredExtractor:
    """Extractor identity covering mode, providers, versions and safe options.

    ``options`` feeds ``ingestion.versioning.build_snapshot`` so any change to
    a provider id, version, option or the mode produces a different snapshot
    and therefore a deterministic re-index, while an unchanged configuration
    compares equal.
    """

    id = "structured"
    version = "1"

    def __init__(  # noqa: PLR0913 -- explicit host configuration
        self,
        registry: ProviderRegistry,
        *,
        mode: Mode | str = Mode.STRUCTURED,
        providers: Iterable[str] | None = None,
        provider_options: Mapping[str, Mapping[str, Any]] | None = None,
        require: Iterable[Capability | str] = (),
        timeout: float = 60.0,
        accept_partial: bool = False,
    ) -> None:
        self._registry = registry
        self._mode = Mode(mode)
        self._providers = tuple(providers) if providers is not None else None
        self._provider_options = {k: dict(v) for k, v in (provider_options or {}).items()}
        self._require = tuple(require)
        self._timeout = timeout
        self._accept_partial = accept_partial
        for pid in self._providers or ():
            registry.get(pid)
        for pid in self._provider_options:
            registry.get(pid)

    @property
    def options(self) -> dict[str, Any]:
        providers = self._providers if self._providers is not None else tuple(self._registry.ids())
        return {
            "mode": self._mode.value,
            "schema_version": SCHEMA_VERSION,
            "accept_partial": self._accept_partial,
            "require": sorted(str(c) for c in self._require),
            "providers": [
                self._registry.fingerprint(pid, self._provider_options.get(pid))
                for pid in providers
            ],
            "extractor_options": safe_provider_options({"timeout": self._timeout}),
        }

    async def aextract_result(
        self, data: bytes, *, name: str = "", media_type: str = "", revision: str | None = None
    ) -> ExtractionResult:
        return await self._registry.extract(
            data,
            revision=revision or hashlib.sha256(data).hexdigest(),
            name=name,
            media_type=media_type,
            mode=self._mode,
            providers=self._providers,
            options=self._provider_options,
            require=self._require,
            timeout=self._timeout,
            accept_partial=self._accept_partial,
        )

    def extract_result(
        self, data: bytes, *, name: str = "", media_type: str = ""
    ) -> ExtractionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aextract_result(data, name=name, media_type=media_type))
        raise ExtractionConfigError("invalid_mode")  # use aextract_result inside a running loop

    def extract_pages(self, data: bytes, *, name: str = "") -> list[tuple[int, str]]:
        result = self.extract_result(data, name=name)
        return [
            (
                page.number,
                "\n\n".join(t for r in page.ordered_regions() if (t := region_text(r))),
            )
            for page in sorted(result.pages, key=lambda p: p.number)
        ]

    def extract(self, data: bytes, *, name: str = "") -> str:
        return "\n\n".join(text for _, text in self.extract_pages(data, name=name) if text)

    async def aextract_documents(
        self, data: bytes, *, source_name: str, external_id: str | None = None, **kwargs: Any
    ) -> list[RawDocument]:
        result = await self.aextract_result(data, name=source_name)
        return to_raw_documents(
            result,
            source_name=source_name,
            external_id=external_id,
            extractor=self.id,
            **kwargs,
        )
