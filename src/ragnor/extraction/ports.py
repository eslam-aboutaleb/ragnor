"""Provider-neutral extraction ports (protocols).

Adapters own heavy imports, credentials, network access and lifecycle. Ragnor
core never imports an engine and never sends content to a remote service on
its own: a host registers a provider explicitly (see ``registry``).

Async and cancellation contract for every port method:

* Methods are coroutines and must be cancellation-safe: on
  ``asyncio.CancelledError`` release held resources and re-raise.
* Ragnor wraps calls in a timeout (``asyncio.wait_for``); a provider must not
  swallow the resulting cancellation or keep running work in the background.
* ``aclose()`` is awaited exactly once per provider instance after use, also
  after timeout, cancellation or failure.
* Errors may carry provider detail internally, but Ragnor reports only a fixed
  code (see :class:`ragnor.extraction.errors.ExtractionError`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ragnor.extraction.types import AssetRef, DocumentProfile, ExtractionResult, Region


@runtime_checkable
class ExtractionProvider(Protocol):
    """Identity shared by every provider; ``options`` must be non-secret JSON."""

    id: str
    version: str
    options: Mapping[str, Any]

    async def aclose(self) -> None: ...


@runtime_checkable
class PageProfiler(Protocol):
    async def profile(self, data: bytes, *, name: str = "") -> DocumentProfile: ...


@runtime_checkable
class PageRenderer(Protocol):
    async def render(self, data: bytes, page: int, *, dpi: int = 150) -> AssetRef: ...


@runtime_checkable
class LayoutDetector(Protocol):
    async def detect(self, data: bytes, page: int) -> Sequence[Region]: ...


@runtime_checkable
class OCREngine(Protocol):
    async def recognize(self, data: bytes, page: int, *, language: str = "eng") -> str: ...


@runtime_checkable
class RegionExtractor(Protocol):
    async def extract_region(self, data: bytes, region: Region) -> Region: ...


@runtime_checkable
class DocumentExtractor(Protocol):
    async def extract(self, data: bytes, *, name: str = "", revision: str) -> ExtractionResult: ...


@runtime_checkable
class Verifier(Protocol):
    async def verify(self, result: ExtractionResult) -> ExtractionResult: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Host-owned, access-controlled store for page images and crops."""

    async def put(self, kind: str, data: bytes, *, media_type: str) -> AssetRef: ...

    async def get(self, ref: AssetRef) -> bytes: ...
