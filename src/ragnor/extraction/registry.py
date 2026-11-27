# ruff: noqa: B023 -- closures are called immediately within the loop iteration
"""Explicit provider registry with deterministic selection and safe fallback.

Nothing is imported or discovered automatically: a host registers factories
(or opts in to ``load_entry_points``). A provider marked ``remote`` is never
selected until the host enables it by id, so content cannot reach a hosted
service through default configuration.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ragnor.extraction.errors import ExtractionConfigError, ExtractionError
from ragnor.extraction.fingerprint import fingerprint_hash, safe_provider_options
from ragnor.extraction.types import (
    AttemptStatus,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionStatus,
    ReasonCode,
)

ENTRY_POINT_GROUP = "ragnor.extraction_providers"
_PROVIDER_ID = re.compile(r"[a-z][a-z0-9_\-]{0,63}")
_EXT_TO_MEDIA = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
}


class Mode(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    STRUCTURED = "structured"


class Capability(StrEnum):
    TEXT = "text"
    STRUCTURE = "structure"
    LAYOUT = "layout"
    TABLES = "tables"
    OCR = "ocr"
    FORMULAS = "formulas"
    FIGURES = "figures"
    RENDER = "render"
    VLM = "vlm"


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    factory: Callable[[Mapping[str, Any]], Any]
    formats: frozenset[str]
    capabilities: frozenset[Capability]
    remote: bool = False
    priority: int = 100
    version: str = "1"
    validate_options: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, compare=False
    )


def media_type_for(name: str = "", media_type: str = "") -> str:
    """Normalise a media type, falling back to the file extension."""
    mt = media_type.split(";", 1)[0].strip().lower()
    if mt:
        return mt
    dot = name.rsplit(".", 1)
    return _EXT_TO_MEDIA.get("." + dot[1].lower(), "") if len(dot) == 2 else ""


class ProviderRegistry:
    """Registry of host-injected extraction providers."""

    def __init__(self) -> None:
        self._specs: dict[str, ProviderSpec] = {}
        self._remote_enabled: set[str] = set()

    # -- registration ---------------------------------------------------
    def register(  # noqa: PLR0913 -- explicit, keyword-only provider description
        self,
        provider_id: str,
        factory: Callable[[Mapping[str, Any]], Any],
        *,
        formats: Iterable[str],
        capabilities: Iterable[Capability | str],
        remote: bool = False,
        priority: int = 100,
        version: str = "1",
        validate_options: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> ProviderSpec:
        if not isinstance(provider_id, str) or not _PROVIDER_ID.fullmatch(provider_id):
            raise ExtractionConfigError("invalid_provider")
        if provider_id in self._specs:
            raise ExtractionConfigError("duplicate_provider")
        if (
            not callable(factory)
            or type(priority) is not int
            or not isinstance(version, str)
            or not version
        ):
            raise ExtractionConfigError("invalid_provider")
        try:
            caps = frozenset(Capability(c) for c in capabilities)
        except ValueError:
            raise ExtractionConfigError("invalid_provider") from None
        fmts = frozenset(str(f).lower() for f in formats)
        if not fmts or not caps:
            raise ExtractionConfigError("invalid_provider")
        spec = ProviderSpec(
            provider_id, factory, fmts, caps, bool(remote), priority, version, validate_options
        )
        self._specs[provider_id] = spec
        return spec

    def unregister(self, provider_id: str) -> None:
        if provider_id not in self._specs:
            raise ExtractionConfigError("unknown_provider")
        del self._specs[provider_id]
        self._remote_enabled.discard(provider_id)

    def enable_remote(self, provider_id: str) -> None:
        """Host opt-in: allow this remote provider to receive document content."""
        if provider_id not in self._specs:
            raise ExtractionConfigError("unknown_provider")
        self._remote_enabled.add(provider_id)

    def disable_remote(self, provider_id: str) -> None:
        self._remote_enabled.discard(provider_id)

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        """Explicit opt-in discovery of installed third-party providers.

        Each entry point must be a callable ``register(registry)``. Entry
        points are only loaded when the host calls this method.
        """
        from importlib.metadata import entry_points  # noqa: PLC0415

        before = set(self._specs)
        for ep in sorted(entry_points(group=group), key=lambda e: e.name):
            try:
                register = ep.load()
                register(self)
            except ExtractionError:
                raise
            except Exception:
                raise ExtractionConfigError("invalid_provider") from None
        return sorted(set(self._specs) - before)

    # -- lookup ---------------------------------------------------------
    def get(self, provider_id: str) -> ProviderSpec:
        try:
            return self._specs[provider_id]
        except KeyError:
            raise ExtractionConfigError("unknown_provider") from None

    def ids(self) -> list[str]:
        return sorted(self._specs)

    def select(
        self,
        *,
        media_type: str,
        mode: Mode | str = Mode.AUTO,
        require: Iterable[Capability | str] = (),
        providers: Iterable[str] | None = None,
    ) -> list[ProviderSpec]:
        """Deterministic fallback order for ``mode``; fails closed.

        ``text`` needs ``Capability.TEXT``; ``structured`` needs
        ``Capability.STRUCTURE``; ``auto`` tries structured providers first and
        then text-only ones. Ties break on ``(priority, provider_id)``.
        """
        try:
            mode = Mode(mode)
            needed = frozenset(Capability(c) for c in require)
        except ValueError:
            raise ExtractionConfigError("invalid_mode") from None
        candidates = list(self._specs.values())
        if providers is not None:
            candidates = [self.get(pid) for pid in dict.fromkeys(providers)]
        mt = media_type.lower()
        supported = [s for s in candidates if mt in s.formats or "*/*" in s.formats]
        if not supported:
            raise ExtractionConfigError("unsupported_format")
        capable = [s for s in supported if needed <= s.capabilities]
        if not capable:
            raise ExtractionConfigError("incompatible_capabilities")
        if mode is Mode.TEXT:
            chosen = [s for s in capable if Capability.TEXT in s.capabilities]
        elif mode is Mode.STRUCTURED:
            chosen = [s for s in capable if Capability.STRUCTURE in s.capabilities]
        else:
            structured = [s for s in capable if Capability.STRUCTURE in s.capabilities]
            text = [s for s in capable if Capability.STRUCTURE not in s.capabilities]
            chosen = [*_ordered(structured), *_ordered(text)]
        if mode is not Mode.AUTO:
            chosen = _ordered(chosen)
        if not chosen:
            raise ExtractionConfigError("incompatible_capabilities")
        allowed = [s for s in chosen if not s.remote or s.provider_id in self._remote_enabled]
        if not allowed:
            raise ExtractionConfigError("remote_not_enabled")
        return allowed

    def create(self, provider_id: str, options: Mapping[str, Any] | None = None) -> Any:
        """Instantiate a provider after validating its options."""
        spec = self.get(provider_id)
        opts = dict(options or {})
        try:
            if spec.validate_options is not None:
                spec.validate_options(opts)
            provider = spec.factory(opts)
        except ExtractionError:
            raise
        except Exception:
            raise ExtractionConfigError("invalid_options") from None
        if (
            getattr(provider, "id", None) != provider_id
            or str(getattr(provider, "version", None)) != spec.version
            or not callable(getattr(provider, "extract", None))
        ):
            raise ExtractionConfigError("invalid_provider")
        return provider

    def fingerprint(
        self, provider_id: str, options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Snapshot identity of a registered provider: id, version, safe options."""
        spec = self.get(provider_id)
        return {
            "id": spec.provider_id,
            "version": spec.version,
            "options": safe_provider_options(options),
        }

    # -- execution ------------------------------------------------------
    async def extract(  # noqa: PLR0913, PLR0912, PLR0915
        self,
        data: bytes,
        *,
        revision: str,
        name: str = "",
        media_type: str = "",
        mode: Mode | str = Mode.AUTO,
        providers: Iterable[str] | None = None,
        options: Mapping[str, Mapping[str, Any]] | None = None,
        require: Iterable[Capability | str] = (),
        timeout: float = 60.0,
        accept_partial: bool = False,
    ) -> ExtractionResult:
        """Run providers in fallback order; fail closed with typed errors.

        Raises :class:`ExtractionError` with the attempts recorded when every
        provider failed (``fallback_exhausted``) or only partial output was
        produced while ``accept_partial`` is False (``partial_only``). The
        running provider is closed on timeout, error and cancellation.
        """
        if not (
            isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
        ):
            raise ExtractionConfigError("invalid_options")
        media = media_type_for(name, media_type)
        order = self.select(media_type=media, mode=mode, require=require, providers=providers)
        attempts: list[ExtractionAttempt] = []
        best_partial: ExtractionResult | None = None
        for spec in order:
            provider = self.create(spec.provider_id, (options or {}).get(spec.provider_id))
            fp_hash = fingerprint_hash(
                self.fingerprint(spec.provider_id, (options or {}).get(spec.provider_id))
            )
            version = spec.version

            def record(status: AttemptStatus, reason: ReasonCode | None, pages: int = 0) -> None:
                attempts.append(
                    ExtractionAttempt(spec.provider_id, version, status, reason, fp_hash, pages)
                )

            try:
                extract: Callable[..., Awaitable[Any]] = provider.extract
                result = await asyncio.wait_for(
                    extract(data, name=name, revision=revision), timeout
                )
            except asyncio.CancelledError:
                record(AttemptStatus.CANCELLED, ReasonCode.CANCELLED)
                await _aclose(provider)
                raise
            except TimeoutError:
                record(AttemptStatus.TIMED_OUT, ReasonCode.TIMEOUT)
                await _aclose(provider)
                continue
            except Exception:
                record(AttemptStatus.FAILED, ReasonCode.PROVIDER_ERROR)
                await _aclose(provider)
                continue
            await _aclose(provider)
            if not isinstance(result, ExtractionResult):
                record(AttemptStatus.FAILED, ReasonCode.INVALID_OUTPUT)
                continue
            if result.status is ExtractionStatus.FAILED:
                record(AttemptStatus.FAILED, result.reason or ReasonCode.PROVIDER_ERROR)
                continue
            if result.status is ExtractionStatus.PARTIAL:
                record(AttemptStatus.PARTIAL, ReasonCode.PARTIAL_OUTPUT, len(result.pages))
                if best_partial is None or len(result.pages) > len(best_partial.pages):
                    best_partial = result
                continue
            record(AttemptStatus.SUCCEEDED, None, len(result.pages))
            return _with_attempts(result, attempts)
        if best_partial is not None and accept_partial:
            return _with_attempts(best_partial, attempts)
        raise ExtractionError(
            "partial_only" if best_partial is not None else "fallback_exhausted",
            attempts=attempts,
        )


def _ordered(specs: list[ProviderSpec]) -> list[ProviderSpec]:
    return sorted(specs, key=lambda s: (s.priority, s.provider_id))


def _with_attempts(result: ExtractionResult, attempts: list[ExtractionAttempt]) -> ExtractionResult:
    from dataclasses import replace  # noqa: PLC0415

    return replace(result, attempts=(*result.attempts, *attempts))


async def _aclose(provider: Any) -> None:
    closer = getattr(provider, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):
        await asyncio.shield(closer())
