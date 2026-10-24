"""Public, dependency-light Python API for the Ragnor SDK.

The lazy exports keep `import ragnor` free of optional providers and
preserve established module-level imports. See docs/sdk/api-stability.md.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = ["Ragnor", "RagnorBuilder", "Document", "Chunk", "SearchResult", "RetrievalError"]

if TYPE_CHECKING:
    from ragnor.client.builder import Ragnor, RagnorBuilder
    from ragnor.client.sync_client import AsyncRagnorOperations as AsyncRagnorOperations
    from ragnor.client.sync_client import SyncRagnor as SyncRagnor
    from ragnor.core.config import DeploymentMode as DeploymentMode
    from ragnor.core.config import RagnorConfig as RagnorConfig
    from ragnor.core.types import Chunk, Document, RetrievalError, SearchResult


_LEGACY_MODULE_ALIASES: dict[str, str] = {
    "ragnor.builder": "ragnor.client.builder",
    "ragnor.sync_client": "ragnor.client.sync_client",
    "ragnor.types": "ragnor.core.types",
    "ragnor.exceptions": "ragnor.core.exceptions",
    "ragnor.config": "ragnor.core.config",
    "ragnor.registry": "ragnor.core.registry",
    "ragnor.lifecycle": "ragnor.core.lifecycle",
    "ragnor.validation": "ragnor.core.validation",
    "ragnor.metrics": "ragnor.stores.metrics",
    "ragnor.connectors": "ragnor.ingestion.connectors",
    "ragnor.events": "ragnor.telemetry.events",
}


def __getattr__(name: str) -> Any:
    if name in {"Ragnor", "RagnorBuilder"}:
        from ragnor.client.builder import Ragnor, RagnorBuilder  # noqa: PLC0415

        return {"Ragnor": Ragnor, "RagnorBuilder": RagnorBuilder}[name]
    if name in {"Document", "Chunk", "SearchResult", "RetrievalError"}:
        from ragnor.core.types import Chunk, Document, RetrievalError, SearchResult  # noqa: PLC0415

        return {
            "Document": Document,
            "Chunk": Chunk,
            "SearchResult": SearchResult,
            "RetrievalError": RetrievalError,
        }[name]
    if name in {"DeploymentMode", "RagnorConfig"}:
        from ragnor.core.config import DeploymentMode, RagnorConfig  # noqa: PLC0415

        return {"DeploymentMode": DeploymentMode, "RagnorConfig": RagnorConfig}[name]
    if name in {"SyncRagnor", "AsyncRagnorOperations"}:
        from ragnor.client.sync_client import AsyncRagnorOperations, SyncRagnor  # noqa: PLC0415

        return {"SyncRagnor": SyncRagnor, "AsyncRagnorOperations": AsyncRagnorOperations}[name]

    if name in {"core", "client"}:
        return importlib.import_module(f"ragnor.{name}")

    fullname = f"ragnor.{name}"
    if fullname in _LEGACY_MODULE_ALIASES:
        return importlib.import_module(fullname)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
