"""Public, dependency-light Python API for the Ragnor SDK.

The lazy exports keep `import ragnor` free of optional providers and
preserve established module-level imports. See docs/sdk/api-stability.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Ragnor", "RagnorBuilder", "Document", "Chunk", "SearchResult", "RetrievalError"]

if TYPE_CHECKING:
    from ragnor.builder import Ragnor, RagnorBuilder
    from ragnor.config import DeploymentMode as DeploymentMode
    from ragnor.config import RagnorConfig as RagnorConfig
    from ragnor.sync_client import AsyncRagnorOperations as AsyncRagnorOperations
    from ragnor.sync_client import SyncRagnor as SyncRagnor
    from ragnor.types import Chunk, Document, RetrievalError, SearchResult


def __getattr__(name: str) -> Any:
    if name in {"Ragnor", "RagnorBuilder"}:
        from ragnor.builder import Ragnor, RagnorBuilder  # noqa: PLC0415

        return {"Ragnor": Ragnor, "RagnorBuilder": RagnorBuilder}[name]
    if name in {"Document", "Chunk", "SearchResult", "RetrievalError"}:
        from ragnor.types import Chunk, Document, RetrievalError, SearchResult  # noqa: PLC0415

        return {
            "Document": Document,
            "Chunk": Chunk,
            "SearchResult": SearchResult,
            "RetrievalError": RetrievalError,
        }[name]
    if name in {"DeploymentMode", "RagnorConfig"}:
        from ragnor.config import DeploymentMode, RagnorConfig  # noqa: PLC0415

        return {"DeploymentMode": DeploymentMode, "RagnorConfig": RagnorConfig}[name]
    if name in {"SyncRagnor", "AsyncRagnorOperations"}:
        from ragnor.sync_client import AsyncRagnorOperations, SyncRagnor  # noqa: PLC0415

        return {"SyncRagnor": SyncRagnor, "AsyncRagnorOperations": AsyncRagnorOperations}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
