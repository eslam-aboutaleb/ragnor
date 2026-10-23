"""Public, dependency-light Python API for the Ragnor SDK.

The lazy exports keep `import ragnor` free of optional providers and
preserve established module-level imports. See docs/sdk/api-stability.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Ragnor", "RagnorBuilder", "Document", "Chunk", "SearchResult", "RetrievalError"]

if TYPE_CHECKING:
    from ragnor.builder import Ragnor, RagnorBuilder
    from ragnor.types import Chunk, Document, RetrievalError, SearchResult


def __getattr__(name: str) -> Any:
    if name in {"Ragnor", "RagnorBuilder"}:
        from ragnor.builder import Ragnor, RagnorBuilder

        return {"Ragnor": Ragnor, "RagnorBuilder": RagnorBuilder}[name]
    if name in {"Document", "Chunk", "SearchResult", "RetrievalError"}:
        from ragnor.types import Chunk, Document, RetrievalError, SearchResult

        return {
            "Document": Document,
            "Chunk": Chunk,
            "SearchResult": SearchResult,
            "RetrievalError": RetrievalError,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
