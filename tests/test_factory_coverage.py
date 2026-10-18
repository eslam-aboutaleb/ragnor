"""Coverage tests for the store factory and lazy backend imports.

Reaches the error paths the happy-path factory tests cannot:
a backend module that exists but fails to import (the factory
wraps the driver's ImportError with an install hint), the lazy
``__getattr__`` re-raise when a backend's driver is missing,
and the AttributeError for unknown attributes on
``ragnor.stores``.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
from typing import Any

import pytest

import ragnor.stores
from ragnor.stores.store_factory import RegistryVectorStoreFactory

BROKEN_MODULE = "ragnor.stores.broken"
QDRANT_MODULE = "ragnor.stores.qdrant"


def test_load_builtin_provider_wraps_driver_import_error() -> None:
    """A backend module that exists but fails to import is wrapped."""
    real_find_spec = importlib.util.find_spec
    real_import_module = importlib.import_module

    def fake_find_spec(name: str, package: str | None = None) -> Any:
        if name == BROKEN_MODULE:
            return importlib.machinery.ModuleSpec(name, None)
        return real_find_spec(name, package)

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == BROKEN_MODULE:
            raise ImportError("No module named 'qdrant_client'")
        return real_import_module(name, package)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        with pytest.raises(ImportError, match=r"could not be loaded") as exc_info:
            RegistryVectorStoreFactory()._load_builtin_provider("broken")

    message = str(exc_info.value)
    assert "ragnor[broken]" in message
    assert "No module named 'qdrant_client'" in message


def test_lazy_backend_import_error_includes_install_hint() -> None:
    """A missing driver surfaces as an ImportError naming the extra."""
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == QDRANT_MODULE:
            raise ImportError("No module named 'qdrant_client'")
        return real_import_module(name, package)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        with pytest.raises(ImportError, match=r"ragnor\[qdrant\]"):
            ragnor.stores.QdrantStore  # noqa: B018


def test_unknown_backend_attribute_raises_attribute_error() -> None:
    """Unknown attributes on ragnor.stores raise AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute"):
        ragnor.stores.NoSuchBackend  # noqa: B018
