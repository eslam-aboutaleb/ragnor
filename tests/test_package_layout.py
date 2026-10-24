"""Public import, pickle and host-environment compatibility across the package move."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ALIASES = {
    "builder": "client.builder",
    "sync_client": "client.sync_client",
    "types": "core.types",
    "exceptions": "core.exceptions",
    "config": "core.config",
    "registry": "core.registry",
    "lifecycle": "core.lifecycle",
    "validation": "core.validation",
    "metrics": "stores.metrics",
    "connectors": "ingestion.connectors",
    "events": "telemetry.events",
}


@pytest.mark.parametrize("canonical_first", [False, True])
def test_legacy_and_canonical_imports_share_module_identity(canonical_first):
    script = f"""
import importlib
aliases = {ALIASES!r}
for old, new in aliases.items():
    names = ['ragnor.' + old, 'ragnor.' + new]
    if {canonical_first!r}:
        names.reverse()
    first, second = [importlib.import_module(name) for name in names]
    assert first is second, (old, new)
    assert first.__spec__.name == 'ragnor.' + new, first.__spec__
from ragnor.events import __all__
assert set(__all__) == {{'CompositeEventSink', 'EventKind', 'EventSink',
    'LoggingEventSink', 'MemoryEventSink', 'NullEventSink', 'RedactionConfig',
    'TelemetryEvent', 'sanitize_error_category'}}
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and constant module names
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_import_and_reload_do_not_install_global_import_hooks():
    script = """
import importlib
import sys
before = tuple(sys.meta_path)
import ragnor
assert tuple(sys.meta_path) == before
importlib.reload(ragnor)
assert tuple(sys.meta_path) == before
import pickle
from ragnor.core.types import Document
assert pickle.loads(b'cragnor.types\\nDocument\\n.') is Document
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and constant script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_test_configuration_never_probes_or_selects_a_local_database(monkeypatch):
    monkeypatch.delenv("RAGNOR_TEST_DATABASE_URL", raising=False)

    def no_implicit_network(*args, **kwargs):
        raise AssertionError("loading test configuration must not probe local services")

    monkeypatch.setattr(socket, "create_connection", no_implicit_network)
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("isolated_test_configuration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "RAGNOR_TEST_DATABASE_URL" not in os.environ
