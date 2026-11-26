"""Core stays dependency-free; third-party providers plug in via the public port."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ragnor.extraction import ExtractionConfigError, ProviderRegistry

SRC = Path(__file__).resolve().parents[1] / "src"
VENDORS = [
    "litellm",
    "pypdf",
    "pypdfium2",
    "PIL",
    "docling",
    "docling_core",
    "sqlalchemy",
    "asyncpg",
    "httpx",
    "boto3",
    "numpy",
    "qdrant_client",
    "chromadb",
    "faiss",
    "tiktoken",
    "mcp",
    "fastapi",
    "opentelemetry",
]


def run(script: str, *, extra_path: str = "") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(SRC), extra_path) if p)}
    return subprocess.run(  # noqa: S603 - trusted constants only
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


def test_importing_the_extraction_namespace_loads_no_optional_engine():
    script = f"""
    import sys
    for name in {VENDORS!r}:
        sys.modules[name] = None  # any import now raises ImportError
    import ragnor, ragnor.extraction
    from ragnor.extraction import ProviderRegistry, StructuredExtractor
    loaded = [m for m in sys.modules if m.split('.')[0] in {VENDORS!r} and sys.modules[m] is not None]
    assert not loaded, loaded
    print('ok')
    """
    proc = run(script)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_core_dependency_list_stays_empty():
    text = (SRC.parent / "pyproject.toml").read_text()
    assert "\ndependencies = []\n" in text


def _write_third_party_package(root: Path) -> None:
    pkg = root / "acme_ragnor_ocr"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent(
            """
            from ragnor.extraction import (
                BoundingBox, Capability, ExtractionResult, Page, Region, RegionKind,
            )

            class AcmeProvider:
                id = "acme-ocr"
                version = "3.1"

                def __init__(self, options):
                    self.options = dict(options)
                    self.closed = False

                async def extract(self, data, *, name="", revision):
                    page = Page(1, 100.0, 100.0)
                    region = Region(
                        "acme-1", RegionKind.TEXT, 1, page.box(1, 1, 90, 20), 0,
                        text=data.decode(), method="acme",
                    )
                    return ExtractionResult(revision, (Page(1, 100.0, 100.0, (region,)),))

                async def aclose(self):
                    self.closed = True

            def register(registry):
                registry.register(
                    "acme-ocr",
                    AcmeProvider,
                    formats=["text/plain"],
                    capabilities=[Capability.TEXT, Capability.STRUCTURE],
                    version="3.1",
                )
            """
        )
    )
    dist = root / "acme_ragnor_ocr-3.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: acme-ragnor-ocr\nVersion: 3.1\n")
    (dist / "entry_points.txt").write_text(
        "[ragnor.extraction_providers]\nacme = acme_ragnor_ocr:register\n"
    )


def test_installed_third_party_provider_integrates_without_core_edits(tmp_path):
    _write_third_party_package(tmp_path)
    script = """
    import asyncio, sys
    from ragnor.extraction import ProviderRegistry
    registry = ProviderRegistry()
    assert registry.ids() == []          # nothing is auto-imported or auto-registered
    assert 'acme_ragnor_ocr' not in sys.modules
    assert registry.load_entry_points() == ['acme-ocr']
    result = asyncio.run(registry.extract(b'hello', revision='r', name='a.txt'))
    region = next(result.regions())
    assert region.text == 'hello' and result.attempts[0].provider_version == '3.1'
    print('ok')
    """
    proc = run(script, extra_path=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_entry_point_failures_are_sanitized(tmp_path):
    _write_third_party_package(tmp_path)
    (tmp_path / "acme_ragnor_ocr" / "__init__.py").write_text(
        "raise RuntimeError('SECRET-token-leak')\n"
    )
    script = """
    from ragnor.extraction import ProviderRegistry, ExtractionConfigError
    try:
        ProviderRegistry().load_entry_points()
    except ExtractionConfigError as exc:
        assert exc.code == 'invalid_provider' and 'SECRET' not in str(exc)
        print('ok')
    """
    proc = run(script, extra_path=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_entry_point_group_with_nothing_installed_is_empty():
    assert ProviderRegistry().load_entry_points("ragnor.no_such_group") == []


def test_registered_provider_closure_under_cancellation_and_timeout():
    registry = ProviderRegistry()
    closed: list[str] = []

    class Slow:
        id = "slow"
        version = "1"
        options: dict = {}

        async def extract(self, data, *, name="", revision):
            await asyncio.sleep(10)

        async def aclose(self):
            closed.append("slow")

    from ragnor.extraction import Capability

    registry.register(
        "slow",
        lambda o: Slow(),
        formats=["text/plain"],
        capabilities=[Capability.TEXT, Capability.STRUCTURE],
    )
    with pytest.raises(Exception) as err:
        asyncio.run(registry.extract(b"x", revision="r", name="a.txt", timeout=0.05))
    assert err.value.code == "fallback_exhausted" and closed == ["slow"]
    with pytest.raises(ExtractionConfigError):
        registry.create("slow", {}) and registry.create("nope")


def test_routing_imports_and_runs_without_any_optional_engine():
    script = f"""
    import sys
    for name in {VENDORS!r}:
        sys.modules[name] = None
    from ragnor.extraction import Available, Capability, PageKind, PageProfile, route_page
    d = route_page(PageProfile(1, PageKind.SCAN), Available(local=frozenset({{Capability.OCR}})))
    assert d.route.value == "ocr", d
    print('ok')
    """
    result = run(script)
    assert result.returncode == 0, result.stderr
