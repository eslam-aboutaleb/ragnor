"""Dependency hygiene: the core install has zero third-party dependencies.

Invariant: ``import ragit.*`` never requires litellm, a database
driver, or a vector-DB client. The LiteLLM integration lives
behind the ``litellm`` extra and is imported lazily -- through
the PEP 562 ``__getattr__`` in ``ragit.embeddings`` and a
function-level import in ``LiteLLMEmbeddingProvider.create()``.

Every litellm-blocked assertion runs in a subprocess: setting
``sys.modules["litellm"] = None`` makes ``import litellm`` raise
ImportError (Python 3 semantics for ``None`` entries), and the
subprocess isolates that poisoning from the test process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import ragit.embeddings

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

_BLOCK_LITELLM = "import sys; sys.modules['litellm'] = None"


def _run_without_litellm(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a subprocess where ``litellm`` cannot be imported."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    )
    return subprocess.run(  # noqa: S603 - trusted constants only
        [sys.executable, "-c", f"{_BLOCK_LITELLM}\n{textwrap.dedent(script)}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


def _assert_success(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"


def test_import_ragit_without_litellm() -> None:
    script = """
    try:
        import litellm
    except ImportError:
        pass
    else:
        raise AssertionError("litellm should be blocked in this subprocess")

    import ragit
    import ragit.embeddings
    """
    _assert_success(_run_without_litellm(script))


def test_get_embedding_function_importable_without_litellm() -> None:
    script = """
    from ragit.embeddings import get_embedding_function

    assert callable(get_embedding_function)
    """
    _assert_success(_run_without_litellm(script))


def test_lazy_litellm_names_raise_helpful_import_error_without_litellm() -> None:
    script = """
    import ragit.embeddings

    try:
        ragit.embeddings.LitellmEmbeddingFunction
    except ImportError as exc:
        assert "ragit[litellm]" in str(exc), str(exc)
    else:
        raise AssertionError("expected ImportError for LitellmEmbeddingFunction")

    # The provider class lives in the registry module, which no
    # longer imports litellm, so it resolves without the extra.
    assert ragit.embeddings.LiteLLMEmbeddingProvider is not None
    """
    _assert_success(_run_without_litellm(script))


def test_provider_create_raises_helpful_import_error_without_litellm() -> None:
    script = """
    from ragit.embeddings import embedding_registry, get_embedding_function


    class _Settings:
        embedding_model = "text-embedding-3-small"
        openai_api_key = "test-key"
        openai_api_base = ""
        embedding_provider = "litellm"


    settings = _Settings()

    # The provider stays registered and instantiable without litellm.
    provider = embedding_registry.get("litellm")(settings)

    try:
        provider.create()
    except ImportError as exc:
        assert "ragit[litellm]" in str(exc), str(exc)
    else:
        raise AssertionError("provider.create() should raise ImportError")

    # The settings-driven factory surfaces the same helpful error.
    try:
        get_embedding_function(settings)
    except ImportError as exc:
        assert "ragit[litellm]" in str(exc), str(exc)
    else:
        raise AssertionError("get_embedding_function should raise ImportError")
    """
    _assert_success(_run_without_litellm(script))


def test_litellm_names_importable_with_extra_installed() -> None:
    from ragit.embeddings import (
        LiteLLMEmbeddingProvider,
        LitellmEmbeddingFunction,
    )

    assert LitellmEmbeddingFunction.__module__ == "ragit.embeddings.litellm"
    assert LiteLLMEmbeddingProvider.__module__ == "ragit.embeddings.registry"


def test_all_exports_include_the_lazy_litellm_names() -> None:
    assert "LitellmEmbeddingFunction" in ragit.embeddings.__all__
    assert "LiteLLMEmbeddingProvider" in ragit.embeddings.__all__
