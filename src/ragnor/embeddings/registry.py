"""Embedding provider registry.

Providers plug in through :data:`embedding_registry` without
editing core code. A provider is a factory class registered
under a name that:

1. Accepts a :class:`ragnor.core.config.RagSettings` object in its
   constructor, and
2. Exposes a ``create()`` method returning an
   :class:`ragnor.embeddings.base.EmbeddingFunction`.

Example::

    @embedding_registry.register("my-provider")
    class MyEmbeddingProvider:
        def __init__(self, settings: RagSettings) -> None:
            self._settings = settings

        def create(self) -> EmbeddingFunction:
            return MyEmbeddingFunction(self._settings.api_key)

:func:`get_embedding_function` resolves the provider name from
``settings.embedding_provider`` (defaulting to ``"litellm"`` when
the settings object does not define the key) and instantiates the
registered provider. No second provider ships with ragnor -- the
seam is the deliverable.
"""

from __future__ import annotations

from ragnor.core.config import RagSettings
from ragnor.embeddings.base import EmbeddingFunction
from ragnor.core.registry import Registry

embedding_registry: Registry[EmbeddingFunction] = Registry()


@embedding_registry.register("litellm")
class LiteLLMEmbeddingProvider:
    """Factory that builds :class:`LitellmEmbeddingFunction` from settings."""

    def __init__(self, settings: RagSettings) -> None:
        self._settings = settings

    def create(self) -> EmbeddingFunction:
        """Return a LiteLLM embedding function configured from settings."""
        try:
            from ragnor.embeddings.litellm import (  # noqa: PLC0415
                LitellmEmbeddingFunction,
            )
        except ImportError as exc:
            raise ImportError(
                "The LiteLLM integration requires the 'litellm' extra: "
                "pip install 'ragnor[litellm]'"
            ) from exc
        return LitellmEmbeddingFunction(
            api_key=self._settings.openai_api_key,
            model_name=self._settings.embedding_model,
            api_base=self._settings.openai_api_base,
        )


def get_embedding_function(settings: RagSettings) -> EmbeddingFunction:
    """Return the embedding function selected by ``settings``.

    The provider name is read from ``settings.embedding_provider``.
    The key is optional: settings objects that do not define it
    (such as the host application's settings until they adopt the
    field) fall back to the ``"litellm"`` provider, which preserves
    the historical behavior.

    Args:
        settings: Settings exposing ``embedding_model`` and
            ``openai_api_key`` (and optionally ``embedding_provider``).

    Returns:
        An embedding function instance ready for use by the
        retrieval and ingestion pipelines.

    Raises:
        KeyError: If the provider name is not registered. The
            message lists the registered provider names.
    """
    provider_name = getattr(settings, "embedding_provider", "litellm")
    provider_cls = embedding_registry.get(provider_name)
    return provider_cls(settings).create()
