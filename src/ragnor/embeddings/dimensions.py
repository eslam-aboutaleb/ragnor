"""Embedding dimension registry.

Single source of truth mapping embedding model names to
their vector dimensions, so changing the embedding model cannot
silently desynchronize the dimension checks at insert time.

The built-in table covers the models ragnor knows about;
:func:`register_dimension` is the public extensibility seam for
models the table does not know (host applications register their
provider's dimension at start-up).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Dimension per supported embedding provider. ``text-embedding-3-small``
# is the only provider configured today; OpenAI's ``text-embedding-ada-002``
# is the historical model and shares its dimension.
_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

FALLBACK_DIMENSION = 1536


def register_dimension(model: str, dimension: int) -> None:
    """Register the vector dimension for an embedding model.

    Extensibility seam replacing the private dimension table: host
    applications and tests can register dimensions for models the
    built-in table does not know about.

    Args:
        model: Embedding model name.
        dimension: Vector dimension the model produces.
    """
    _DIMENSIONS[model] = dimension


def has_registered_embedding_dimension(model: str) -> bool:
    """Check if dimension is declared, without using fallback or logging."""
    return model in _DIMENSIONS


def get_embedding_dimension(model: str | None = None) -> int:
    """Return the vector dimension for an embedding model.

    Known models return their registered dimension. Unknown models
    warn and fall back to :data:`FALLBACK_DIMENSION` rather than
    raising, so a configuration-only change does not crash start-up;
    the warning makes the wrong dimension visible.

    Args:
        model: The embedding model name. Host applications resolve
            their configured model and pass it explicitly; a bare
            call has no model to look up and takes the same
            warn-and-fallback path as an unknown model.

    Returns:
        The dimension, or the fallback when the model is not in the
        known table.
    """
    if model is None:
        logger.warning(
            "No embedding model specified; assuming dimension %d. "
            "Pass the configured model name to get_embedding_dimension().",
            FALLBACK_DIMENSION,
        )
        return FALLBACK_DIMENSION

    dimension = _DIMENSIONS.get(model)
    if dimension is not None:
        return dimension

    logger.warning(
        "No embedding dimension registered for model '%s'; assuming %d. "
        "Add it with ragnor.embeddings.dimensions.register_dimension() if that is wrong.",
        model,
        FALLBACK_DIMENSION,
    )
    return FALLBACK_DIMENSION
