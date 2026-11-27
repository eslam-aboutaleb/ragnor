"""Provider-agnostic settings read from generic environment variable names.

Nothing here names a vendor. Key and base URL use the owner's gateway
convention (LITELLM_API_KEY, LITELLM_API_BASE Actions secrets). Models are
LiteLLM model strings passed as workflow inputs, not secrets. Secret values are
never printed, logged or written to results: ``Settings.public()`` carries only whether each is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# The only secrets the benchmark workflow reads (same names as the owner's other repos).
API_KEY = "LITELLM_API_KEY"
API_BASE = "LITELLM_API_BASE"
EMBEDDING_MODEL = "RAGNOR_BENCH_EMBEDDING_MODEL"
EMBEDDING_API_KEY = "RAGNOR_BENCH_EMBEDDING_API_KEY"
EMBEDDING_API_BASE = "RAGNOR_BENCH_EMBEDDING_API_BASE"
VLM_MODEL = "RAGNOR_BENCH_VLM_MODEL"
VLM_API_KEY = "RAGNOR_BENCH_VLM_API_KEY"
VLM_API_BASE = "RAGNOR_BENCH_VLM_API_BASE"
QDRANT_URL = "RAGNOR_BENCH_QDRANT_URL"  # set by the workflow, not a secret

SECRET_NAMES = (API_KEY, API_BASE)  # repository secrets
ENV_NAMES = (
    API_KEY,
    API_BASE,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE,
    VLM_MODEL,
    VLM_API_KEY,
    VLM_API_BASE,
)
SUITES = ("docling", "embeddings", "qdrant", "vlm")


class ConfigError(ValueError):
    """A suite cannot run because required settings are missing."""


@dataclass(frozen=True)
class Endpoint:
    model: str
    api_key: str
    api_base: str | None


@dataclass(frozen=True)
class Settings:
    embedding: Endpoint | None
    vlm: Endpoint | None
    qdrant_url: str | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)

        def endpoint(model_name: str, key_name: str, base_name: str) -> Endpoint | None:
            model = env.get(model_name, "").strip()
            key = env.get(key_name, "").strip() or env.get(API_KEY, "").strip()
            base = env.get(base_name, "").strip() or env.get(API_BASE, "").strip() or None
            return Endpoint(model, key, base) if model and key else None

        return cls(
            endpoint(EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_API_BASE),
            endpoint(VLM_MODEL, VLM_API_KEY, VLM_API_BASE),
            env.get(QDRANT_URL, "").strip() or None,
        )

    def require(self, suite: str) -> None:
        """Raise ``ConfigError`` naming the missing settings, never their values."""
        need = {
            "docling": [],
            "embeddings": []
            if self.embedding
            else [EMBEDDING_MODEL, f"{API_KEY} (or {EMBEDDING_API_KEY})"],
            "qdrant": (
                [] if self.embedding else [EMBEDDING_MODEL, f"{API_KEY} (or {EMBEDDING_API_KEY})"]
            )
            + ([] if self.qdrant_url else [QDRANT_URL]),
            "vlm": [] if self.vlm else [VLM_MODEL, f"{API_KEY} (or {VLM_API_KEY})"],
        }[suite]
        if need:
            raise ConfigError(f"suite {suite!r} needs: {', '.join(need)}")

    def public(self) -> dict[str, object]:
        return {
            "embedding_model": self.embedding.model if self.embedding else None,
            "vlm_model": self.vlm.model if self.vlm else None,
            "api_base_override": bool(
                (self.embedding and self.embedding.api_base) or (self.vlm and self.vlm.api_base)
            ),
            "qdrant": bool(self.qdrant_url),
        }
