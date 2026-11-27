"""LiteLLM-backed embedder and page reader (vendor named only by env model string)."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.pdf_extraction.providers import ProviderPage
from benchmarks.real_models.budget import CostMeter
from benchmarks.real_models.config import Endpoint

FALLBACK_EMBED_USD_PER_1K_CHARS = 0.0001
FALLBACK_VLM_USD_PER_PAGE = 0.02
VLM_PROMPT = (
    "Transcribe every piece of text on this page in reading order. Keep table rows "
    "on one line each with cells separated by ' | '. Output the text only."
)


class ModelCallError(RuntimeError):
    """A model call failed. The message carries the error class, never provider text."""


def _cost(response: Any, fallback: float) -> float:
    import litellm  # noqa: PLC0415 -- optional benchmark dependency

    try:
        value = litellm.completion_cost(completion_response=response)
    except Exception:  # unknown model price: charge the conservative fallback
        return fallback
    return float(value) if value else fallback


@dataclass
class LiteLLMEmbedder:
    endpoint: Endpoint
    meter: CostMeter
    batch: int = 32
    latencies_ms: list[float] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        import litellm  # noqa: PLC0415

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start : start + self.batch]
            estimate = sum(len(t) for t in chunk) / 1000 * FALLBACK_EMBED_USD_PER_1K_CHARS
            self.meter.reserve(estimate)
            began = time.perf_counter()
            try:
                response = litellm.embedding(
                    model=self.endpoint.model,
                    input=chunk,
                    api_key=self.endpoint.api_key,
                    api_base=self.endpoint.api_base,
                )
            except Exception as exc:
                raise ModelCallError(f"embedding call failed: {type(exc).__name__}") from None
            self.latencies_ms.append((time.perf_counter() - began) * 1000)
            self.meter.charge("embeddings", _cost(response, estimate))
            out.extend(list(item["embedding"]) for item in response.data)
        return out


@dataclass
class LiteLLMPageReader:
    """EX-02 ``ExtractionProvider`` that transcribes each rendered page with a VLM."""

    endpoint: Endpoint
    meter: CostMeter
    max_pages: int = 40
    dpi: int = 130
    id: str = "vlm-transcribe"
    version: str = "1"
    config: dict[str, Any] = field(default_factory=dict)
    _pages_done: int = 0

    def __post_init__(self) -> None:
        self.id = "vlm-transcribe"
        self.config = {"model": self.endpoint.model, "dpi": self.dpi, "prompt": "transcribe-v1"}

    def library_versions(self) -> dict[str, str]:
        from importlib.metadata import version  # noqa: PLC0415

        return {"litellm": version("litellm"), "pypdfium2": version("pypdfium2")}

    def _render(self, data: bytes) -> list[bytes]:
        import pypdfium2 as pdfium  # noqa: PLC0415

        document = pdfium.PdfDocument(data)
        images = []
        for page in document:
            buffer = io.BytesIO()
            page.render(scale=self.dpi / 72).to_pil().convert("RGB").save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return images

    def extract(self, data: bytes, *, name: str = "") -> list[ProviderPage]:
        import litellm  # noqa: PLC0415

        pages = []
        for number, png in enumerate(self._render(data), start=1):
            if self._pages_done >= self.max_pages:
                raise ModelCallError("page limit reached; remaining documents were not run")
            self.meter.reserve(FALLBACK_VLM_USD_PER_PAGE)
            url = "data:image/png;base64," + base64.b64encode(png).decode()
            began = time.perf_counter()
            try:
                response = litellm.completion(
                    model=self.endpoint.model,
                    api_key=self.endpoint.api_key,
                    api_base=self.endpoint.api_base,
                    max_tokens=1500,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VLM_PROMPT},
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                )
            except Exception as exc:
                raise ModelCallError(f"vlm call failed: {type(exc).__name__}") from None
            cost = _cost(response, FALLBACK_VLM_USD_PER_PAGE)
            self.meter.charge("vlm", cost)
            self._pages_done += 1
            text = (response.choices[0].message.content or "").strip()
            pages.append(
                ProviderPage(
                    page=number,
                    text=text,
                    status="ok" if text else "empty",
                    duration_ms=(time.perf_counter() - began) * 1000,
                    cost_usd=cost,
                )
            )
        return pages
