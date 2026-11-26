"""Private renderer; called only inside the bounded SDK worker."""

from __future__ import annotations

import base64
import io
from dataclasses import asdict

from ragnor.ingestion.extraction_types import PDFExtractionError


def render_pages(data, baseline, config, limits):
    import pypdfium2 as pdfium  # noqa: PLC0415

    selected = set(config.get("page_numbers") or range(1, len(baseline) + 1))
    if not selected <= {p.page for p in baseline}:
        raise PDFExtractionError("invalid_document")
    if len(selected) > config["max_ocr_pages"]:
        raise PDFExtractionError("resource_limit")
    images, size = [], 0
    with pdfium.PdfDocument(data) as document:
        document.init_forms()
        for number in sorted(selected):
            page = document[number - 1]
            try:
                page.set_cropbox(*page.get_mediabox())
                scale = config["dpi"] / 72
                import math  # noqa: PLC0415

                if (
                    math.ceil(page.get_width() * scale) * math.ceil(page.get_height() * scale)
                    > limits.max_pixels
                ):
                    raise PDFExtractionError("resource_limit")
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB").copy()
                finally:
                    bitmap.close()
                try:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    size += len(encoded)
                    if size > limits.max_output_bytes:
                        raise PDFExtractionError("output_limit")
                    images.append(
                        {
                            "page": number,
                            "image_base64": encoded,
                            "width": image.width,
                            "height": image.height,
                        }
                    )
                finally:
                    image.close()
            finally:
                page.close()
    return {"pages": [asdict(p) for p in baseline], "images": images}
