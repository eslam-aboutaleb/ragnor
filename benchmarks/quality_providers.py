"""Offline benchmark adapters. No SDK registration or production dependency."""

from __future__ import annotations

import csv
import html
import io
import platform
import resource
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from ragnor.ingestion.extractors import PDFExtractor


@dataclass
class QualityPage:
    page: int
    text: str
    status: str
    words: list[dict] | None = None
    tables: list[dict] | None = None
    python_peak_rss_bytes: int = 0
    ocr_peak_rss_bytes: int | None = None


def rss(children=False) -> int:
    result = resource.getrusage(
        resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    ).ru_maxrss
    return int(result if platform.system() == "Darwin" else result * 1024)


class PypdfProvider:
    supports_layout = False
    supports_tables = False
    id = "pypdf"
    options = {"policy": "strict"}
    version = property(lambda self: version("pypdf"))

    def extract_outcomes(self, data: bytes) -> list[QualityPage]:
        outcomes = PDFExtractor().extract_outcomes(data)
        return [
            QualityPage(row.page, row.text, str(row.status), python_peak_rss_bytes=rss())
            for row in outcomes
        ]


class PdfplumberProvider:
    supports_layout = True
    supports_tables = True
    id = "pdfplumber"
    options = {"use_text_flow": True, "table_strategy": "lines", "crop": "mediabox"}
    version = property(lambda self: version("pdfplumber"))

    def extract_outcomes(self, data: bytes) -> list[QualityPage]:
        import pdfplumber  # noqa: PLC0415

        result = []
        with pdfplumber.open(io.BytesIO(data)) as document:
            for number, original_page in enumerate(document.pages, 1):
                # Upstream annotations refer to MediaBox renders, not PDF CropBox.
                page = original_page.crop(original_page.mediabox)
                x0, y0, _, _ = page.bbox

                def box(value, origin=(x0, y0), size=(page.width, page.height)):
                    return [
                        (value[0] - origin[0]) / size[0],
                        (value[1] - origin[1]) / size[1],
                        (value[2] - origin[0]) / size[0],
                        (value[3] - origin[1]) / size[1],
                    ]

                words = [
                    {
                        "text": word["text"],
                        "bbox": box([word["x0"], word["top"], word["x1"], word["bottom"]]),
                    }
                    for word in page.extract_words(use_text_flow=True)
                ]
                tables = []
                for table in page.find_tables():
                    rows = table.extract()
                    markup = (
                        "<table>"
                        + "".join(
                            "<tr>"
                            + "".join("<td>" + html.escape(cell or "") + "</td>" for cell in row)
                            + "</tr>"
                            for row in rows
                        )
                        + "</table>"
                    )
                    tables.append({"bbox": box(table.bbox), "html": markup})
                text = " ".join(word["text"] for word in words)
                result.append(
                    QualityPage(
                        number, text, "extracted" if text else "no_text", words, tables, rss()
                    )
                )
        return result


class TesseractProvider:
    supports_layout = True
    supports_tables = False
    id = "tesseract"
    options = {"dpi": 200, "language": "eng", "psm": 3}

    @staticmethod
    def executable():
        command = shutil.which("tesseract")
        if command is None:
            raise RuntimeError("Install the host Tesseract executable")
        return command

    @property
    def version(self):
        return subprocess.run(  # noqa: S603 -- host executable, fixed arguments, no shell
            [self.executable(), "--version"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.splitlines()[0]

    def extract_outcomes(self, data: bytes) -> list[QualityPage]:
        import pypdfium2 as pdfium  # noqa: PLC0415

        result = []
        with pdfium.PdfDocument(data) as document:
            for number in range(len(document)):
                page = document[number]
                bitmap = page.render(scale=self.options["dpi"] / 72)
                image = bitmap.to_pil()
                output = io.BytesIO()
                image.save(output, format="PNG")
                width, height = image.size
                bitmap.close()
                page.close()
                completed = subprocess.run(  # noqa: S603 -- host executable, fixed arguments, no shell
                    [self.executable(), "stdin", "stdout", "-l", "eng", "--psm", "3", "tsv"],
                    input=output.getvalue(),
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                words = []
                for row in csv.DictReader(io.StringIO(completed.stdout.decode()), delimiter="\t"):
                    if row["level"] != "5" or not row["text"].strip():
                        continue
                    x, y, w, h = (int(row[key]) for key in ["left", "top", "width", "height"])
                    words.append(
                        {
                            "text": row["text"],
                            "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
                        }
                    )
                text = " ".join(word["text"] for word in words)
                result.append(
                    QualityPage(
                        number + 1,
                        text,
                        "extracted" if text else "no_text",
                        words,
                        None,
                        rss(),
                        rss(children=True),
                    )
                )
        return result


PROVIDERS: dict[str, Any] = {
    "pypdf": PypdfProvider,
    "pdfplumber": PdfplumberProvider,
    "tesseract": TesseractProvider,
}
