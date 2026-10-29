"""Offline benchmark adapters. No SDK registration or production dependency."""

from __future__ import annotations

import csv
import html
import io
import platform
import resource
import hashlib
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass, field
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
    regions: list[dict] | None = None
    semantic_fields: dict | None = None
    verified_complete: bool | None = None
    raster_metrics: dict = field(default_factory=dict)
    cpu_seconds: float | None = None
    evidence_sha256: str | None = None


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
            QualityPage(
                row.page,
                row.text,
                str(row.status),
                python_peak_rss_bytes=rss(),
                verified_complete=row.complete,
            )
            for row in outcomes
        ]


class PdfplumberProvider:
    supports_layout = True
    supports_tables = True
    id = "pdfplumber"
    options = {
        "use_text_flow": True,
        "table_strategy": "lines",
        "crop": "mediabox",
        "region_analysis": "separate laparams={}",
    }
    version = property(lambda self: version("pdfplumber"))

    def extract_outcomes(self, data: bytes) -> list[QualityPage]:
        import pdfplumber  # noqa: PLC0415

        result = []
        with (
            pdfplumber.open(io.BytesIO(data)) as document,
            pdfplumber.open(io.BytesIO(data), laparams={}) as analysis_document,
        ):
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
                analysis_page = analysis_document.pages[number - 1].crop(original_page.mediabox)
                blocks = [
                    {"bbox": box([block["x0"], block["top"], block["x1"], block["bottom"]])}
                    for block in analysis_page.objects.get("textboxhorizontal", [])
                ]
                result.append(
                    QualityPage(
                        number,
                        text,
                        "extracted" if text else "no_text",
                        words,
                        tables,
                        rss(),
                        regions=blocks,
                    )
                )
        return result


class TesseractProvider:
    supports_layout = True
    supports_tables = False
    id = "tesseract"
    options = {"dpi": 200, "language": "eng", "psm": 3}

    def __init__(self, *, language="eng", tessdata_dir: str | None = None):
        self.tessdata_dir = tessdata_dir
        self.options = {
            "dpi": 200,
            "language": language,
            "psm": 3,
            "renderer": "explicit_tsv",
            "subprocess_timeout_seconds": 30,
        }
        if tessdata_dir:
            self.options["traineddata_sha256"] = {
                name: hashlib.sha256(
                    (Path(tessdata_dir) / f"{name}.traineddata").read_bytes()
                ).hexdigest()
                for name in language.split("+")
            }

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

    def command(self):
        command = [
            self.executable(),
            "stdin",
            "stdout",
            "-l",
            self.options["language"],
            "--psm",
            "3",
        ]
        if self.tessdata_dir:
            command.extend(["--tessdata-dir", self.tessdata_dir])
        # A custom tessdata directory may contain models without configs/tsv.
        # Select renderers explicitly instead of relying on that external file.
        command.extend(["-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=0"])
        return command

    def extract_outcomes(self, data: bytes) -> list[QualityPage]:
        import pypdfium2 as pdfium  # noqa: PLC0415

        result = []
        with pdfium.PdfDocument(data) as document:
            for number in range(len(document)):
                page = document[number]
                bitmap = page.render(scale=self.options["dpi"] / 72)
                image = bitmap.to_pil().copy()
                output = io.BytesIO()
                image.save(output, format="PNG")
                width, height = image.size
                bitmap.close()
                page.close()
                completed = subprocess.run(  # noqa: S603 -- host executable, fixed arguments, no shell
                    self.command(),
                    input=output.getvalue(),
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                words, regions = parse_tsv(completed.stdout.decode(), width, height)
                text = " ".join(word["text"] for word in words)
                from benchmarks.coverage_metrics import ink_coverage  # noqa: PLC0415

                result.append(
                    QualityPage(
                        number + 1,
                        text,
                        "extracted" if text else "no_text",
                        words,
                        None,
                        rss(),
                        rss(children=True),
                        regions=regions,
                        raster_metrics=ink_coverage(image, words),
                    )
                )
        return result


def parse_tsv(raw: str, width: int, height: int) -> tuple[list[dict], list[dict]]:
    reader = csv.DictReader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {"level", "text", "left", "top", "width", "height", "block_num"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("OCR output is not the declared TSV protocol")
    words, blocks = [], {}
    for row in reader:
        if None in row or any(row[key] is None for key in required):
            raise ValueError("Malformed OCR word record")
        if row["level"] != "5" or not row["text"].strip():
            continue
        x, y, w, h = (int(row[key]) for key in ["left", "top", "width", "height"])
        words.append(
            {
                "text": row["text"],
                "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
            }
        )
        bounds = blocks.setdefault(row["block_num"], [1.0, 1.0, 0.0, 0.0])
        bounds[:] = [
            min(bounds[0], x / width),
            min(bounds[1], y / height),
            max(bounds[2], (x + w) / width),
            max(bounds[3], (y + h) / height),
        ]
    return words, [{"bbox": bounds} for bounds in blocks.values()]


class SDKProvider:
    """Measure installed SDK adapters, including their own worker deadline.

    No duplicate extraction implementation and no outer timeout process that
    could abandon the SDK worker's separate process group.
    """

    supports_layout = True
    supports_tables = True

    def __init__(self, *, engine="auto", language="eng", tessdata_dir=None):
        from ragnor.ingestion.pdf_extractors import (  # noqa: PLC0415
            AutoPDFExtractor,
            PdfPlumberExtractor,
            TesseractPDFExtractor,
        )

        self.runtime_attestations = {}
        self.engine = engine
        self.id = "sdk-" + engine
        native = PdfPlumberExtractor()
        ocr = TesseractPDFExtractor(language=language, tessdata_dir=tessdata_dir)
        self.adapter = {"native": native, "ocr": ocr}.get(engine)
        if engine == "auto":
            self.adapter = AutoPDFExtractor(native=native, ocr=ocr)
        if self.adapter is None:
            raise ValueError("Unknown SDK engine")
        self.supports_tables = engine != "ocr"

    @property
    def version(self):
        return self.adapter.version

    @property
    def options(self):
        return {**self.adapter.options, "runtime_attestations": self.runtime_attestations}

    def extract_bounded(self, data, timeout):
        from dataclasses import asdict, replace  # noqa: PLC0415

        self.adapter.limits = replace(self.adapter.limits, timeout_seconds=timeout)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        pages = self.adapter.extract_outcomes(data)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        import json  # noqa: PLC0415

        for page in pages:
            runtime = page.diagnostics.get("runtime", {})
            digest = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
            self.runtime_attestations[digest] = runtime
        result = [
            QualityPage(
                page=row.page,
                text=row.text,
                status=str(row.status),
                words=[asdict(word) for word in row.words],
                tables=[{"bbox": table.bbox, "html": table.to_html()} for table in row.tables]
                if self.supports_tables
                else None,
                regions=[asdict(region) for region in row.regions],
                verified_complete=row.complete,
                evidence_sha256=row.evidence_sha256,
            )
            for row in pages
        ]
        if result:
            result[0].cpu_seconds = (
                after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            )
        return result

    def extract_outcomes(self, data):
        return self.extract_bounded(data, self.adapter.limits.timeout_seconds)


class SDKNativeProvider(SDKProvider):
    def __init__(self):
        super().__init__(engine="native")


class SDKOCRProvider(SDKProvider):
    def __init__(self, **kwargs):
        super().__init__(engine="ocr", **kwargs)


PROVIDERS: dict[str, Any] = {
    "sdk-native": SDKNativeProvider,
    "sdk-ocr": SDKOCRProvider,
    "sdk-auto": SDKProvider,
    "pypdf": PypdfProvider,
    "pdfplumber": PdfplumberProvider,
    "tesseract": TesseractProvider,
}
