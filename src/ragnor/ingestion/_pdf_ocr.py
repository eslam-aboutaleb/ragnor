"""Local Tesseract/PDFium implementation, isolated from SDK host threads."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from ragnor.ingestion.extraction_types import (
    ExtractionRegion,
    ExtractionWord,
    PDFExtractionError,
)
from ragnor.ingestion.extractors import PageExtractionStatus


def _run(command, args, timeout, max_output, data=None):
    with tempfile.TemporaryFile() as output:
        subprocess.run(  # noqa: S603 -- explicit host executable, argv only, never a shell
            [command, *args],
            input=data,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
        output.seek(0)
        value = output.read(max_output + 1)
    if len(value) > max_output:
        raise PDFExtractionError("output_limit")
    return value


def prepare(config):
    command = shutil.which(config["executable"])
    if command is None:
        raise ImportError("Install the host Tesseract executable")
    model_dir = config["tessdata_dir"]
    if model_dir is None:
        listing = _run(command, ["--list-langs"], 5, 32_000).decode()
        match = re.search(r'"([^"\n]+)"', listing.splitlines()[0])
        if match is None:
            raise PDFExtractionError("provider_failed")
        model_dir = match.group(1)
    models = {}
    languages = set(config["language"].split("+"))
    if config["psm"] == 12:
        languages.add("osd")
    for language in sorted(languages):
        path = Path(model_dir) / f"{language}.traineddata"
        if not path.is_file():
            raise ImportError("Install the selected Tesseract language data")
        with path.open("rb") as stream:
            models[language] = hashlib.file_digest(stream, "sha256").hexdigest()
    version_line = _run(command, ["--version"], 5, 32_000).decode().splitlines()[0]
    match = re.search(r"\btesseract\s+(v?\d[\w.+-]*)", version_line, flags=re.I)
    if match is None:
        raise PDFExtractionError("provider_failed")
    with Path(command).open("rb") as stream:
        executable_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return (
        command,
        str(model_dir),
        {
            "version": "tesseract " + match.group(1),
            "models_sha256": models,
            "executable_sha256": executable_hash,
        },
    )


def parse_words(raw, width, height, budget):
    reader = csv.DictReader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {
        "level",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    if not required <= set(reader.fieldnames or []):
        raise ValueError("Invalid OCR TSV protocol")
    words, blocks, lines = [], {}, []
    last_line = None
    for row in reader:
        if None in row or any(row[key] is None for key in required):
            raise ValueError("Malformed OCR record")
        if row["level"] != "5" or not row["text"].strip():
            continue
        x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
        confidence = float(row["conf"])
        if not math.isfinite(confidence) or not -1 <= confidence <= 100:
            raise ValueError("Invalid OCR confidence")
        word = ExtractionWord(
            row["text"],
            (x / width, y / height, (x + w) / width, (y + h) / height),
            confidence if confidence >= 0 else None,
        )
        budget.add("words", 1)
        budget.add("text_chars", len(word.text))
        words.append(word)
        line = (row["block_num"], row["par_num"], row["line_num"])
        if line != last_line:
            lines.append([])
            last_line = line
        lines[-1].append(word.text)
        box = blocks.setdefault(row["block_num"], list(word.bbox))
        box[:] = [
            min(box[0], word.bbox[0]),
            min(box[1], word.bbox[1]),
            max(box[2], word.bbox[2]),
            max(box[3], word.bbox[3]),
        ]
    return (
        tuple(words),
        tuple(ExtractionRegion("text", tuple(box)) for box in blocks.values()),
        "\n".join(" ".join(line) for line in lines),
    )


def _coverage(image, words):
    from PIL import Image, ImageChops, ImageDraw  # noqa: PLC0415 -- optional OCR dependency

    gray = image.convert("L")
    histogram = gray.histogram()
    count = sum(histogram)
    total = sum(i * n for i, n in enumerate(histogram))
    weight, partial, best, threshold = 0, 0, -1.0, 0
    for i, n in enumerate(histogram):
        weight += n
        partial += i * n
        if weight in {0, count}:
            continue
        variance = (
            weight
            * (count - weight)
            * (partial / weight - (total - partial) / (count - weight)) ** 2
        )
        if variance > best:
            best, threshold = variance, i
    # A uniform non-white image is unknown visual content, not a verified blank.
    if best < 0:
        return {
            "word_box_ink_coverage": None,
            "foreground_pixels": 0 if histogram[255] == count else count,
        }
    threshold = min(threshold, 245)
    ink = gray.point(lambda value: 255 if value <= threshold else 0)
    foreground = ink.histogram()[255]
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for word in words:
        x0, y0, x1, y1 = word.bbox
        draw.rectangle(
            [
                int(x0 * image.width),
                int(y0 * image.height),
                math.ceil(x1 * image.width) - 1,
                math.ceil(y1 * image.height) - 1,
            ],
            fill=255,
        )
    covered = ImageChops.multiply(ink, mask).histogram()[255]
    return {
        "word_box_ink_coverage": covered / foreground if foreground else None,
        "foreground_pixels": foreground,
        "ink_threshold": threshold,
    }


def extract_ocr(data, baseline, config, budget, selected):
    import pypdfium2 as pdfium  # noqa: PLC0415 -- only used in an isolated process

    command, model_dir, runtime = prepare(config)
    results = {}
    with pdfium.PdfDocument(data) as document:
        document.init_forms()
        if len(document) != len(baseline):
            raise PDFExtractionError("invalid_document")
        for number in sorted(selected):
            evidence = baseline[number - 1]
            page = document[number - 1]
            try:
                page.set_cropbox(*page.get_mediabox())
                scale = config["dpi"] / 72
                width, height = page.get_width(), page.get_height()
                if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                    raise ValueError("Invalid page dimensions")
                if math.ceil(width * scale) * math.ceil(height * scale) > budget.limits.max_pixels:
                    raise PDFExtractionError("resource_limit")
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB").copy()
                finally:
                    bitmap.close()
                payload = io.BytesIO()
                image.save(payload, format="PNG")
                raw = _run(
                    command,
                    [
                        "stdin",
                        "stdout",
                        "-l",
                        config["language"],
                        "--psm",
                        str(config["psm"]),
                        "--tessdata-dir",
                        model_dir,
                        "-c",
                        "tessedit_create_tsv=1",
                        "-c",
                        "tessedit_create_txt=0",
                    ],
                    config["page_timeout"],
                    budget.limits.max_output_bytes,
                    payload.getvalue(),
                )
                words, regions, text = parse_words(raw.decode(), image.width, image.height, budget)
                diagnostics = _coverage(image, words)
                confidence = [word.confidence for word in words if word.confidence is not None]
                diagnostics.update(
                    {
                        "route": "ocr",
                        "verification": "pending",
                        "word_count": len(words),
                        "mean_confidence": sum(confidence) / len(confidence)
                        if confidence
                        else None,
                        "pixel_width": image.width,
                        "pixel_height": image.height,
                    }
                )
                blank = image.getextrema() == ((255, 255), (255, 255), (255, 255)) and not words
                status = (
                    PageExtractionStatus.VERIFIED_BLANK
                    if blank
                    else (PageExtractionStatus.PARTIAL if text else PageExtractionStatus.NEEDS_OCR)
                )
                results[number] = replace(
                    evidence,
                    status=status,
                    text=text,
                    words=words,
                    regions=regions,
                    provider="tesseract",
                    version="1",
                    diagnostics=diagnostics,
                    reason="rendered_white_page"
                    if blank
                    else "ocr_verification_required"
                    if text
                    else "ocr_no_text",
                )
            except PDFExtractionError:
                raise
            except Exception as exc:
                reason = (
                    "ocr_timeout"
                    if isinstance(exc, subprocess.TimeoutExpired)
                    else "ocr_page_failed"
                )
                results[number] = replace(
                    evidence,
                    status=PageExtractionStatus.FAILED,
                    text="",
                    reason=reason,
                    provider="tesseract",
                    version="1",
                )
            finally:
                page.close()
    checks = {Path(command): runtime["executable_sha256"]}
    checks.update(
        {
            Path(model_dir) / f"{name}.traineddata": digest
            for name, digest in runtime["models_sha256"].items()
        }
    )
    for path, expected in checks.items():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise PDFExtractionError("runtime_changed")
    return results, runtime
