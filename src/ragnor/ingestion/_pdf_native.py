"""Native layout adapter internals, imported only inside the PDF worker."""

from __future__ import annotations

import io
import unicodedata
from dataclasses import replace

from ragnor.ingestion.extraction_types import (
    ExtractionCell,
    ExtractionRegion,
    ExtractionTable,
    ExtractionWord,
    PDFExtractionError,
)
from ragnor.ingestion.extractors import PageExtractionOutcome, PageExtractionStatus


def _box(bbox, page):
    x0, top, _, _ = page.bbox
    values = (
        (bbox[0] - x0) / page.width,
        (bbox[1] - top) / page.height,
        (bbox[2] - x0) / page.width,
        (bbox[3] - top) / page.height,
    )
    # Float round-off at a clipped page edge is not an out-of-page object.
    return tuple(min(1.0, max(0.0, v)) if -1e-9 <= v <= 1 + 1e-9 else v for v in values)


def _table(table, page, budget) -> ExtractionTable:
    budget.add("cells", len(table.cells))
    xs = sorted({round(v, 6) for box in table.cells for v in (box[0], box[2])})
    ys = sorted({round(v, 6) for box in table.cells for v in (box[1], box[3])})
    positions_x = {value: index for index, value in enumerate(xs)}
    positions_y = {value: index for index, value in enumerate(ys)}
    texts = {}
    for row, values in zip(table.rows, table.extract(), strict=True):
        for cell, value in zip(row.cells, values, strict=True):
            if cell is not None:
                texts[tuple(round(v, 6) for v in cell)] = value or ""
    cells = []
    for bounds in table.cells:
        key = tuple(round(v, 6) for v in bounds)
        x0, y0, x1, y1 = key
        text = texts[key]
        budget.add("text_chars", len(text))
        cells.append(
            ExtractionCell(
                positions_y[y0],
                positions_x[x0],
                text,
                _box(bounds, page),
                positions_y[y1] - positions_y[y0],
                positions_x[x1] - positions_x[x0],
            )
        )
    return ExtractionTable(_box(table.bbox, page), len(ys) - 1, len(xs) - 1, tuple(cells))


def _rtl_share(text: str) -> float:
    letters = [c for c in text if unicodedata.bidirectional(c) in {"L", "R", "AL"}]
    return (
        sum(unicodedata.bidirectional(c) in {"R", "AL"} for c in letters) / len(letters)
        if letters
        else 0.0
    )


def _is_rtl_word(text: str) -> bool:
    return _rtl_share(text) > 0.5


def to_logical_order(objects: list[dict]) -> list[dict]:
    """Turn pdfminer's visual-order words on right-to-left lines into logical order.

    pdfminer returns glyphs left to right as drawn, so an Arabic or Hebrew line
    arrives with its words and the characters inside each word reversed.
    Lines are runs of words with the same baseline. A line whose letters are
    mostly right-to-left has its words reversed and each right-to-left word's
    characters reversed; left-to-right runs (Latin words, numbers) keep their
    internal order. Left-to-right lines are returned untouched.
    """
    out: list[dict] = []
    line: list[dict] = []

    def flush() -> None:
        if not line:
            return
        if _rtl_share(" ".join(w["text"] for w in line)) <= 0.5:
            out.extend(line)
            line.clear()
            return
        fixed = [
            {**w, "text": w["text"][::-1]} if _is_rtl_word(w["text"]) else w for w in reversed(line)
        ]
        # reversing the line also reversed runs of LTR words; restore them
        i = 0
        while i < len(fixed):
            if _is_rtl_word(fixed[i]["text"]):
                i += 1
                continue
            j = i
            while j < len(fixed) and not _is_rtl_word(fixed[j]["text"]):
                j += 1
            fixed[i:j] = fixed[i:j][::-1]
            i = j
        out.extend(fixed)
        line.clear()

    for word in objects:
        if line and abs(word["top"] - line[-1]["top"]) > 2:
            flush()
        line.append(word)
    flush()
    return out


def _text(words, tables, include_tables):
    if not include_tables:
        return " ".join(word.text for word in words)
    emitted, parts = set(), []
    for word in words:
        x, y = (word.bbox[0] + word.bbox[2]) / 2, (word.bbox[1] + word.bbox[3]) / 2
        index = next(
            (
                i
                for i, table in enumerate(tables)
                if table.bbox[0] <= x <= table.bbox[2] and table.bbox[1] <= y <= table.bbox[3]
            ),
            None,
        )
        if index is None:
            parts.append(word.text)
        elif index not in emitted:
            parts.append(tables[index].to_html())
            emitted.add(index)
    parts.extend(table.to_html() for i, table in enumerate(tables) if i not in emitted)
    return " ".join(parts)


def extract_native(data, baseline, config, budget):
    import pdfplumber  # noqa: PLC0415 -- optional worker dependency

    results = []
    with (
        pdfplumber.open(io.BytesIO(data)) as document,
        pdfplumber.open(io.BytesIO(data), laparams={}) as analysis,
    ):
        if len(document.pages) != len(baseline):
            raise PDFExtractionError("invalid_document")
        for original, evidence in zip(document.pages, baseline, strict=True):
            try:
                page = original.crop(original.mediabox)
                objects = to_logical_order(page.extract_words(**config["word_settings"]))
                budget.add("words", len(objects))
                words, unlocated_words = [], 0
                for word in objects:
                    try:
                        words.append(
                            ExtractionWord(
                                word["text"],
                                _box((word["x0"], word["top"], word["x1"], word["bottom"]), page),
                            )
                        )
                    except ValueError:
                        unlocated_words += 1
                words = tuple(words)
                budget.add("text_chars", sum(len(word["text"]) for word in objects))
                tables, rejected_tables = [], 0
                for detected in page.find_tables(config["table_settings"]):
                    try:
                        tables.append(_table(detected, page, budget))
                    except (ValueError, KeyError):
                        # Ambiguous vendor grids must not erase all page text.
                        rejected_tables += 1
                tables = tuple(tables)
                layout = analysis.pages[evidence.page - 1].crop(original.mediabox)
                regions = tuple(
                    ExtractionRegion(
                        "text", _box((obj["x0"], obj["top"], obj["x1"], obj["bottom"]), page)
                    )
                    for obj in layout.objects.get("textboxhorizontal", [])
                )
                text = (
                    " ".join(word["text"] for word in objects)
                    if unlocated_words
                    else _text(words, tables, config["include_tables"])
                )
                status = evidence.status
                if text and (
                    rejected_tables or unlocated_words or status != PageExtractionStatus.EXTRACTED
                ):
                    status = PageExtractionStatus.PARTIAL
                elif not text and status == PageExtractionStatus.EXTRACTED:
                    status = PageExtractionStatus.FAILED
                results.append(
                    replace(
                        evidence,
                        status=status,
                        text=text,
                        provider="pdfplumber",
                        version="2",
                        words=words,
                        tables=tables,
                        regions=regions,
                        reason="native_text_extracted"
                        if status == PageExtractionStatus.EXTRACTED
                        else "visual_verification_required"
                        if status == PageExtractionStatus.PARTIAL
                        else evidence.reason,
                        diagnostics={
                            "route": "native",
                            "rejected_tables": rejected_tables,
                            "unlocated_words": unlocated_words,
                            "verification": "pending"
                            if status == PageExtractionStatus.PARTIAL
                            else "not_required",
                        },
                    )
                )
            except PDFExtractionError:
                raise
            except Exception:
                results.append(
                    PageExtractionOutcome(
                        evidence.page,
                        PageExtractionStatus.FAILED,
                        reason="native_page_failed",
                        provider="pdfplumber",
                        version="2",
                    )
                )
            finally:
                original.close()
                analysis.pages[evidence.page - 1].close()
    return results
