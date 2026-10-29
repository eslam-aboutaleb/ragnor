"""Independent PDF quality metrics; no document contents in returned scores.

TEDS follows the tree-edit definition in https://arxiv.org/abs/1911.10683.
The implementation canonicalizes table sections and inline markup; this is a
normalized table TEDS variant, not a claim of official leaderboard parity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from benchmarks.pdf_extraction import edit_distance, text_error_rate


def normalize(text: str) -> str:
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:[.,]\w+)*|%", normalize(text).casefold())


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<!\w)[+-]?\d[\d,.]*(?:%)?", normalize(text))


def multiset_recall(reference: list[str], actual: list[str]) -> float | None:
    expected = Counter(reference)
    return sum((expected & Counter(actual)).values()) / len(reference) if reference else None


@dataclass
class Cell:
    text: str = ""
    rowspan: int = 1
    colspan: int = 1


class TableParser(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[Cell]] = []
        self.cell: Cell | None = None
        self.depth = 0
        self.feed(html)
        self.close()
        if self.depth or self.cell is not None or not self.rows:
            raise ValueError("Invalid or empty table")
        self._validate_grid()

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth > 1:
                raise ValueError("Nested tables require a separate schema")
        elif tag == "tr":
            if self.cell is not None:
                raise ValueError("Unclosed table cell")
            self.rows.append([])
        elif tag in {"td", "th"}:
            if not self.rows or self.cell is not None:
                raise ValueError("Cell outside row")
            attributes = dict(attrs)
            spans = [int(attributes.get(key, 1)) for key in ("rowspan", "colspan")]
            if any(value < 1 or value > 1000 for value in spans):
                raise ValueError("Invalid table span")
            self.cell = Cell(rowspan=spans[0], colspan=spans[1])
            self.rows[-1].append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell.text += " "

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            if self.cell is not None:
                self.cell.text = normalize(self.cell.text)
            self.cell = None
        elif tag == "table":
            self.depth -= 1

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.text += data

    def _validate_grid(self):
        if sum(len(row) for row in self.rows) > 1000:
            raise ValueError("Table exceeds metric cell limit")
        self.grid()  # Validate that spans cannot extend beyond supplied rows.

    def grid(self) -> dict[tuple[int, int], Cell]:
        result = {}
        for row_index, row in enumerate(self.rows):
            column = 0
            for cell in row:
                while (row_index, column) in result:
                    column += 1
                if row_index + cell.rowspan > len(self.rows):
                    raise ValueError("Rowspan extends beyond table")
                for r in range(row_index, row_index + cell.rowspan):
                    for c in range(column, column + cell.colspan):
                        if (r, c) in result:
                            raise ValueError("Overlapping table cells")
                        result[r, c] = cell
                column += cell.colspan
        return result


@dataclass
class Node:
    tag: str
    cell: Cell | None = None
    children: list[Node] = field(default_factory=list)


def _tree(table: TableParser) -> Node:
    return Node(
        "table",
        children=[Node("tr", children=[Node("td", cell) for cell in row]) for row in table.rows],
    )


def table_scores(reference_html: str, actual_html: str) -> dict[str, float]:
    reference, actual = TableParser(reference_html), TableParser(actual_html)

    from apted import APTED, Config  # noqa: PLC0415 -- benchmark-only dependency

    class Costs(Config):
        structure_only = False

        def rename(self, left, right):
            if left.tag != right.tag:
                return 1
            if left.cell is None:
                return 0
            a, b = left.cell, right.cell
            if (a.rowspan, a.colspan) != (b.rowspan, b.colspan):
                return 1
            if self.structure_only:
                return 0
            return edit_distance(a.text, b.text) / max(len(a.text), len(b.text), 1)

    size = max(
        1 + len(table.rows) + sum(len(row) for row in table.rows) for table in (reference, actual)
    )
    costs = Costs()
    teds = max(
        0.0, 1 - APTED(_tree(reference), _tree(actual), costs).compute_edit_distance() / size
    )
    costs.structure_only = True
    structure = max(
        0.0, 1 - APTED(_tree(reference), _tree(actual), costs).compute_edit_distance() / size
    )
    expected, predicted = reference.grid(), actual.grid()
    exact = sum(
        key in predicted and cell.text == predicted[key].text for key, cell in expected.items()
    )
    spans = sum(
        key in predicted
        and (cell.rowspan, cell.colspan) == (predicted[key].rowspan, predicted[key].colspan)
        for key, cell in expected.items()
    )
    numeric = {key: cell for key, cell in expected.items() if numeric_tokens(cell.text)}
    correct_numbers = sum(
        key in predicted and numeric_tokens(cell.text) == numeric_tokens(predicted[key].text)
        for key, cell in numeric.items()
    )
    return {
        "teds_normalized": teds,
        "teds_structure": structure,
        "cell_exact_accuracy": exact / max(len(expected), 1),
        "span_accuracy": spans / max(len(expected), 1),
        "numeric_cell_accuracy": correct_numbers / len(numeric) if numeric else None,
    }


def validate_bbox(box: list[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box):
        raise ValueError("Invalid normalized bounding box")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Empty or inverted bounding box")


def iou(left: list[float], right: list[float]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )

    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    return intersection / max(area(left) + area(right) - intersection, 1e-12)


def match_boxes(reference: list[dict], actual: list[dict], threshold=0.5) -> list[tuple[int, int]]:
    candidates = sorted(
        (
            (iou(a["bbox"], b["bbox"]), i, j)
            for i, a in enumerate(reference)
            for j, b in enumerate(actual)
        ),
        reverse=True,
    )
    found, used_reference, used_actual = [], set(), set()
    for score, i, j in candidates:
        if score >= threshold and i not in used_reference and j not in used_actual:
            found.append((i, j))
            used_reference.add(i)
            used_actual.add(j)
    return found


def layout_scores(regions: list[dict], words: list[dict]) -> dict[str, Any]:
    """Score region text omission and upstream pair order using word centers."""
    expected = [region for region in regions if region.get("text")]
    grouped: dict[int, list[str]] = {i: [] for i in range(len(expected))}
    first_positions = {}
    for position, word in enumerate(words):
        x0, y0, x1, y1 = word["bbox"]
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        matches = [
            i
            for i, region in enumerate(expected)
            if region["bbox"][0] <= x <= region["bbox"][2]
            and region["bbox"][1] <= y <= region["bbox"][3]
        ]
        if matches:
            i = min(
                matches,
                key=lambda i: (expected[i]["bbox"][2] - expected[i]["bbox"][0])
                * (expected[i]["bbox"][3] - expected[i]["bbox"][1]),
            )
            grouped[i].append(word["text"])
            first_positions.setdefault(i, position)
    total_pairs = len(expected) * (len(expected) - 1) // 2
    pairs = [(i, j) for i in first_positions for j in first_positions if i < j]
    recalls = [
        multiset_recall(tokens(region["text"]), tokens(" ".join(grouped[i])))
        for i, region in enumerate(expected)
    ]
    return {
        "text_regions": len(expected),
        "empty_text_regions": sum(not grouped[i] for i in range(len(expected))),
        "region_token_recall": (
            sum(x for x in recalls if x is not None) / len(recalls) if recalls else None
        ),
        "reading_order_pair_accuracy": (
            sum(first_positions[i] < first_positions[j] for i, j in pairs) / len(pairs)
        )
        if pairs
        else None,
        "reading_order_pair_coverage": len(pairs) / total_pairs if total_pairs else None,
    }


def page_text_scores(reference: str, actual: str) -> dict[str, Any]:
    reference, actual = normalize(reference), normalize(actual)
    return {
        "cer": text_error_rate(reference, actual),
        "wer": text_error_rate(reference.split(), actual.split()),
        "token_recall": multiset_recall(tokens(reference), tokens(actual)),
        "numeric_token_recall": multiset_recall(numeric_tokens(reference), numeric_tokens(actual)),
    }
