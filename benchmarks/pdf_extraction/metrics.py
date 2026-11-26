"""Extraction quality metrics.

All metrics are pure functions over annotations and extracted text/structure,
so unit tests can pin hand-computed expected scores. Text comparison uses
NFKC normalization with whitespace collapsed to single spaces.

Table structure similarity is a sequence-based approximation of TEDS: the
table grid is serialized to a tag token stream (``<tr>``, ``<td rs=2 cs=1>``
...) and scored as ``1 - edit_distance / max_length``. It is deterministic
and sensitive to splits, merges and misplaced spans, but it is not the full
tree-edit-distance TEDS from the original paper; that gap is documented in
docs/extraction-benchmarks.md and in every report's ``metric_notes``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from benchmarks.pdf_extraction.manifest import PageAnnotation, Region, Table

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_text(text: str) -> str:
    """NFKC-normalize and collapse whitespace so layout artifacts don't dominate."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def squashed(text: str) -> str:
    """Normalized text with all whitespace removed.

    Containment checks use this form because PDF extractors legitimately
    differ on whether adjacent text runs are separated by spaces or newlines;
    word order and characters still have to match.
    """
    return _WS.sub("", unicodedata.normalize("NFKC", text))


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance over arbitrary token sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(expected: str, actual: str) -> float | None:
    """Character error rate; ``None`` when there is no ground truth to score."""
    expected_n = normalize_text(expected)
    if not expected_n:
        return None
    return edit_distance(list(expected_n), list(normalize_text(actual))) / len(expected_n)


def word_error_rate(expected: str, actual: str) -> float | None:
    """Word error rate; ``None`` when there is no ground truth to score."""
    expected_n = normalize_text(expected)
    if not expected_n:
        return None
    expected_tokens = expected_n.split(" ")
    actual_tokens = normalize_text(actual).split(" ") if normalize_text(actual) else []
    return edit_distance(expected_tokens, actual_tokens) / len(expected_tokens)


def region_recovered(region: Region, page_text: str) -> bool:
    """A region counts as recovered when its squashed text appears in the page text."""
    expected = squashed(region.expected_text)
    if not expected:
        return False
    return expected in squashed(page_text)


def region_recall(regions: list[Region], page_text: str) -> float | None:
    """Fraction of expected text-bearing regions recovered; ``None`` with no ground truth."""
    if not regions:
        return None
    return sum(region_recovered(r, page_text) for r in regions) / len(regions)


def ocr_coverage(regions: list[Region], page_text: str) -> float | None:
    """Character-weighted recovery of expected ink regions.

    Counting page area or rectangles alone cannot certify completeness, so
    coverage is weighted by expected character mass, not region count.
    """
    mass = [(len(squashed(r.expected_text)), region_recovered(r, page_text)) for r in regions]
    total = sum(m for m, _ in mass)
    if total == 0:
        return None
    return sum(m for m, hit in mass if hit) / total


def reading_order_accuracy(regions: list[Region], page_text: str) -> float | None:
    """Fraction of consecutive reading-order pairs recovered in the right order.

    ``None`` when fewer than one comparable pair exists (regions missing or
    not recovered).
    """
    ordered = sorted(regions, key=lambda r: r.reading_order)
    squashed_page = squashed(page_text)
    positions = [squashed_page.find(squashed(r.expected_text)) for r in ordered]
    pairs = [(a, b) for a, b in zip(positions, positions[1:], strict=False) if a >= 0 and b >= 0]
    if not pairs:
        return None
    return sum(a < b for a, b in pairs) / len(pairs)


def serialize_table(table: Table) -> list[str]:
    """Serialize a table grid to a structural tag token stream."""
    tokens: list[str] = []
    for row in range(table.rows):
        tokens.append("<tr>")
        for cell in sorted((c for c in table.cells if c.row == row), key=lambda c: (c.row, c.col)):
            span = f"rs={cell.rowspan} cs={cell.colspan}"
            tokens.append(f"<td {span}>")
            tokens.append("</td>")
        tokens.append("</tr>")
    return tokens


def table_structure_score(expected: Table, actual: Table) -> float:
    """Sequence-TEDS structural similarity in [0, 1] (see module docstring)."""
    expected_tokens = serialize_table(expected)
    actual_tokens = serialize_table(actual)
    if not expected_tokens and not actual_tokens:
        return 1.0
    distance = edit_distance(expected_tokens, actual_tokens)
    return 1.0 - distance / max(len(expected_tokens), len(actual_tokens), 1)


def _cell_map(table: Table) -> dict[tuple[int, int], str]:
    return {(c.row, c.col): normalize_text(c.text) for c in table.cells}


def cell_value_accuracy(expected: Table, actual: Table | None, page_text: str) -> float | None:
    """Fraction of expected cell values recovered.

    With provider structure (``actual``), cells are compared at their grid
    position. Without structure, a cell counts when its normalized value
    appears anywhere in the page text.
    """
    if not expected.cells:
        return None
    if actual is None:
        text = squashed(page_text)
        hits = sum(squashed(c.text) in text for c in expected.cells if c.text.strip())
        nonempty = sum(1 for c in expected.cells if c.text.strip())
        return hits / nonempty if nonempty else None
    actual_cells = _cell_map(actual)
    expected_cells = _cell_map(expected)
    hits = sum(actual_cells.get(pos) == text for pos, text in expected_cells.items())
    return hits / len(expected_cells)


def merged_cell_fidelity(expected: Table, actual: Table | None) -> float | None:
    """Fraction of expected merged cells present with the same span; needs structure."""
    merged = [c for c in expected.cells if c.rowspan > 1 or c.colspan > 1]
    if not merged:
        return None
    if actual is None:
        return None
    actual_spans = {(c.row, c.col): (c.rowspan, c.colspan) for c in actual.cells}
    hits = sum(actual_spans.get((c.row, c.col)) == (c.rowspan, c.colspan) for c in merged)
    return hits / len(merged)


def _numbers_in(text: str) -> list[float]:
    values = []
    for match in _NUM.findall(text.replace("٬", ",")):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def field_accuracy(page: PageAnnotation, page_text: str) -> float | None:
    """Exact/normalized hit rate of critical fields; ``None`` when none are annotated."""
    if not page.critical_fields:
        return None
    text = normalize_text(page_text)
    squashed_page = squashed(page_text)
    numbers = _numbers_in(text)
    hits = 0
    for item in page.critical_fields:
        value = normalize_text(item.value)
        if squashed(item.value) in squashed_page:
            hits += 1
        elif item.numeric:
            expected = _numbers_in(value)
            if expected and any(abs(n - expected[0]) < 1e-6 for n in numbers):
                hits += 1
    return hits / len(page.critical_fields)


@dataclass
class PageScore:
    """Metrics for one page; ``None`` means not applicable (no ground truth)."""

    cer: float | None = None
    wer: float | None = None
    region_recall: float | None = None
    region_precision: float | None = None
    ocr_coverage: float | None = None
    reading_order: float | None = None
    table_structure: float | None = None
    cell_value_accuracy: float | None = None
    merged_cell_fidelity: float | None = None
    field_accuracy: float | None = None
    silent_omission: bool = False
    regions_expected: int = 0
    regions_recovered: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "region_recall": self.region_recall,
            "region_precision": self.region_precision,
            "ocr_coverage": self.ocr_coverage,
            "reading_order": self.reading_order,
            "table_structure": self.table_structure,
            "cell_value_accuracy": self.cell_value_accuracy,
            "merged_cell_fidelity": self.merged_cell_fidelity,
            "field_accuracy": self.field_accuracy,
            "silent_omission": self.silent_omission,
            "regions_expected": self.regions_expected,
            "regions_recovered": self.regions_recovered,
        }


def score_page(
    page: PageAnnotation,
    page_text: str,
    actual_tables: list[Table] | None = None,
    provider_blocks: list[dict] | None = None,
) -> PageScore:
    """Score extracted text/structure for one page against its ground truth."""
    regions = page.text_regions()
    expected_text = normalize_text(" ".join(r.expected_text for r in regions))
    score = PageScore(
        cer=char_error_rate(expected_text, page_text),
        wer=word_error_rate(expected_text, page_text),
        region_recall=region_recall(regions, page_text),
        ocr_coverage=ocr_coverage(regions, page_text),
        reading_order=reading_order_accuracy(regions, page_text),
        field_accuracy=field_accuracy(page, page_text),
        regions_expected=len(regions),
        regions_recovered=sum(region_recovered(r, page_text) for r in regions),
    )
    score.silent_omission = bool(page_text.strip()) and score.regions_recovered < len(regions)
    _score_tables(page, page_text, actual_tables, score)
    if provider_blocks is not None:
        score.region_precision = _block_precision(regions, provider_blocks)
    return score


def _score_tables(
    page: PageAnnotation, page_text: str, actual_tables: list[Table] | None, score: PageScore
) -> None:
    if not page.tables:
        return
    actual_by_id = {t.region_id: t for t in actual_tables or []}
    structure, cells, merged = [], [], []
    for expected in page.tables:
        actual = actual_by_id.get(expected.region_id)
        if actual is not None:
            structure.append(table_structure_score(expected, actual))
        cell_acc = cell_value_accuracy(expected, actual, page_text)
        if cell_acc is not None:
            cells.append(cell_acc)
        fidelity = merged_cell_fidelity(expected, actual)
        if fidelity is not None:
            merged.append(fidelity)
    score.table_structure = sum(structure) / len(structure) if structure else None
    score.cell_value_accuracy = sum(cells) / len(cells) if cells else None
    score.merged_cell_fidelity = sum(merged) / len(merged) if merged else None


def _block_precision(regions: list[Region], blocks: list[dict]) -> float | None:
    """Fraction of provider text blocks overlapping an expected text region."""
    if not blocks:
        return None
    hits = 0
    for block in blocks:
        block_text = normalize_text(str(block.get("text", "")))
        if any(block_text and block_text in normalize_text(r.expected_text) for r in regions):
            hits += 1
    return hits / len(blocks)
