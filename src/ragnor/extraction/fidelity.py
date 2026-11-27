"""Fidelity metrics over structured results (pure functions).

``teds_like`` is a sequence-edit approximation of TEDS over the tag stream,
not the tree-edit-distance of the original paper.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from ragnor.extraction.types import Table

_WS = re.compile(r"\s+")


def norm(text: str | None) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def _edit(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _tags(table: Table) -> list[str]:
    out: list[str] = []
    for r in range(table.rows):
        out.append("<tr>")
        for c in sorted((c for c in table.cells if c.row == r), key=lambda c: c.col):
            out.append(f"<td rs={c.rowspan} cs={c.colspan}>")
        out.append("</tr>")
    return out


def teds_like(expected: Table, actual: Table) -> float:
    e, a = _tags(expected), _tags(actual)
    return 1.0 - _edit(e, a) / max(len(e), len(a), 1)


def cell_accuracy(expected: Table, actual: Table) -> float:
    """Fraction of expected cells equal (NFKC, whitespace-collapsed) at the same anchor."""
    got = {(c.row, c.col): norm(c.text) for c in actual.cells}
    if not expected.cells:
        return 1.0
    return sum(got.get((c.row, c.col)) == norm(c.text) for c in expected.cells) / len(
        expected.cells
    )


def exact_cell_accuracy(expected: Table, actual: Table) -> float:
    """Like :func:`cell_accuracy` but byte-exact (financial values)."""
    got = {(c.row, c.col): c.text for c in actual.cells}
    if not expected.cells:
        return 1.0
    return sum(got.get((c.row, c.col)) == c.text for c in expected.cells) / len(expected.cells)


def order_accuracy(expected_ids: Sequence[str], actual_ids: Sequence[str]) -> float:
    """Fraction of consecutive expected pairs kept in order."""
    pos = {rid: i for i, rid in enumerate(actual_ids)}
    pairs = [
        (a, b)
        for a, b in zip(expected_ids, expected_ids[1:], strict=False)
        if a in pos and b in pos
    ]
    if not pairs:
        return 1.0 if len(expected_ids) <= 1 else 0.0
    return sum(pos[a] < pos[b] for a, b in pairs) / len(pairs)


def region_recall(expected_ids: Sequence[str], actual_ids: Sequence[str]) -> float:
    return len(set(expected_ids) & set(actual_ids)) / len(expected_ids) if expected_ids else 1.0
