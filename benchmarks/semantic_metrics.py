"""Critical-field and restricted rational-expression semantics, without eval."""

from __future__ import annotations

import ast
import operator
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from benchmarks.quality_metrics import normalize


def numeric_value(raw: str) -> Decimal:
    value = unicodedata.normalize("NFKC", raw).strip()
    value = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in value)
    value = (
        value.replace("\u066a", "%").replace("\u066b", ".").replace("\u066c", ",").replace(" ", "")
    )
    if "," in value and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", value):
        raise ValueError("Ambiguous numeric grouping; supply a canonical reference value")
    value = value.replace(",", "")
    percentage = value.endswith("%")
    if percentage:
        value = value[:-1]
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric field") from exc
    if not number.is_finite() or abs(number.adjusted()) > 308:
        raise ValueError("Unbounded numeric field")
    return number / 100 if percentage else number


def _expression(raw: str):
    import sympy  # noqa: PLC0415 -- optional benchmark dependency

    if len(raw) > 256:
        raise ValueError("Formula exceeds length bound")
    tree = ast.parse(raw, mode="eval")
    nodes = list(ast.walk(tree))
    names = {node.id for node in nodes if isinstance(node, ast.Name)}
    if (
        len(nodes) > 64
        or len(names) > 2
        or any(not name.isalpha() or len(name) > 16 for name in names)
    ):
        raise ValueError("Formula exceeds complexity bound")

    # Bound total polynomial degree before symbolic construction/simplification.
    def degree(node):
        if isinstance(node, ast.Name):
            return 1
        if isinstance(node, ast.BinOp):
            left, right = degree(node.left), degree(node.right)
            if isinstance(node.op, ast.Pow):
                if (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) is not int
                    or abs(node.right.value) > 3
                ):
                    raise ValueError("Only bounded literal integer powers are supported")
                result = left * abs(node.right.value)
            elif isinstance(node.op, (ast.Mult, ast.Div)):
                result = left + right
            else:
                result = max(left, right)
            if result > 12:
                raise ValueError("Formula exceeds degree bound")
            return result
        return max((degree(child) for child in ast.iter_child_nodes(node)), default=0)

    degree(tree.body)

    def convert(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = numeric_value(str(node.value))
            return sympy.Rational(str(value))
        if isinstance(node, ast.Name):
            return sympy.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            operation = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }.get(type(node.op))
            if operation is not None:
                return operation(left, right)
            if isinstance(node.op, ast.Pow) and right.is_Integer and abs(int(right)) <= 3:
                return left**right
        raise ValueError("Unsupported formula syntax")

    return convert(tree.body)


def formula_equivalent(reference: str, actual: str) -> bool:
    import sympy  # noqa: PLC0415

    left, right = _expression(reference), _expression(actual)
    if left.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo) or right.has(
        sympy.zoo, sympy.nan, sympy.oo, -sympy.oo
    ):
        return False
    return sympy.cancel(left - right) == 0


def score_fields(reference: list[dict], predictions: dict | None) -> dict:
    rows = []
    for field in reference:
        value = (predictions or {}).get(field["id"])
        correct = False
        if value is not None:
            try:
                if field["kind"] in {"number", "chart_value"}:
                    tolerance = numeric_value(field.get("tolerance_abs", "0"))
                    correct = (
                        abs(numeric_value(field["value"]) - numeric_value(str(value))) <= tolerance
                    )
                elif field["kind"] == "formula":
                    correct = formula_equivalent(field["value"], str(value))
                else:
                    correct = normalize(field["value"]) == normalize(str(value))
            except (ValueError, SyntaxError, TypeError):
                correct = False
        rows.append(
            {
                "field_id": field["id"],
                "kind": field["kind"],
                "answered": value is not None,
                "correct": bool(correct),
            }
        )
    return {
        "reference_fields": len(rows),
        "field_task_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None,
        "field_answer_coverage": sum(row["answered"] for row in rows) / len(rows) if rows else None,
        "field_failures": [row for row in rows if not row["correct"]],
        "semantic_capability_declared": predictions is not None,
    }
