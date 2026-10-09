"""Evaluation dataset types for RAG systems.

This module is **types only**: the
curated sample data stays in the host application
(domain test data, not library code) and constructs these
``EvalSample`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Difficulty(StrEnum):
    """Difficulty levels for evaluation queries."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvalSample:
    """A single evaluation sample with query, expected retrieval, and gold answer."""

    query: str
    expected_sections: list[str]
    gold_answer: str
    category: str
    difficulty: Difficulty
    metadata: dict[str, Any] = field(default_factory=dict)
