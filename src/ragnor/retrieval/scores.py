"""Explicit score semantics for retrieval results.

Distance, similarity, Reciprocal Rank Fusion (RRF), and reranker
scores represent fundamentally different concepts with different
scales, ordering, and distributions:

* **Distance** (e.g. L2, cosine distance): lower is closer/better.
  Distance values are unbounded or in [0, 2]; they are metric distances,
  NOT probabilities.
* **Similarity** (e.g. cosine similarity, dot product): higher is
  closer/better. Dot products are unbounded; cosine similarities are
  in [-1, 1] or [0, 1]. They are NOT calibrated probabilities.
* **RRF** (Reciprocal Rank Fusion score): higher rank is better.
  Computed as ``sum(1 / (k + rank))``, typically in [0, 0.05]. It is a
  rank-order statistic reflecting reciprocal ranking, NOT a probability.
* **Lexical** (e.g. BM25, token overlap): higher is better. Unbounded
  term match score, NOT a probability.
* **Reranker** (e.g. cross-encoder logit, relevance score): higher is
  better. Model-specific and uncalibrated unless explicitly converted
  with a calibrated sigmoid.

These scores are **not interchangeable confidence probabilities**.
Comparing or thresholding across score kinds without explicit conversion
will produce invalid rankings or false confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScoreKind(StrEnum):
    """Semantic classification of a retrieval score."""

    DISTANCE = "distance"
    SIMILARITY = "similarity"
    RRF = "rrf"
    LEXICAL = "lexical"
    RERANKER = "reranker"


@dataclass(frozen=True)
class RawScore:
    """An explicit retrieval score with its semantic kind and ordering.

    Attributes:
        value: Numeric score value.
        kind: Semantic kind of the score (distance, similarity, RRF, etc.).
        metric: Optional metric or model name (e.g. ``"cosine"``,
            ``"l2"``, ``"cohere/rerank-v3.5"``).
        lower_is_better: When ``True``, smaller values indicate closer /
            better matches (e.g. distances). When ``False``, larger values
            indicate better matches (e.g. similarities, RRF, reranker).
    """

    value: float
    kind: ScoreKind
    metric: str | None = None
    lower_is_better: bool = False

    @classmethod
    def distance(cls, value: float, metric: str = "l2") -> RawScore:
        """Create a distance score where lower is better."""
        return cls(value=value, kind=ScoreKind.DISTANCE, metric=metric, lower_is_better=True)

    @classmethod
    def similarity(cls, value: float, metric: str = "cosine") -> RawScore:
        """Create a similarity score where higher is better."""
        return cls(value=value, kind=ScoreKind.SIMILARITY, metric=metric, lower_is_better=False)

    @classmethod
    def rrf(cls, value: float) -> RawScore:
        """Create an RRF rank score where higher is better."""
        return cls(value=value, kind=ScoreKind.RRF, metric="rrf", lower_is_better=False)

    @classmethod
    def lexical(cls, value: float, metric: str = "token_overlap") -> RawScore:
        """Create a lexical score where higher is better."""
        return cls(value=value, kind=ScoreKind.LEXICAL, metric=metric, lower_is_better=False)

    @classmethod
    def reranker(cls, value: float, model: str | None = None) -> RawScore:
        """Create a reranker score where higher is better."""
        return cls(value=value, kind=ScoreKind.RERANKER, metric=model, lower_is_better=False)
