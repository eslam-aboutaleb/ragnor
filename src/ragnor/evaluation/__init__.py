"""RAG evaluation subsystem.

Retrieval metrics, answer-quality judges, the
evaluation runner, and the evaluation dataset
types. All host-specific dependencies (retriever,
dataset, settings) are injected by the caller.
"""

from ragnor.evaluation.answer import (
    AnswerMetrics,
    AnswerScore,
    evaluate_answer_heuristic,
    evaluate_answer_llm,
)
from ragnor.evaluation.dataset import (
    Difficulty,
    EvalSample,
)
from ragnor.evaluation.retrieval import (
    LabeledQuery,
    RagEvaluationHarness,
    RetrievalMetrics,
)
from ragnor.evaluation.runner import (
    AnswerFunction,
    EvalConfig,
    EvalResult,
    RetrievalFunction,
    run_evaluation,
)

__all__ = [
    "AnswerFunction",
    "AnswerMetrics",
    "AnswerScore",
    "Difficulty",
    "EvalConfig",
    "EvalResult",
    "EvalSample",
    "LabeledQuery",
    "RetrievalFunction",
    "RetrievalMetrics",
    "RagEvaluationHarness",
    "evaluate_answer_heuristic",
    "evaluate_answer_llm",
    "run_evaluation",
]
