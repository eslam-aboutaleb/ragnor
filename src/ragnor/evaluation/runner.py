"""RAG evaluation runner.

Orchestrates end-to-end evaluation of
a RAG pipeline by running a dataset through retrieval and
answer generation, then computing metrics across all
dimensions.

The retrieval function and the dataset are **injected** by
the caller: ragnor never imports a host application's
retriever or dataset module. The host application wires
its own hybrid retriever and curated dataset in a thin
runner adapter.

Supports three modes:
  - **retrieval-only**: evaluates retrieval quality (Recall@K,
    Precision@K, MRR)
  - **context**: scores the joined retrieved context against
    the gold answer (the legacy behavior under an honest
    name -- it measures whether the retrieved chunks contain
    the answer, not whether a model answers faithfully)
  - **end-to-end**: evaluates both retrieval and answer
    quality (faithfulness, relevance, completeness,
    conciseness) on a real model output produced by an
    injected answer generator

Can be run as a CLI script or imported for use in tests and
the eval API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ragnor.evaluation.answer import (
    AnswerMetrics,
    evaluate_answer_heuristic,
    evaluate_answer_llm,
)
from ragnor.evaluation.dataset import Difficulty, EvalSample
from ragnor.evaluation.retrieval import RetrievalMetrics
from ragnor.core.config import RagSettings

logger = logging.getLogger(__name__)

RetrievalFunction = Callable[[str], Awaitable[list[dict[str, Any]]]]
"""Async retrieval function: ``(query: str) -> list[result dicts]``."""

AnswerFunction = Callable[[str, str], Awaitable[str]]
"""Async answer generator: ``(query: str, context: str) -> answer``."""


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    mode: str = "end-to-end"  # "retrieval-only", "context", or "end-to-end"
    judge: str = "heuristic"  # "heuristic" or "llm"
    n_results: int = 5
    category: str | None = None
    difficulty: Difficulty | None = None


@dataclass
class EvalResult:
    """Complete results from an evaluation run."""

    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    answer_metrics: dict[str, Any] = field(default_factory=dict)
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


async def _run_retrieval_eval(
    retrieval_fn: RetrievalFunction,
    dataset: list[EvalSample],
) -> tuple[RetrievalMetrics, list[dict[str, Any]]]:
    """Run retrieval evaluation using the existing RagEvaluationHarness.

    Maps EvalSample entries to LabeledQuery format and runs them
    through the injected retrieval function.

    Returns:
        Tuple of (aggregated metrics, per-sample details).
    """
    per_sample: list[dict[str, Any]] = []

    for sample in dataset:
        start = time.monotonic()
        try:
            results = await retrieval_fn(sample.query)
            latency_ms = (time.monotonic() - start) * 1000

            # Extract retrieved sections
            retrieved_sections = set()
            for r in results:
                section = r.get("metadata", {}).get("section", "")
                if section:
                    retrieved_sections.add(section)

            expected = set(sample.expected_sections)
            hits = expected & retrieved_sections
            recall = len(hits) / len(expected) if expected else 0.0
            precision = len(hits) / max(len(retrieved_sections), 1)

            # MRR: find rank of first relevant section
            mrr = 0.0
            for rank, r in enumerate(results, 1):
                section = r.get("metadata", {}).get("section", "")
                if section in expected:
                    mrr = 1.0 / rank
                    break

            # Build context for answer evaluation
            context_parts = [r.get("document", "") for r in results]
            context = "\n---\n".join(context_parts)

            per_sample.append(
                {
                    "query": sample.query,
                    "category": sample.category,
                    "difficulty": sample.difficulty.value,
                    "expected_sections": sample.expected_sections,
                    "retrieved_sections": list(retrieved_sections),
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "mrr": round(mrr, 4),
                    "latency_ms": round(latency_ms, 2),
                    "chunks_found": len(results),
                    "context": context,
                }
            )
        except Exception as exc:
            logger.error("Retrieval eval failed for '%s': %s", sample.query, exc)
            per_sample.append(
                {
                    "query": sample.query,
                    "category": sample.category,
                    "difficulty": sample.difficulty.value,
                    "error": str(exc),
                }
            )

    # Compute aggregate retrieval metrics
    valid = [s for s in per_sample if "error" not in s]
    n = len(valid)
    metrics = RetrievalMetrics(
        recall_at_k=sum(s["recall"] for s in valid) / n if n else 0.0,
        precision_at_k=sum(s["precision"] for s in valid) / n if n else 0.0,
        mrr=sum(s["mrr"] for s in valid) / n if n else 0.0,
        avg_latency_ms=sum(s["latency_ms"] for s in valid) / n if n else 0.0,
        total_queries=len(dataset),
        errors=len(per_sample) - n,
    )

    return metrics, per_sample


# ---------------------------------------------------------------------------
# Answer quality evaluation
# ---------------------------------------------------------------------------


async def _run_answer_eval(
    dataset: list[EvalSample],
    per_sample_retrieval: list[dict[str, Any]],
    answer_fn: AnswerFunction | None = None,
    judge: str = "heuristic",
    settings: RagSettings | None = None,
) -> tuple[AnswerMetrics, list[dict[str, Any]]]:
    """Run answer quality evaluation on the RAG pipeline output.

    In ``context`` mode (``answer_fn is None``) the joined
    retrieved context is scored against the gold answer --
    the legacy behavior under an honest name. In
    ``end-to-end`` mode the injected ``answer_fn`` generates
    a real answer from ``(query, context)`` and the judges
    receive the actual generated answer and the actual
    retrieved context as separate inputs.

    Args:
        dataset: The evaluation samples.
        per_sample_retrieval: Retrieval results for each sample (provides context).
        answer_fn: Injected async answer generator
            ``(query, context) -> answer``. When ``None``
            (context mode) the retrieved context itself is
            scored against the gold answer.
        judge: "heuristic" or "llm" evaluation strategy.
        settings: Host settings for the LLM judge (only used when
            ``judge == "llm"``).

    Returns:
        Tuple of (aggregated metrics, per-sample answer details).
    """
    metrics = AnswerMetrics(total_evaluated=len(dataset))
    answer_details: list[dict[str, Any]] = []

    for i, sample in enumerate(dataset):
        try:
            retrieval_data = per_sample_retrieval[i] if i < len(per_sample_retrieval) else {}
            context = retrieval_data.get("context", "")
            chunks_found = retrieval_data.get("chunks_found", 0)

            if answer_fn is not None:
                generated_answer = await answer_fn(sample.query, context)
            else:
                generated_answer = context
                if not generated_answer:
                    generated_answer = "(no context retrieved)"

            if judge == "llm":
                score = await evaluate_answer_llm(
                    query=sample.query,
                    answer=generated_answer,
                    gold_answer=sample.gold_answer,
                    retrieved_context=context,
                    settings=settings,
                )
            else:
                score = evaluate_answer_heuristic(
                    query=sample.query,
                    answer=generated_answer,
                    gold_answer=sample.gold_answer,
                    retrieved_context=context,
                )

            metrics.avg_faithfulness += score.faithfulness
            metrics.avg_relevance += score.relevance
            metrics.avg_completeness += score.completeness
            metrics.avg_conciseness += score.conciseness
            metrics.avg_overall += score.overall

            answer_details.append(
                {
                    "query": sample.query,
                    "gold_answer": sample.gold_answer,
                    "generated_answer": generated_answer[:500],
                    "chunks_found": chunks_found,
                    "faithfulness": score.faithfulness,
                    "relevance": score.relevance,
                    "completeness": score.completeness,
                    "conciseness": score.conciseness,
                    "overall": score.overall,
                    "reasoning": score.reasoning,
                }
            )

        except Exception as exc:
            logger.error("Answer eval failed for '%s': %s", sample.query, exc)
            metrics.errors += 1
            answer_details.append(
                {
                    "query": sample.query,
                    "error": str(exc),
                }
            )

    # Average the metrics
    n = metrics.total_evaluated - metrics.errors
    if n > 0:
        metrics.avg_faithfulness = round(metrics.avg_faithfulness / n, 4)
        metrics.avg_relevance = round(metrics.avg_relevance / n, 4)
        metrics.avg_completeness = round(metrics.avg_completeness / n, 4)
        metrics.avg_conciseness = round(metrics.avg_conciseness / n, 4)
        metrics.avg_overall = round(metrics.avg_overall / n, 4)

    metrics.per_sample = answer_details
    return metrics, answer_details


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


async def run_evaluation(
    config: EvalConfig | None = None,
    retrieval_fn: RetrievalFunction | None = None,
    dataset: list[EvalSample] | None = None,
    settings: RagSettings | None = None,
    answer_fn: AnswerFunction | None = None,
) -> EvalResult:
    """Run a complete RAG evaluation.

    Args:
        config: Evaluation configuration. If None, uses defaults
            (end-to-end mode with heuristic judge).
        retrieval_fn: Injected async retrieval function mapping a
            query string to a list of result dicts. The host
            application supplies its hybrid retriever here.
        dataset: Injected evaluation samples, already filtered to
            the config's category/difficulty.
        settings: Host settings, read by the LLM judge when
            ``config.judge == "llm"``.
        answer_fn: Injected async answer generator
            ``(query, context) -> generated answer``. Required
            when ``config.mode == "end-to-end"``; the host
            application supplies its answer model here.

    Returns:
        EvalResult with all metrics and per-sample details.

    Raises:
        ValueError: If the retrieval function or dataset is
            missing, the mode is unknown, or the mode is
            ``end-to-end`` without an ``answer_fn``.
    """
    if config is None:
        config = EvalConfig()
    if retrieval_fn is None or dataset is None:
        raise ValueError(
            "run_evaluation requires an injected retrieval_fn and dataset; "
            "the host application wires its retriever and dataset in"
        )
    if config.mode not in ("retrieval-only", "context", "end-to-end"):
        raise ValueError(
            f"invalid evaluation mode {config.mode!r}; valid modes are "
            "'retrieval-only', 'context', and 'end-to-end'"
        )
    if config.mode == "end-to-end" and answer_fn is None:
        raise ValueError(
            "mode='end-to-end' requires an answer generator; pass "
            "answer_fn=... (an async (query, context) -> answer "
            "callable) or use one of the other modes: 'retrieval-only', "
            "'context', 'end-to-end'"
        )

    start_time = time.monotonic()
    result = EvalResult(
        timestamp=datetime.now(tz=UTC).isoformat(),
        config={
            "mode": config.mode,
            "judge": config.judge,
            "n_results": config.n_results,
            "category": config.category,
            "difficulty": config.difficulty.value if config.difficulty else None,
        },
    )

    if not dataset:
        result.summary = "No evaluation samples matched the filters."
        return result

    logger.info(
        "Starting RAG evaluation: mode=%s, judge=%s, samples=%d",
        config.mode,
        config.judge,
        len(dataset),
    )

    # Step 1: Retrieval evaluation
    retrieval_metrics, per_sample_retrieval = await _run_retrieval_eval(
        retrieval_fn=retrieval_fn,
        dataset=dataset,
    )
    result.retrieval_metrics = asdict(retrieval_metrics)

    # Step 2: Answer quality evaluation (context and end-to-end modes)
    if config.mode in ("context", "end-to-end"):
        answer_metrics, answer_details = await _run_answer_eval(
            dataset=dataset,
            per_sample_retrieval=per_sample_retrieval,
            answer_fn=answer_fn,
            judge=config.judge,
            settings=settings,
        )
        result.answer_metrics = {
            k: v for k, v in asdict(answer_metrics).items() if k != "per_sample"
        }

        # Merge per-sample results: the retrieved context, the
        # generated answer, and the gold answer stay under their
        # own names so each sample carries the full triple; the
        # metric scores keep the answer_ prefix.
        for i, retrieval_data in enumerate(per_sample_retrieval):
            merged = {**retrieval_data}
            if i < len(answer_details):
                detail = answer_details[i]
                merged["generated_answer"] = detail.get("generated_answer", "")
                merged["gold_answer"] = detail.get("gold_answer", "")
                merged.update(
                    {
                        f"answer_{k}": v
                        for k, v in detail.items()
                        if k
                        not in (
                            "query",
                            "context",
                            "generated_answer",
                            "gold_answer",
                            "chunks_found",
                        )
                    }
                )
            result.per_sample_results.append(merged)
    else:
        result.per_sample_results = per_sample_retrieval

    result.duration_seconds = round(time.monotonic() - start_time, 2)

    # Build summary
    lines = [
        f"RAG Evaluation Complete ({config.mode} mode)",
        f"  Samples: {len(dataset)}",
        f"  Duration: {result.duration_seconds}s",
        "",
        "Retrieval Metrics:",
        f"  Recall@{config.n_results}: {retrieval_metrics.recall_at_k:.4f}",
        f"  Precision@{config.n_results}: {retrieval_metrics.precision_at_k:.4f}",
        f"  MRR: {retrieval_metrics.mrr:.4f}",
        f"  Avg Latency: {retrieval_metrics.avg_latency_ms:.1f}ms",
        f"  Errors: {retrieval_metrics.errors}",
    ]

    if config.mode in ("context", "end-to-end"):
        lines.extend(
            [
                "",
                "Answer Quality Metrics:",
                f"  Faithfulness: {answer_metrics.avg_faithfulness:.4f}",
                f"  Relevance: {answer_metrics.avg_relevance:.4f}",
                f"  Completeness: {answer_metrics.avg_completeness:.4f}",
                f"  Conciseness: {answer_metrics.avg_conciseness:.4f}",
                f"  Overall: {answer_metrics.avg_overall:.4f}",
                f"  Errors: {answer_metrics.errors}",
            ]
        )

    result.summary = "\n".join(lines)
    logger.info("\n%s", result.summary)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    retrieval_fn: RetrievalFunction | None = None,
    dataset: list[EvalSample] | None = None,
    settings: RagSettings | None = None,
    runner: Any = None,
) -> None:
    """CLI entry point for running RAG evaluation.

    Args:
        retrieval_fn: Injected retrieval function (the host
            application wires its retriever here).
        dataset: Injected evaluation dataset.
        settings: Host settings for the LLM judge.
        runner: Async callable invoked as
            ``runner(config, retrieval_fn, dataset, settings)``.
            Defaults to :func:`run_evaluation`; a host
            application passes its own runner to inject
            domain dependencies behind the CLI flags --
            including the ``answer_fn`` answer generator
            that ``mode="end-to-end"`` requires::

                async def host_runner(config, retrieval_fn,
                                    dataset, settings):
                    return await run_evaluation(
                        config, retrieval_fn, dataset, settings,
                        answer_fn=host_answer_fn,
                    )

                main(retrieval_fn, dataset, settings,
                     runner=host_runner)
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--mode",
        choices=["retrieval-only", "context", "end-to-end"],
        default="end-to-end",
        help="Evaluation mode (default: end-to-end)",
    )
    parser.add_argument(
        "--judge",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Answer evaluation strategy (default: heuristic)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of retrieval results (default: 5)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by question category (free-form)",
    )
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in Difficulty],
        default=None,
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for detailed results",
    )

    args = parser.parse_args()

    config = EvalConfig(
        mode=args.mode,
        judge=args.judge,
        n_results=args.n_results,
        category=args.category or None,
        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
    )

    logging.basicConfig(level=logging.INFO)
    run = runner or run_evaluation
    result = asyncio.run(run(config, retrieval_fn, dataset, settings))

    print("\n" + result.summary)

    if args.output:
        output_data = asdict(result)
        # Remove large context fields from per-sample to keep output clean
        for sample in output_data.get("per_sample_results", []):
            sample.pop("context", None)

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
