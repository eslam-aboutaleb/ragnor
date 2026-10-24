"""Answer quality evaluation for RAG systems.

Provides two evaluation strategies:

1. **LLM-as-Judge**: uses a configured LLM (via LiteLLM) to
   score answer quality on multiple dimensions (faithfulness,
   relevance, completeness, conciseness).
2. **Heuristic scoring**: fast, deterministic keyword/overlap
   scoring that works offline without LLM API calls.

Both strategies produce standardized ``AnswerScore`` results
that can be aggregated across the full evaluation dataset.
The LLM judge reads its model from a :class:`RagSettings`
passed in by the caller — ragnor never imports the host
application's config module.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ragnor.core.config import RagSettings

logger = logging.getLogger(__name__)


@dataclass
class AnswerScore:
    """Scores for a single answer evaluation.

    All scores are floats in [0.0, 1.0].
    """

    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    overall: float = 0.0
    reasoning: str = ""


@dataclass
class AnswerMetrics:
    """Aggregated answer quality metrics across an evaluation run."""

    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_conciseness: float = 0.0
    avg_overall: float = 0.0
    total_evaluated: int = 0
    errors: int = 0
    per_sample: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic (offline) evaluator
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard-like token overlap between two texts."""
    tokens_a = set(_normalize_text(text_a).split())
    tokens_b = set(_normalize_text(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _key_fact_recall(
    answer: str,
    gold_answer: str,
    domain_terms: set[str] | None = None,
) -> float:
    """Check what fraction of gold answer's key facts appear in the answer.

    Key facts are the numbers in the gold answer plus any
    host-supplied domain vocabulary (``domain_terms``) that
    appears in it.
    """
    # Extract key facts: numbers, and optional domain terms
    number_pattern = re.compile(r"\b\d[\d,]*\b")

    key_facts = set(number_pattern.findall(gold_answer))

    if domain_terms:
        gold_lower = gold_answer.lower()
        key_facts.update(term for term in domain_terms if term in gold_lower)

    if not key_facts:
        return _token_overlap(answer, gold_answer)

    answer_lower = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in answer_lower or fact in answer)
    return hits / len(key_facts)


def evaluate_answer_heuristic(
    query: str,
    answer: str,
    gold_answer: str,
    retrieved_context: str = "",
    domain_terms: set[str] | None = None,
) -> AnswerScore:
    """Score an answer using deterministic heuristics (no LLM needed).

    Args:
        query: The original user question.
        answer: The generated answer to evaluate.
        gold_answer: The reference/gold-standard answer.
        retrieved_context: The retrieved context chunks (for faithfulness check).
        domain_terms: Host-specific vocabulary whose presence in the
            gold answer counts as a key fact. Defaults to None
            (numbers only), so the judge is domain-neutral out of the box.

    Returns:
        AnswerScore with heuristic-based scores.
    """
    if not answer or not answer.strip():
        return AnswerScore(reasoning="Empty answer")

    # Faithfulness: how much of the answer is grounded in context
    if retrieved_context:
        faithfulness = _token_overlap(answer, retrieved_context)
    else:
        faithfulness = _token_overlap(answer, gold_answer)

    # Relevance: overlap between answer and question terms
    relevance = min(_token_overlap(answer, query) * 3.0, 1.0)

    # Completeness: key fact recall from gold answer
    completeness = _key_fact_recall(answer, gold_answer, domain_terms)

    # Conciseness: penalize excessively long answers
    answer_len = len(answer.split())
    gold_len = max(len(gold_answer.split()), 1)
    ratio = answer_len / gold_len
    if ratio <= 2.0:
        conciseness = 1.0
    elif ratio <= 4.0:
        conciseness = max(0.3, 1.0 - (ratio - 2.0) * 0.35)
    else:
        conciseness = 0.2

    overall = 0.30 * faithfulness + 0.25 * relevance + 0.30 * completeness + 0.15 * conciseness

    return AnswerScore(
        faithfulness=round(faithfulness, 4),
        relevance=round(relevance, 4),
        completeness=round(completeness, 4),
        conciseness=round(conciseness, 4),
        overall=round(overall, 4),
        reasoning="heuristic",
    )


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluator
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate the quality of a generated answer.

## Evaluation Criteria

Score each dimension from 0.0 to 1.0:

1. **Faithfulness** (0-1): Is the answer factually grounded in the retrieved context?
   Does it avoid hallucinating information not present in the context?
   - 1.0 = Fully faithful, all claims supported by context
   - 0.0 = Completely hallucinated or contradicts context

2. **Relevance** (0-1): Does the answer directly address the user's question?
   - 1.0 = Directly and precisely answers the question
   - 0.0 = Completely irrelevant

3. **Completeness** (0-1): Does the answer cover all key points from the gold answer?
   - 1.0 = Covers all key information
   - 0.0 = Misses all key information

4. **Conciseness** (0-1): Is the answer appropriately concise without unnecessary verbosity?
   - 1.0 = Perfectly concise
   - 0.0 = Extremely verbose or padded

## Input

**User Question:** {query}

**Retrieved Context:**
{context}

**Generated Answer:** {answer}

**Gold/Reference Answer:** {gold_answer}

## Output Format

Respond with ONLY a JSON object (no markdown, no extra text):
{{
  "faithfulness": <float>,
  "relevance": <float>,
  "completeness": <float>,
  "conciseness": <float>,
  "reasoning": "<brief explanation>"
}}
"""


async def evaluate_answer_llm(
    query: str,
    answer: str,
    gold_answer: str,
    retrieved_context: str = "",
    settings: RagSettings | None = None,
) -> AnswerScore:
    """Score an answer using the configured LLM as a judge.

    Args:
        query: The original user question.
        answer: The generated answer to evaluate.
        gold_answer: The reference/gold-standard answer.
        retrieved_context: The retrieved context chunks.
        settings: Host settings providing ``llm_model``; when the setting
            is absent the judge cannot run and the heuristic fallback applies.

    Returns:
        AnswerScore with LLM-judged scores, or the heuristic
        fallback when the judge fails.
    """
    import litellm  # noqa: PLC0415

    prompt = _JUDGE_PROMPT.format(
        query=query,
        context=retrieved_context or "(no context provided)",
        answer=answer,
        gold_answer=gold_answer,
    )

    try:
        if settings is None:
            raise ValueError("LLM judge requires RagSettings with llm_model")

        llm_model = getattr(settings, "llm_model", None)
        if llm_model is None:
            raise ValueError("LLM judge requires settings with llm_model")

        response = await litellm.acompletion(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        scores = json.loads(json_match.group()) if json_match else json.loads(content)

        faithfulness = float(scores.get("faithfulness", 0))
        relevance = float(scores.get("relevance", 0))
        completeness = float(scores.get("completeness", 0))
        conciseness = float(scores.get("conciseness", 0))
        reasoning = scores.get("reasoning", "")

        overall = 0.30 * faithfulness + 0.25 * relevance + 0.30 * completeness + 0.15 * conciseness

        return AnswerScore(
            faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4),
            completeness=round(completeness, 4),
            conciseness=round(conciseness, 4),
            overall=round(overall, 4),
            reasoning=reasoning,
        )

    except Exception as exc:
        logger.error("LLM judge evaluation failed: %s", exc)
        score = evaluate_answer_heuristic(
            query=query,
            answer=answer,
            gold_answer=gold_answer,
            retrieved_context=retrieved_context,
        )
        score.reasoning = f"llm_judge_unavailable ({type(exc).__name__}): {score.reasoning}"
        return score
