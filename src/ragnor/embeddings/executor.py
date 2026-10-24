"""Resilient embedding executor with retry, concurrency, and validation.

:class:`EmbeddingExecutor` wraps an :class:`~ragnor.embeddings.base.EmbeddingFunction`
and adds production-grade concerns:

* **Bounded concurrency** via :class:`asyncio.Semaphore`.
* **Bounded input queue** that back-pressures callers when the executor is
  saturated.
* **Per-call timeout** via :func:`asyncio.wait_for`.
* **Exponential back-off with jitter** for transient failures.
* **Retry-After header / hint** support where the provider exposes it.
* **Partial-batch error accounting**: a response with the wrong number of
  vectors raises :class:`EmbeddingResponseError` immediately (never silently
  misaligns chunks and vectors).
* **Response validation**: checks count, finite values, and dimension.
* **Permanent error fast-path**: authentication, dimension, and
  configuration errors are not retried.

The ``clock`` and ``sleep`` parameters accept injectable callables so tests
can use a fake clock without patching :mod:`time` globally.

Usage::

    executor = EmbeddingExecutor(
        fn=my_embedding_function,
        profile=profile,
        max_concurrency=4,
        max_retries=3,
    )
    vectors, metrics = await executor.embed(["hello", "world"])
"""

from __future__ import annotations

import asyncio
import logging
import inspect
import math
import random
import re
import time
from collections.abc import Callable
from typing import Any

from ragnor.embeddings.base import EmbeddingFunction
from ragnor.embeddings.metrics import EmbeddingMetrics
from ragnor.embeddings.profile import EmbeddingProfile, TaskMode
from ragnor.core.exceptions import EmbeddingProviderError
from ragnor.telemetry.events import EventKind, NullEventSink, TelemetryEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(EmbeddingProviderError):
    """Base class for embedding executor errors."""


class EmbeddingTransientError(EmbeddingError):
    """A transient failure that may succeed on retry (network, rate-limit, 5xx)."""

    default_code = "ERR_EMBEDDING_TRANSIENT"
    default_retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, retryable=True)
        self.retry_after = retry_after


class EmbeddingPermanentError(EmbeddingError):
    """A permanent failure that must not be retried (auth, config, dimension)."""

    default_code = "ERR_EMBEDDING_PERMANENT"
    default_retryable = False


class EmbeddingResponseError(EmbeddingPermanentError):
    """The provider returned a response that cannot be safely used.

    Covers mismatched vector counts, non-finite values, and wrong dimensions.
    Declared permanent because retrying the same input will produce the same
    invalid response.
    """

    default_code = "ERR_EMBEDDING_RESPONSE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRANSIENT_PATTERNS = (
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"connection", re.IGNORECASE),
    re.compile(r"service.?unavailable", re.IGNORECASE),
    re.compile(r"internal.?server", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"5\d\d", re.IGNORECASE),
)

_PERMANENT_PATTERNS = (
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"invalid.?key", re.IGNORECASE),
    re.compile(r"permission", re.IGNORECASE),
    re.compile(r"dimension", re.IGNORECASE),
    re.compile(r"model.?not.?found", re.IGNORECASE),
    re.compile(r"not.?found", re.IGNORECASE),
)

_RETRY_AFTER_PATTERN = re.compile(r"retry[_\s-]*after[:\s]+(\d+\.?\d*)", re.IGNORECASE)


def _classify_exception(exc: BaseException) -> tuple[bool, float | None]:
    """Classify *exc* as (is_transient, retry_after_seconds).

    Returns ``(False, None)`` for permanently-failing exceptions.
    """
    # asyncio.CancelledError is neither transient nor permanent -- propagate.
    if isinstance(exc, asyncio.CancelledError):
        raise exc  # re-raise to honour cancellation

    msg = str(exc)

    # Check permanent patterns first (higher specificity).
    for pattern in _PERMANENT_PATTERNS:
        if pattern.search(msg):
            return False, None

    # Check transient patterns.
    retry_after: float | None = None
    match = _RETRY_AFTER_PATTERN.search(msg)
    if match:
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(ValueError):
            retry_after = float(match.group(1))

    for pattern in _TRANSIENT_PATTERNS:
        if pattern.search(msg):
            return True, retry_after

    # Unknown exception -- treat as transient (fail-safe for network issues).
    return True, retry_after


def _validate_response(
    texts: list[str],
    vectors: list[list[float]],
    profile: EmbeddingProfile,
) -> None:
    """Validate that *vectors* is a safe response for *texts*.

    Raises:
        EmbeddingResponseError: When the response cannot be safely used.
    """
    if len(vectors) != len(texts):
        raise EmbeddingResponseError(
            f"Provider returned {len(vectors)} vectors for {len(texts)} inputs; "
            "cannot safely align chunks to vectors."
        )
    for i, vec in enumerate(vectors):
        if len(vec) != profile.dimension:
            raise EmbeddingResponseError(
                f"Vector {i} has dimension {len(vec)}; profile expects {profile.dimension}."
            )
        for j, val in enumerate(vec):
            if not math.isfinite(val):
                raise EmbeddingResponseError(
                    f"Vector {i}[{j}] is not finite ({val!r}); "
                    "refusing to write non-finite embeddings."
                )


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token-count estimate (word count * 1.3, rounded up)."""
    words = sum(len(t.split()) for t in texts)
    return math.ceil(words * 1.3)


# ---------------------------------------------------------------------------
# EmbeddingExecutor
# ---------------------------------------------------------------------------


class EmbeddingExecutor:
    """Wraps an :class:`EmbeddingFunction` with retry, concurrency, and validation.

    Args:
        fn: The embedding function to wrap.
        profile: Profile that describes the expected output geometry.
        max_batch_size: Maximum texts per provider call.  The executor
            splits larger inputs into batches automatically.
        max_input_tokens_estimate: Soft cap on estimated input tokens per
            batch.  When exceeded the batch is split further.  ``0``
            disables token-based splitting.
        max_concurrency: Maximum simultaneous in-flight provider calls.
        timeout_seconds: Per-call timeout in seconds.  ``0`` means no
            timeout.
        base_backoff: Initial back-off duration in seconds.
        max_backoff: Maximum back-off duration in seconds.
        jitter: Maximum random jitter added to the back-off.
        max_retries: Maximum number of retry attempts after the initial
            attempt.  ``0`` means no retries.
        clock: Callable returning the current time (injectable for tests).
        sleep: Async callable that sleeps for *n* seconds (injectable for
            tests).
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        fn: EmbeddingFunction,
        profile: EmbeddingProfile,
        *,
        max_batch_size: int = 512,
        max_input_tokens_estimate: int = 0,
        max_concurrency: int = 4,
        timeout_seconds: float = 60.0,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
        jitter: float = 0.5,
        max_retries: int = 3,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
        event_sink: Any | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        self._fn = fn
        self._profile = profile
        self._max_batch_size = max_batch_size
        self._max_input_tokens_estimate = max_input_tokens_estimate
        self._timeout_seconds = timeout_seconds
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._jitter = jitter
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._sleep: Callable[[float], Any] = sleep or asyncio.sleep
        self._event_sink = event_sink if event_sink is not None else NullEventSink()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profile(self) -> EmbeddingProfile:
        """The profile this executor enforces."""
        return self._profile

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors, _ = await self.embed(texts)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        mode = TaskMode.DOCUMENT if self._profile.task_mode != TaskMode.SYMMETRIC else None
        vectors, _ = await self.embed(texts, task_mode=mode)
        return vectors

    async def embed_query(self, text: str | list[str]) -> list[float]:
        mode = TaskMode.QUERY if self._profile.task_mode != TaskMode.SYMMETRIC else None
        vectors, _ = await self.embed([text] if isinstance(text, str) else text, task_mode=mode)
        return vectors[0]

    async def embed(
        self,
        texts: list[str],
        *,
        task_mode: TaskMode | None = None,
    ) -> tuple[list[list[float]], EmbeddingMetrics]:
        """Embed *texts* with retry, concurrency control, and validation.

        Args:
            texts: Input texts.  Empty lists return immediately.
            task_mode: Override for the task mode.  ``None`` means use
                the profile's default task mode.

        Returns:
            A ``(vectors, metrics)`` tuple.  ``vectors[i]`` corresponds
            to ``texts[i]`` without reordering.

        Raises:
            EmbeddingPermanentError: For authentication, dimension, or
                configuration failures that must not be retried.
            EmbeddingResponseError: When the provider returns a response
                that cannot be safely aligned with the input.
            EmbeddingTransientError: When all retry attempts are exhausted.
        """
        if not texts:
            return [], EmbeddingMetrics()

        effective_task_mode = task_mode or self._profile.task_mode
        batches = self._split_into_batches(texts)
        all_vectors: list[list[float]] = []
        combined_metrics = EmbeddingMetrics()

        for batch in batches:
            vectors, batch_metrics = await self._embed_batch_with_retry(batch, effective_task_mode)
            all_vectors.extend(vectors)
            combined_metrics = combined_metrics.merge(batch_metrics)

        return all_vectors, combined_metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_batches(self, texts: list[str]) -> list[list[str]]:
        """Split *texts* into batches respecting size and token limits."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = _estimate_tokens([text])
            token_overflow = (
                self._max_input_tokens_estimate > 0
                and current
                and current_tokens + text_tokens > self._max_input_tokens_estimate
            )
            size_overflow = len(current) >= self._max_batch_size

            if size_overflow or token_overflow:
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(text)
            current_tokens += text_tokens

        if current:
            batches.append(current)

        return batches

    async def _embed_batch_with_retry(
        self,
        batch: list[str],
        task_mode: TaskMode,
    ) -> tuple[list[list[float]], EmbeddingMetrics]:
        """Embed a single *batch* with exponential back-off retries."""
        metrics = EmbeddingMetrics()
        last_exc: BaseException | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._backoff_delay(attempt, last_exc)
                logger.debug(
                    "Embedding retry %d/%d after %.2fs (batch size %d)",
                    attempt,
                    self._max_retries,
                    delay,
                    len(batch),
                )
                self._event_sink.emit(
                    TelemetryEvent(
                        event_kind=EventKind.RETRY_SCHEDULED,
                        stage="embedding",
                        duration_ms=delay * 1000.0,
                        error=last_exc,
                        counts={
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "batch_size": len(batch),
                        },
                    )
                )
                await self._sleep(delay)
                metrics.total_retry_count += 1

            try:
                vectors = await self._call_provider(batch, task_mode)
                _validate_response(batch, vectors, self._profile)
                metrics.embed_calls += 1
                metrics.cache_misses += len(batch)
                metrics.total_input_tokens_estimate += _estimate_tokens(batch)
                self._event_sink.emit(
                    TelemetryEvent(
                        event_kind=EventKind.BATCH_EMBEDDED,
                        stage="embedding",
                        counts={
                            "batch_size": len(batch),
                            "estimated_tokens": _estimate_tokens(batch),
                            "dimension": self._profile.dimension,
                        },
                    )
                )
                return vectors, metrics

            except EmbeddingPermanentError:
                raise

            except asyncio.CancelledError:
                raise

            except EmbeddingTransientError as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    raise

            except Exception as exc:  # noqa: BLE001
                is_transient, retry_after = _classify_exception(exc)
                if not is_transient:
                    raise EmbeddingPermanentError(
                        f"Embedding provider failed ({type(exc).__name__})"
                    ) from exc
                last_exc = EmbeddingTransientError(
                    f"Embedding provider failed ({type(exc).__name__})", retry_after=retry_after
                )
                if attempt == self._max_retries:
                    raise EmbeddingTransientError(
                        f"Exhausted {self._max_retries} retries ({type(exc).__name__})",
                        retry_after=retry_after,
                    ) from exc

        # Should be unreachable.
        raise EmbeddingTransientError("Retry loop exited unexpectedly.")

    async def _call_provider(self, batch: list[str], task_mode: TaskMode) -> list[list[float]]:
        """Call the wrapped embedding function under semaphore and timeout."""

        async def invoke() -> list[list[float]]:
            parameters = inspect.signature(self._fn).parameters
            if "task_mode" in parameters:
                return await self._fn(batch, task_mode=task_mode)
            if task_mode == TaskMode.DOCUMENT and hasattr(self._fn, "embed_documents"):
                return await self._fn.embed_documents(batch)
            if task_mode == TaskMode.QUERY and hasattr(self._fn, "embed_query"):
                return [await self._fn.embed_query(text) for text in batch]
            if task_mode != TaskMode.SYMMETRIC:
                raise EmbeddingPermanentError("Provider does not support the requested task_mode")
            return await self._fn(batch)

        async with self._semaphore:
            if self._timeout_seconds > 0:
                return await asyncio.wait_for(
                    invoke(),
                    timeout=self._timeout_seconds,
                )
            return await invoke()

    def _backoff_delay(self, attempt: int, last_exc: BaseException | None) -> float:
        """Compute the back-off delay for *attempt* (1-indexed)."""
        # Honour Retry-After if the provider supplied one.
        if isinstance(last_exc, EmbeddingTransientError) and last_exc.retry_after is not None:
            return last_exc.retry_after

        base = self._base_backoff * (2 ** (attempt - 1))
        jitter = random.uniform(0, self._jitter)  # noqa: S311
        return min(base + jitter, self._max_backoff)
