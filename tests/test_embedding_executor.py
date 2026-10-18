"""Tests for EmbeddingExecutor: retry, concurrency, timeout, and response validation.

Covers:
* Bounded concurrency: semaphore never exceeded.
* Fake-clock retry tests: delays are injected without real sleep.
* Permanent failures are not retried.
* Transient failures are retried up to max_retries.
* Partial responses (fewer vectors than inputs) raise EmbeddingResponseError.
* Non-finite vector values raise EmbeddingResponseError.
* Wrong dimension raises EmbeddingResponseError.
* Exponential backoff is computed correctly.
* Retry-After hint is honoured.
* Cancellation propagates correctly.
* Batching splits inputs correctly.
* Metrics accumulate correctly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from ragnor.embeddings.executor import (
    EmbeddingExecutor,
    EmbeddingPermanentError,
    EmbeddingResponseError,
    EmbeddingTransientError,
    _classify_exception,
    _validate_response,
)
from ragnor.embeddings.profile import EmbeddingProfile


def _make_profile(dimension: int = 3) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider_id="test",
        model_id="test-model",
        dimension=dimension,
    )


def _make_vectors(n: int, dim: int = 3) -> list[list[float]]:
    return [[float(i)] * dim for i in range(n)]


def _sync_fn(vectors: list[list[float]]) -> Callable[[list[str]], Any]:
    """Build a fake embedding callable that always returns ``vectors``."""

    async def _fn(inp: list[str]) -> list[list[float]]:
        return vectors

    return _fn


def _make_executor(  # noqa: PLR0913, PLR0917
    fn: Any,
    profile: EmbeddingProfile | None = None,
    *,
    max_retries: int = 0,
    max_concurrency: int = 4,
    timeout_seconds: float = 0.0,
    base_backoff: float = 0.01,
    max_backoff: float = 1.0,
    jitter: float = 0.0,
    max_batch_size: int = 512,
    sleep: Callable | None = None,
) -> EmbeddingExecutor:
    return EmbeddingExecutor(
        fn=fn,
        profile=profile or _make_profile(),
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        jitter=jitter,
        max_batch_size=max_batch_size,
        sleep=sleep or (lambda _: asyncio.sleep(0)),
    )


class TestBasicEmbedding:
    async def test_empty_input_returns_empty_list(self) -> None:
        called = False

        async def fn(inp: list[str]) -> list[list[float]]:
            nonlocal called
            called = True
            return []

        executor = _make_executor(fn)
        vectors, metrics = await executor.embed([])
        assert vectors == []
        assert metrics.embed_calls == 0
        assert not called

    async def test_successful_embed_returns_vectors(self) -> None:
        texts = ["a", "b", "c"]
        expected = _make_vectors(3)
        fn = _sync_fn(expected)
        executor = _make_executor(fn)
        vectors, metrics = await executor.embed(texts)
        assert vectors == expected
        assert metrics.embed_calls == 1
        assert metrics.cache_misses == 3

    async def test_metrics_record_token_estimate(self) -> None:
        texts = ["hello world", "foo bar baz"]
        fn = _sync_fn(_make_vectors(2))
        executor = _make_executor(fn)
        _, metrics = await executor.embed(texts)
        assert metrics.total_input_tokens_estimate > 0
        assert metrics.is_token_estimate is True


class TestBatching:
    async def test_batch_size_splits_calls(self) -> None:
        texts = ["t0", "t1", "t2", "t3", "t4"]
        calls: list[list[str]] = []

        async def counting_fn(inp: list[str]) -> list[list[float]]:
            calls.append(list(inp))
            return _make_vectors(len(inp))

        executor = _make_executor(counting_fn, max_batch_size=2)
        vectors, metrics = await executor.embed(texts)

        assert len(calls) == 3
        assert calls[0] == ["t0", "t1"]
        assert calls[1] == ["t2", "t3"]
        assert calls[2] == ["t4"]
        assert len(vectors) == 5
        assert metrics.embed_calls == 3

    async def test_token_limit_splits_calls(self) -> None:
        # Each text is ~1 token estimate; limit to 3 estimated tokens per batch.
        texts = ["a", "b", "c", "d"]
        calls: list[list[str]] = []

        async def counting_fn(inp: list[str]) -> list[list[float]]:
            calls.append(list(inp))
            return _make_vectors(len(inp))

        executor = _make_executor(
            counting_fn,
            max_batch_size=512,
            # Token estimate: ceil(1 * 1.3) = 2 per word; so 2 per text.
            # limit=3 means at most 1 text per batch since 2*2=4>3.
        )
        executor._max_input_tokens_estimate = 3
        vectors, _ = await executor.embed(texts)
        assert len(vectors) == 4


class TestResponseValidation:
    async def test_fewer_vectors_than_texts_raises(self) -> None:
        texts = ["a", "b", "c"]
        fn = _sync_fn(_make_vectors(2))  # only 2 vectors for 3 texts
        executor = _make_executor(fn)
        with pytest.raises(EmbeddingResponseError, match="3 inputs"):
            await executor.embed(texts)

    async def test_more_vectors_than_texts_raises(self) -> None:
        texts = ["a", "b"]
        fn = _sync_fn(_make_vectors(3))  # 3 vectors for 2 texts
        executor = _make_executor(fn)
        with pytest.raises(EmbeddingResponseError, match="2 inputs"):
            await executor.embed(texts)

    async def test_wrong_dimension_raises(self) -> None:
        profile = _make_profile(dimension=3)
        fn = _sync_fn([[1.0, 2.0]])  # dim=2, not 3
        executor = _make_executor(fn, profile=profile)
        with pytest.raises(EmbeddingResponseError, match="dimension"):
            await executor.embed(["text"])

    async def test_non_finite_value_raises(self) -> None:
        fn = _sync_fn([[float("nan"), 0.0, 0.0]])
        executor = _make_executor(fn)
        with pytest.raises(EmbeddingResponseError, match="finite"):
            await executor.embed(["text"])

    async def test_inf_value_raises(self) -> None:
        fn = _sync_fn([[float("inf"), 0.0, 0.0]])
        executor = _make_executor(fn)
        with pytest.raises(EmbeddingResponseError, match="finite"):
            await executor.embed(["text"])

    def test_validate_response_standalone(self) -> None:
        profile = _make_profile(dimension=2)
        texts = ["a", "b"]
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        _validate_response(texts, vectors, profile)  # no exception

    def test_validate_response_count_mismatch(self) -> None:
        profile = _make_profile(dimension=2)
        with pytest.raises(EmbeddingResponseError):
            _validate_response(["a", "b"], [[1.0, 2.0]], profile)


class TestRetry:
    async def test_transient_error_retried_up_to_max(self) -> None:
        call_count = 0

        async def flaky_fn(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise Exception("timeout error")  # noqa: TRY002

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        executor = _make_executor(
            flaky_fn,
            max_retries=3,
            base_backoff=0.1,
            jitter=0.0,
            sleep=fake_sleep,
        )
        with pytest.raises(EmbeddingTransientError):
            await executor.embed(["text"])

        # initial + 3 retries = 4 total calls
        assert call_count == 4
        assert len(sleep_calls) == 3

    async def test_permanent_error_is_not_retried(self) -> None:
        call_count = 0

        async def auth_fail_fn(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise Exception("Authentication failed")  # noqa: TRY002

        executor = _make_executor(auth_fail_fn, max_retries=5)
        with pytest.raises(EmbeddingPermanentError):
            await executor.embed(["text"])

        assert call_count == 1  # not retried

    async def test_explicit_transient_error_retried(self) -> None:
        call_count = 0

        async def rate_limited(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise EmbeddingTransientError("rate limit hit")

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        executor = _make_executor(rate_limited, max_retries=2, jitter=0.0, sleep=fake_sleep)
        with pytest.raises(EmbeddingTransientError):
            await executor.embed(["text"])

        assert call_count == 3
        assert len(sleep_calls) == 2

    async def test_retry_after_hint_is_honoured(self) -> None:
        """Retry-After from transient error overrides exponential backoff."""
        call_count = 0

        async def flaky_fn(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise EmbeddingTransientError("rate limit", retry_after=42.0)
            return _make_vectors(len(inp))

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        executor = _make_executor(flaky_fn, max_retries=2, sleep=fake_sleep)
        vectors, metrics = await executor.embed(["text"])

        assert len(vectors) == 1
        assert sleep_calls[0] == pytest.approx(42.0)
        assert metrics.total_retry_count == 1

    async def test_exponential_backoff_grows_correctly(self) -> None:
        """Back-off delays double each attempt (with jitter=0)."""
        call_count = 0

        async def always_fail(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise Exception("timeout error")  # noqa: TRY002

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        executor = _make_executor(
            always_fail,
            max_retries=3,
            base_backoff=1.0,
            max_backoff=100.0,
            jitter=0.0,
            sleep=fake_sleep,
        )
        with pytest.raises(EmbeddingTransientError):
            await executor.embed(["text"])

        # attempt 1 -> delay 1.0; attempt 2 -> 2.0; attempt 3 -> 4.0
        assert sleep_calls == pytest.approx([1.0, 2.0, 4.0])

    async def test_max_backoff_caps_delay(self) -> None:
        call_count = 0

        async def always_fail(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            raise Exception("timeout error")  # noqa: TRY002

        sleep_calls: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        executor = _make_executor(
            always_fail,
            max_retries=5,
            base_backoff=10.0,
            max_backoff=15.0,
            jitter=0.0,
            sleep=fake_sleep,
        )
        with pytest.raises(EmbeddingTransientError):
            await executor.embed(["text"])

        for delay in sleep_calls:
            assert delay <= 15.0

    async def test_success_on_third_attempt_records_retries(self) -> None:
        call_count = 0

        async def eventually_works(inp: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("timeout error")  # noqa: TRY002
            return _make_vectors(len(inp))

        async def fake_sleep(_: float) -> None:
            pass

        executor = _make_executor(eventually_works, max_retries=5, jitter=0.0, sleep=fake_sleep)
        vectors, metrics = await executor.embed(["text"])
        assert len(vectors) == 1
        assert metrics.total_retry_count == 2
        assert metrics.embed_calls == 1


class TestConcurrencyBound:
    async def test_semaphore_is_not_exceeded(self) -> None:
        """Concurrent calls in flight never exceed max_concurrency."""
        max_concurrency = 2
        in_flight: list[int] = []
        peak: list[int] = [0]
        active = [0]

        async def slow_fn(inp: list[str]) -> list[list[float]]:
            active[0] += 1
            in_flight.append(active[0])
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0)
            active[0] -= 1
            return _make_vectors(len(inp))

        executor = _make_executor(
            slow_fn,
            max_concurrency=max_concurrency,
            max_batch_size=1,  # one text per batch to force parallelism
        )
        texts = ["t0", "t1", "t2", "t3", "t4"]
        vectors, _ = await executor.embed(texts)
        assert len(vectors) == 5
        assert peak[0] <= max_concurrency


class TestCancellation:
    async def test_cancelled_task_propagates(self) -> None:
        async def blocking_fn(inp: list[str]) -> list[list[float]]:
            await asyncio.sleep(10)  # effectively blocks
            return _make_vectors(len(inp))

        executor = _make_executor(blocking_fn)
        task = asyncio.create_task(executor.embed(["text"]))
        await asyncio.sleep(0)  # let task start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestClassifyException:
    def test_timeout_is_transient(self) -> None:
        exc = Exception("connection timeout")
        is_transient, retry_after = _classify_exception(exc)
        assert is_transient is True
        assert retry_after is None

    def test_auth_is_permanent(self) -> None:
        exc = Exception("Authentication failed: invalid API key")
        is_transient, _ = _classify_exception(exc)
        assert is_transient is False

    def test_retry_after_extracted(self) -> None:
        exc = Exception("rate limit hit, retry-after: 30.5")
        is_transient, retry_after = _classify_exception(exc)
        assert is_transient is True
        assert retry_after == pytest.approx(30.5)

    def test_unknown_exception_is_transient(self) -> None:
        exc = Exception("something unexpected happened")
        is_transient, _ = _classify_exception(exc)
        assert is_transient is True


class TestConstructorValidation:
    def test_max_batch_size_zero_raises(self) -> None:
        async def dummy(inp: list[str]) -> list[list[float]]:
            return []

        with pytest.raises(ValueError, match="max_batch_size"):
            EmbeddingExecutor(
                fn=dummy,
                profile=_make_profile(),
                max_batch_size=0,
            )

    def test_max_concurrency_zero_raises(self) -> None:
        async def dummy(inp: list[str]) -> list[list[float]]:
            return []

        with pytest.raises(ValueError, match="max_concurrency"):
            EmbeddingExecutor(
                fn=dummy,
                profile=_make_profile(),
                max_concurrency=0,
            )

    def test_negative_max_retries_raises(self) -> None:
        async def dummy(inp: list[str]) -> list[list[float]]:
            return []

        with pytest.raises(ValueError, match="max_retries"):
            EmbeddingExecutor(
                fn=dummy,
                profile=_make_profile(),
                max_retries=-1,
            )

    def test_profile_property(self) -> None:
        async def dummy(inp: list[str]) -> list[list[float]]:
            return []

        profile = _make_profile()
        executor = EmbeddingExecutor(fn=dummy, profile=profile)
        assert executor.profile is profile
