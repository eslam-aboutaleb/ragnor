"""Deployment qualification harness for host-supplied SDK clients.

The repository test-suite qualifies the bundled PostgreSQL/pgvector and
Qdrant adapters, real service containers in CI, and explicitly synthetic
local protocol endpoints. Those results do not attest arbitrary
host-injected embedding or vector clients: every deployment must qualify
its own clients before production use. This module ships a small,
zero-dependency harness for that purpose. It is pytest-agnostic so a host
can run it from its own test suite, a deployment startup check, or a CI
job without importing Ragnor's test tree.

The harness never raises for a failed check. It records one
:class:`QualificationCheck` per criterion, and
:meth:`QualificationReport.require_passed` raises a sanitized
:class:`QualificationError` naming only the failed check identifiers.
Every check detail passes through ``sanitize_message`` so a provider
payload cannot leak credentials through a report.

Usage::

    report = await qualify_embedding_client(my_client, expected_dimension=1536)
    report.require_passed()
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ragnor.core.exceptions import RagnorError, sanitize_message
from ragnor.embeddings.base import EmbeddingFunction

__all__ = [
    "QualificationCheck",
    "QualificationError",
    "QualificationReport",
    "qualify_client_lifecycle",
    "qualify_embedding_client",
]

_DEFAULT_TIMEOUT = 5.0


class QualificationError(RagnorError):
    """A deployed client failed one or more qualification checks."""

    default_code = "ERR_QUALIFICATION_FAILED"
    default_retryable = False


@dataclass(frozen=True)
class QualificationCheck:
    """Outcome of one qualification criterion.

    ``detail`` is sanitized at construction; it carries diagnostics, never
    credential material.
    """

    name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", sanitize_message(self.detail))


@dataclass(frozen=True)
class QualificationReport:
    """Aggregate result for one qualified client."""

    target: str
    checks: tuple[QualificationCheck, ...]

    @property
    def passed(self) -> bool:
        """True when every recorded check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of the checks that did not pass, in execution order."""
        return tuple(check.name for check in self.checks if not check.passed)

    def require_passed(self) -> None:
        """Raise :class:`QualificationError` when any check failed.

        The exception message names failed checks only; diagnostics stay in
        the per-check ``detail`` fields, which are already sanitized.
        """
        if self.passed:
            return
        raise QualificationError(
            f"{self.target} failed deployment qualification: {', '.join(self.failed_checks)}",
            details={"target": self.target, "failed_checks": list(self.failed_checks)},
        )


async def _run_with_deadline(
    name: str,
    call: Callable[[], Awaitable[object]],
    timeout: float,
) -> tuple[QualificationCheck, object | None]:
    """Run ``call`` under a deadline; a timeout cancels the call."""
    try:
        result = await asyncio.wait_for(call(), timeout)
    except TimeoutError:
        return (
            QualificationCheck(
                name,
                False,
                f"did not complete within the {timeout}s deadline; the call was cancelled",
            ),
            None,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return (
            QualificationCheck(name, False, f"raised {type(exc).__name__}: {exc}"),
            None,
        )
    return QualificationCheck(name, True), result


async def _cancellation_check(
    name: str,
    call: Callable[[], Awaitable[object]],
) -> QualificationCheck:
    """Verify a caller-visible cancellation propagates as CancelledError.

    ``task.cancel()`` reports whether the cancellation was actually
    delivered. A client that completes first has nothing to qualify; a
    client that is cancelled but returns normally (or raises a different
    exception) breaks the startup/cancellation contract the SDK relies on
    for deadline classification and owned-resource cleanup.
    """
    task = asyncio.ensure_future(call())
    await asyncio.sleep(0)
    delivered = task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return QualificationCheck(name, True)
    except Exception as exc:
        return QualificationCheck(
            name,
            False,
            f"cancellation surfaced as {type(exc).__name__} instead of CancelledError: {exc}",
        )
    if delivered:
        return QualificationCheck(
            name,
            False,
            "cancellation was delivered but the client returned normally",
        )
    return QualificationCheck(
        name,
        True,
        "call completed before cancellation could be delivered; nothing to qualify",
    )


async def _cleanup_checks(
    client: object,
    probe: Callable[[], Awaitable[object]],
    timeout: float,
) -> list[QualificationCheck]:
    """Verify the disconnect/cleanup contract of a deployed client."""
    aclose = getattr(client, "aclose", None)
    if not callable(aclose):
        return [
            QualificationCheck(
                "cleanup_contract",
                False,
                "client exposes no aclose(); owned resources cannot be released on "
                "SDK shutdown and the deployment cannot meet the cleanup criterion",
            )
        ]
    first, _ = await _run_with_deadline("cleanup_contract", aclose, timeout)
    if not first.passed:
        return [first]
    try:
        await aclose()
    except Exception as exc:
        return [
            QualificationCheck(
                "cleanup_contract",
                False,
                f"aclose() is not idempotent; the second call raised {type(exc).__name__}: {exc}",
            )
        ]
    checks = [QualificationCheck("cleanup_contract", True, "aclose() completed and is idempotent")]
    try:
        await asyncio.wait_for(probe(), timeout)
    except TimeoutError:
        checks.append(
            QualificationCheck(
                "disconnect_terminal",
                False,
                "a call after aclose() hung past the deadline instead of failing fast",
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        checks.append(QualificationCheck("disconnect_terminal", True))
    else:
        checks.append(
            QualificationCheck(
                "disconnect_terminal",
                False,
                "calls after aclose() still succeed; shutdown does not disconnect the client",
            )
        )
    return checks


async def qualify_embedding_client(
    client: object,
    *,
    expected_dimension: int | None = None,
    startup_timeout: float = _DEFAULT_TIMEOUT,
) -> QualificationReport:
    """Qualify a deployed embedding client against the SDK startup contract.

    Checks, in order: structural conformance to
    :class:`~ragnor.embeddings.base.EmbeddingFunction`, a real call under a
    deadline (startup), batch cardinality/finite-component and optional
    dimension contract, caller-visible cancellation propagation, and the
    cleanup/disconnect contract (``aclose`` presence, idempotence, and
    fail-fast behavior after disconnect).

    Passing this harness qualifies the client integration only; it is not
    model-quality evidence.
    """
    probe_texts = ["ragnor deployment qualification probe"]

    async def probe() -> object:
        return await client(probe_texts)  # type: ignore[operator]

    checks: list[QualificationCheck] = []
    checks.append(
        QualificationCheck(
            "embedding_function_protocol",
            isinstance(client, EmbeddingFunction),
            ""
            if isinstance(client, EmbeddingFunction)
            else "client does not satisfy the EmbeddingFunction protocol "
            "(async __call__, embed_query, embed_documents)",
        )
    )

    startup, vectors = await _run_with_deadline(
        "startup_call_within_deadline", probe, startup_timeout
    )
    checks.append(startup)

    if startup.passed and isinstance(vectors, list):
        checks.append(_batch_contract_check(vectors, expected_dimension))

    checks.append(await _cancellation_check("cancellation_propagates", probe))
    checks.extend(await _cleanup_checks(client, probe, startup_timeout))
    return QualificationReport(target=type(client).__name__, checks=tuple(checks))


def _batch_contract_check(
    vectors: list[object], expected_dimension: int | None
) -> QualificationCheck:
    name = "batch_contract"
    if not vectors or not all(isinstance(vector, list) for vector in vectors):
        return QualificationCheck(name, False, "call did not return a non-empty list of vectors")
    for vector in vectors:
        if not vector or not all(isinstance(c, int | float) and math.isfinite(c) for c in vector):
            return QualificationCheck(name, False, "vectors must be non-empty finite numeric lists")
    if expected_dimension is not None and any(
        len(vector) != expected_dimension for vector in vectors
    ):
        return QualificationCheck(
            name,
            False,
            f"returned vectors do not match expected dimension {expected_dimension}",
        )
    return QualificationCheck(name, True)


async def qualify_client_lifecycle(
    client: object,
    probe: Callable[[], Awaitable[object]],
    *,
    startup_timeout: float = _DEFAULT_TIMEOUT,
) -> QualificationReport:
    """Qualify startup, cancellation and cleanup for any deployed client.

    ``probe`` is a host-supplied callable that exercises the real client
    (for example ``lambda: store.count(filters={})`` for a vector store, or
    a lightweight readiness request for a custom backend). The harness runs
    the probe under a deadline, verifies cancellation propagation, then
    verifies the cleanup/disconnect contract.
    """
    checks: list[QualificationCheck] = []
    startup, _ = await _run_with_deadline("startup_probe_within_deadline", probe, startup_timeout)
    checks.append(startup)
    checks.append(await _cancellation_check("cancellation_propagates", probe))
    checks.extend(await _cleanup_checks(client, probe, startup_timeout))
    return QualificationReport(target=type(client).__name__, checks=tuple(checks))
