"""Generic outbox drainer.

Runs as its own process so
job processing never blocks an API request: the API
writes a job row in the same transaction as the row it
refers to; this worker atomically claims pending jobs, processes them, and marks
them ``completed``.

The drainer is parameterized by an
:class:`~ragnor.jobs.outbox.EmbeddingJobStore` and a
:class:`~ragnor.jobs.outbox.JobProcessor`, so it works
against any job table. Multiple drainers can run
concurrently: an adapter's atomic claim keeps two workers from
processing the same job, so this scales horizontally
without coordination.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from ragnor.jobs.outbox import (
    EmbeddingJobStore,
    JobProcessor,
    process_pending_jobs,
    reclaim_stale_jobs,
)

logger = logging.getLogger("ragnor.jobs.drainer")


async def drain_forever(  # noqa: PLR0913, PLR0917
    store: EmbeddingJobStore,
    processor: JobProcessor,
    interval: float,
    batch_size: int,
    worker_id: str,
    stop: asyncio.Event | None = None,
) -> None:
    """Process jobs until the stop event is set.

    A failing pass is logged and retried on the next tick rather
    than terminating the worker: a transient processing outage
    should not require a restart.

    Args:
        store: The job store to drain.
        processor: The per-job processor.
        interval: Seconds to sleep when the queue is empty.
        batch_size: Jobs claimed per pass.
        worker_id: Identity stamped into ``locked_by``.
        stop: Optional event that ends the loop when set.
    """
    stop = stop or asyncio.Event()

    logger.info(
        "Job drainer started (interval=%ss, batch=%d, worker_id=%s).",
        interval,
        batch_size,
        worker_id,
    )

    while not stop.is_set():
        try:
            reclaimed = await reclaim_stale_jobs(store)
            if reclaimed:
                logger.info("Reclaimed %d stale job(s).", reclaimed)
            processed = await process_pending_jobs(
                store,
                processor,
                limit=batch_size,
                worker_id=worker_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Drain pass failed (%s); retrying on the next tick.", type(exc).__name__)
            processed = 0

        if processed:
            logger.info("Processed %d job(s).", processed)
            continue

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval)

    logger.info("Job drainer stopped.")


def _install_signal_handlers(stop: asyncio.Event) -> None:
    """Ask the loop to stop on SIGINT/SIGTERM so in-flight work can finish."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)


async def _run(
    store: EmbeddingJobStore,
    processor: JobProcessor,
    interval: float,
    batch_size: int,
    worker_id: str,
) -> None:
    """Entry point: drain until a termination signal arrives."""
    stop = asyncio.Event()
    _install_signal_handlers(stop)
    await drain_forever(store, processor, interval, batch_size, worker_id, stop)


def main(
    store: EmbeddingJobStore,
    processor: JobProcessor,
    interval: float,
    batch_size: int,
    worker_id: str,
) -> int:
    """Console entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(store, processor, interval, batch_size, worker_id))
    return 0
