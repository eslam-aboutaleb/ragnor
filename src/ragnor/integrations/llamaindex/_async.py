"""Run a coroutine from sync code, with or without a running event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion and return its result.

    With no running loop this is ``asyncio.run``. Inside a running loop
    (a notebook, an async web handler calling a sync LlamaIndex API) the
    coroutine runs on a short-lived worker thread with its own loop, so the
    caller's loop is never blocked re-entrantly and never patched.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
