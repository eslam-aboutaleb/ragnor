"""Compatibility import for :mod:`ragnor.client.sync_client`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.client import sync_client as _implementation

if TYPE_CHECKING:
    from ragnor.client.sync_client import (
        Any as Any,
        AsyncRagnorBuilder as AsyncRagnorBuilder,
        AsyncRagnorOperations as AsyncRagnorOperations,
        Awaitable as Awaitable,
        Callable as Callable,
        ConfigurationError as ConfigurationError,
        Coroutine as Coroutine,
        OperationTimeoutError as OperationTimeoutError,
        Protocol as Protocol,
        RagnorError as RagnorError,
        SyncRagnor as SyncRagnor,
        T as T,
        TypeVar as TypeVar,
        asyncio as asyncio,
        runtime_checkable as runtime_checkable,
        suppress as suppress,
        threading as threading,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
