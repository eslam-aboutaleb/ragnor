"""Compatibility import for :mod:`ragnor.core.config`."""

from __future__ import annotations

# ruff: noqa: PLC0414 - explicit aliases declare typed compatibility exports

import sys
from typing import TYPE_CHECKING

from ragnor.core import config as _implementation

if TYPE_CHECKING:
    from ragnor.core.config import (
        ConfigurationError as ConfigurationError,
        DeploymentMode as DeploymentMode,
        Protocol as Protocol,
        RagSettings as RagSettings,
        RagnorConfig as RagnorConfig,
        SettingsProvider as SettingsProvider,
        StrEnum as StrEnum,
        dataclass as dataclass,
        runtime_checkable as runtime_checkable,
        sanitize_message as sanitize_message,
    )

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
