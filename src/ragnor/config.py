"""Compatibility import for :mod:`ragnor.core.config`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import config as _implementation

if TYPE_CHECKING:
    from ragnor.core.config import *  # noqa: F403 - preserve legacy type-checker exports

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
