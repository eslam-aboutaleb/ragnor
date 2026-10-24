"""Compatibility import for :mod:`ragnor.client.builder`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.client import builder as _implementation

if TYPE_CHECKING:
    from ragnor.client.builder import *  # noqa: F403 - preserve legacy type-checker exports

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
