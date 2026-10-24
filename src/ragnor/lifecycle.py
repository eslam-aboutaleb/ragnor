"""Compatibility import for :mod:`ragnor.core.lifecycle`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.core import lifecycle as _implementation

if TYPE_CHECKING:
    from ragnor.core.lifecycle import *  # noqa: F403 - preserve legacy type-checker exports

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
