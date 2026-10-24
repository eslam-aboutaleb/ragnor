"""Compatibility import for :mod:`ragnor.client.sync_client`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ragnor.client import sync_client as _implementation

if TYPE_CHECKING:
    from ragnor.client.sync_client import *  # noqa: F403 - preserve legacy type-checker exports

# Keep module identity, private helpers and patch targets compatible too.
sys.modules[__name__] = _implementation
