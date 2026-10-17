"""Convenient public alias for :mod:`ragit.ingestion.connectors`."""

from ragit.ingestion import connectors as _connectors

__all__ = _connectors.__all__
for _name in __all__:
    globals()[_name] = getattr(_connectors, _name)
