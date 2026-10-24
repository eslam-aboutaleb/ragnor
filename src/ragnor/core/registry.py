"""Generic provider registry.

Providers (embedders, chunkers, vector stores) register themselves
under a name and are looked up case-insensitively. Host applications
instantiate a registry per process; this module creates no
module-level singletons.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Case-insensitive registry mapping provider names to classes."""

    def __init__(self) -> None:
        self._providers: dict[str, type[T]] = {}
        self._default_name: str | None = None

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers a provider class under ``name``.

        Args:
            name: Provider name. Lookup is case-insensitive.

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def decorator(cls: type[T]) -> type[T]:
            self._providers[name.lower()] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name`` (case-insensitive).

        Raises:
            KeyError: If no provider is registered under ``name``. The
                message lists the registered names.
        """
        try:
            return self._providers[name.lower()]
        except KeyError:
            raise KeyError(
                f"Unknown provider {name!r}. Registered: {', '.join(self.list()) or '(none)'}"
            ) from None

    def list(self) -> list[str]:
        """Return the registered provider names in registration order."""
        return list(self._providers.keys())

    def default(self, name: str) -> None:
        """Set the default provider name.

        Args:
            name: Provider name to use when no explicit provider is given.

        Raises:
            KeyError: If no provider is registered under ``name``.
        """
        if name.lower() not in self._providers:
            raise KeyError(
                f"Unknown provider {name!r}. Registered: {', '.join(self.list()) or '(none)'}"
            )
        self._default_name = name.lower()

    @property
    def default_name(self) -> str | None:
        """The registered default provider name, or ``None`` if unset."""
        return self._default_name
