"""Unit tests for ragit.registry."""

from __future__ import annotations

import pytest

from ragit.registry import Registry


def test_register_and_get_is_case_insensitive() -> None:
    registry: Registry[object] = Registry()

    @registry.register("litellm")
    class LiteLLMProvider:
        pass

    assert registry.get("litellm") is LiteLLMProvider
    assert registry.get("LiteLLM") is LiteLLMProvider
    assert registry.get("LITELLM") is LiteLLMProvider


def test_register_returns_the_class_unchanged() -> None:
    registry: Registry[object] = Registry()

    @registry.register("alpha")
    class Alpha:
        pass

    assert registry.get("alpha") is Alpha


def test_list_returns_registered_names_in_order() -> None:
    registry: Registry[object] = Registry()

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    assert registry.list() == ["alpha", "beta"]


def test_get_unknown_name_raises_key_error_listing_registered() -> None:
    registry: Registry[object] = Registry()

    @registry.register("alpha")
    class Alpha:
        pass

    with pytest.raises(KeyError, match="alpha"):
        registry.get("nope")


def test_get_from_empty_registry_raises_key_error() -> None:
    registry: Registry[object] = Registry()
    with pytest.raises(KeyError, match="none"):
        registry.get("nope")


def test_default_sets_the_default_provider() -> None:
    registry: Registry[object] = Registry()

    @registry.register("alpha")
    class Alpha:
        pass

    registry.default("ALPHA")
    assert registry.default_name == "alpha"


def test_default_unknown_name_raises_key_error() -> None:
    registry: Registry[object] = Registry()
    with pytest.raises(KeyError, match="nope"):
        registry.default("nope")
