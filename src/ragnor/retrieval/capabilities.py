"""Backend capability declarations and constraint validation.

Storage backends declare their supported features and filter operators.
When a retrieval or storage request specifies mandatory capability
constraints that the target backend does not support, execution **fails
closed** with :class:`CapabilityUnsupportedError` rather than silently
ignoring constraints or returning incomplete/unfiltered results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ragnor.core.exceptions import UnsupportedCapabilityError
from ragnor.core.types import RetrievalError


class CapabilityUnsupportedError(UnsupportedCapabilityError, RetrievalError):
    """Raised when a mandatory capability constraint is not supported by the backend.

    Fail-closed behavior: unsupported security, filter, or transaction
    guarantees reject the request immediately to prevent silent leakage
    or incorrect results.
    """

    default_code = "ERR_CAPABILITY_UNSUPPORTED"


class FilterOperator(StrEnum):
    """Filter operators that a storage backend may support."""

    EQ = "eq"
    IN = "in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    NE = "ne"


class BackendFeature(StrEnum):
    """Functional features that a storage backend may declare."""

    EQUALITY_FILTER = "equality_filter"
    IN_FILTER = "in_filter"
    RANGE_FILTER = "range_filter"
    NATIVE_LEXICAL_SEARCH = "native_lexical_search"
    APPROXIMATE_LEXICAL_SEARCH = "approximate_lexical_search"
    DOCUMENT_FETCH = "document_fetch"
    DOCUMENT_DELETE = "document_delete"
    GENERATION_VISIBILITY = "generation_visibility"
    ISOLATION_SCOPE = "isolation_scope"
    TRANSACTION_GUARANTEE = "transaction_guarantee"


@dataclass(frozen=True)
class StoreCapabilities:
    """Declared capabilities of a vector or lexical storage backend.

    Attributes:
        filter_operators: Set of supported :class:`FilterOperator` values.
        features: Set of supported :class:`BackendFeature` values.
    """

    filter_operators: frozenset[str] = field(default_factory=frozenset)
    features: frozenset[str] = field(default_factory=frozenset)

    def supports_operator(self, operator: str | FilterOperator) -> bool:
        """Return whether ``operator`` is supported."""
        return str(operator) in self.filter_operators

    def supports_feature(self, feature: str | BackendFeature) -> bool:
        """Return whether ``feature`` is supported."""
        return str(feature) in self.features

    def validate_constraints(self, mandatory_constraints: Iterable[str]) -> None:
        """Validate that all mandatory constraints are supported.

        Fails closed: raises :class:`CapabilityUnsupportedError` on the
        first unsupported constraint.

        Args:
            mandatory_constraints: Capability or feature names required
                for safe query execution.

        Raises:
            CapabilityUnsupportedError: If any constraint is unsupported.
        """
        for constraint in mandatory_constraints:
            norm = str(constraint)
            is_supported = (
                norm in self.features
                or norm in self.filter_operators
                or (
                    norm == "transactions"
                    and self.supports_feature(BackendFeature.TRANSACTION_GUARANTEE)
                )
                or (
                    norm == "lexical"
                    and (
                        self.supports_feature(BackendFeature.NATIVE_LEXICAL_SEARCH)
                        or self.supports_feature(BackendFeature.APPROXIMATE_LEXICAL_SEARCH)
                    )
                )
            )
            if not is_supported:
                raise CapabilityUnsupportedError(
                    f"Backend does not support required capability constraint: {constraint!r}. "
                    "Failing closed to prevent unconstrained or leaky query execution."
                )

    def validate_filters(self, filters: Mapping[str, Any]) -> None:
        """Validate that the operators implied by ``filters`` are supported.

        If a filter value is a list/tuple/set, requires :attr:`FilterOperator.IN`.
        Otherwise requires :attr:`FilterOperator.EQ`.

        Raises:
            CapabilityUnsupportedError: If a required operator is unsupported.
        """
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set, frozenset)):
                if not self.supports_operator(FilterOperator.IN):
                    raise CapabilityUnsupportedError(
                        f"Backend does not support IN filters for field {key!r}. "
                        "Failing closed to prevent dropping mandatory filter."
                    )
            elif not self.supports_operator(FilterOperator.EQ):
                raise CapabilityUnsupportedError(
                    f"Backend does not support equality filters for field {key!r}."
                )


# ---------------------------------------------------------------------------
# Standard capability profiles
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_CAPABILITIES = StoreCapabilities(
    filter_operators=frozenset({FilterOperator.EQ, FilterOperator.IN}),
    features=frozenset(
        {
            BackendFeature.EQUALITY_FILTER,
            BackendFeature.IN_FILTER,
            BackendFeature.APPROXIMATE_LEXICAL_SEARCH,
            BackendFeature.DOCUMENT_FETCH,
            BackendFeature.DOCUMENT_DELETE,
            BackendFeature.ISOLATION_SCOPE,
        }
    ),
)

DEFAULT_PGVECTOR_CAPABILITIES = StoreCapabilities(
    filter_operators=frozenset({FilterOperator.EQ, FilterOperator.IN}),
    features=frozenset(
        {
            BackendFeature.EQUALITY_FILTER,
            BackendFeature.IN_FILTER,
            BackendFeature.NATIVE_LEXICAL_SEARCH,
            BackendFeature.DOCUMENT_FETCH,
            BackendFeature.DOCUMENT_DELETE,
            BackendFeature.ISOLATION_SCOPE,
            BackendFeature.TRANSACTION_GUARANTEE,
        }
    ),
)


def get_backend_capabilities(backend: Any) -> StoreCapabilities:
    """Extract explicitly declared capabilities; unknown backends get none.

    Args:
        backend: A vector index, lexical index, document store, or vector store.

    Returns:
        The declared or inferred :class:`StoreCapabilities`.
    """
    if hasattr(backend, "capabilities") and isinstance(backend.capabilities, StoreCapabilities):
        return backend.capabilities

    # Unknown adapters have no verified capabilities. Inspecting class names
    # ("PgVector" / "Memory") or presence of a method cannot establish
    # transaction, tenant isolation or filter semantics. Return none and
    # require explicit capability declarations for guarded operations.
    # Backward-compatible equality-only assumption for legacy local adapters.
    # Never infer isolation, durability or transactional guarantees.
    return StoreCapabilities(filter_operators=frozenset({FilterOperator.EQ}))
