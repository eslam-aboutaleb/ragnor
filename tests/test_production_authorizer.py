"""A production KnowledgeBridge requires explicit host authorization."""

from __future__ import annotations

import pytest

from ragnor.evidence.types import ScopeAuthorizationError
from ragnor.tools.bridge import KnowledgeBridge
from ragnor.tools.security import ScopeAuthorizer


class HostRetriever:
    async def retrieve(self, query: str, n_results: int = 5, **filters):
        return []


def test_production_bridge_requires_authorizer() -> None:
    with pytest.raises(ValueError, match="authorizer"):
        KnowledgeBridge(retriever=HostRetriever(), production_mode=True)


@pytest.mark.asyncio
async def test_production_bridge_denies_unmapped_clients_with_host_policy() -> None:
    bridge = KnowledgeBridge(
        retriever=HostRetriever(),
        authorizer=ScopeAuthorizer(default_allow=False),
        production_mode=True,
    )
    with pytest.raises(ScopeAuthorizationError):
        await bridge.search_knowledge(query="refund", scope="tenant-a", client_id="stranger")


def test_existing_nonproduction_bridge_stays_backwards_compatible() -> None:
    bridge = KnowledgeBridge(retriever=HostRetriever())
    assert bridge.authorizer.is_authorized(None, "default", None)


def test_production_rejects_unrestricted_explicit_authorizer() -> None:
    with pytest.raises(ValueError, match="restrictive"):
        KnowledgeBridge(
            retriever=HostRetriever(),
            authorizer=ScopeAuthorizer(),
            production_mode=True,
        )
