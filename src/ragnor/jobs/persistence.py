"""ORM-independent factory contracts for durable job persistence."""

from __future__ import annotations

from typing import Protocol

from ragnor.db.ports import SessionProvider
from ragnor.jobs.outbox import EmbeddingJobStore


class GenerationReservations(Protocol):
    """Durably bind a scoped generation to one immutable content digest."""

    scope: str

    async def reserve_generation(self, generation_key: str, content_digest: str) -> None:
        """Allow identical retries and reject changed generation content."""
        raise NotImplementedError


class JobPersistenceFactory(SessionProvider, Protocol):
    """One host database binding from which SDK persistence ports are built."""

    def job_store(
        self, *, scope: str, lease_seconds: float = 300.0, require_publication: bool = False
    ) -> EmbeddingJobStore:
        """Return a scoped durable outbox using the host's configured database."""
        raise NotImplementedError

    def reservations(self, *, scope: str) -> GenerationReservations:
        """Return scoped immutable-generation reservations on that database."""
        raise NotImplementedError
