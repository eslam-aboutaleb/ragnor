"""Optional PostgreSQL CAS for complete immutable vector-generation content."""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class PostgresGenerationReservations:
    """Durable tenant-scoped reservations, committed before external writes.

    Apply migration 003 before use. The host owns the session pool. Never
    delete a reservation while its generation is readable or can be replayed.
    A failed external write deliberately leaves its reservation intact.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, scope: str) -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be nonempty")
        self.scope = scope
        self._sessions = sessions

    async def reserve_generation(self, generation_key: str, content_digest: str) -> None:
        if not isinstance(generation_key, str) or not generation_key.strip():
            raise ValueError("generation_key must be nonempty")
        if not isinstance(content_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", content_digest):
            raise ValueError("content_digest must be a SHA-256 digest")
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                text(
                    "INSERT INTO ragnor_generation_reservations "
                    "(scope, generation_key, content_digest) VALUES (:scope, :key, :digest) "
                    "ON CONFLICT (scope, generation_key) DO UPDATE "
                    "SET content_digest=EXCLUDED.content_digest "
                    "WHERE ragnor_generation_reservations.content_digest=EXCLUDED.content_digest "
                    "RETURNING content_digest"
                ),
                {"scope": self.scope, "key": generation_key, "digest": content_digest},
            )
            if result.scalar_one_or_none() is None:
                raise ValueError("immutable generation content mismatch on replay")
