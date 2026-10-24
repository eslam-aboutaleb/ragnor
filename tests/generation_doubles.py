"""Explicitly ephemeral generation CAS for local-only contract tests."""


class LocalGenerationReservations:
    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.digests: dict[str, str] = {}

    async def reserve_generation(self, generation_key: str, content_digest: str) -> None:
        prior = self.digests.setdefault(generation_key, content_digest)
        if prior != content_digest:
            raise ValueError("immutable generation content mismatch on replay")
