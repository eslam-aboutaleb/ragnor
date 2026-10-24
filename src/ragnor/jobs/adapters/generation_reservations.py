"""Compatibility name for host-bound generation reservations."""

from ragnor.jobs.adapters.relational import RelationalGenerationReservations


class PostgresGenerationReservations(RelationalGenerationReservations):
    """Compatibility alias for the portable SQLAlchemy reservation store."""
