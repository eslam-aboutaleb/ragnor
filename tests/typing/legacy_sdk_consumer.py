"""Installed legacy imports must retain their static public types."""

from typing import assert_type

from ragnor.builder import Ragnor, RagnorBuilder
from ragnor.config import DeploymentMode, RagnorConfig
from ragnor.exceptions import ConfigurationError, RagnorError
from ragnor.lifecycle import AsyncCloseable
from ragnor.metrics import DistanceMetric
from ragnor.registry import Registry
from ragnor.sync_client import AsyncRagnorOperations, SyncRagnor
from ragnor.types import Chunk, Document, SearchResult
from ragnor.validation import validate_identifier

assert_type(Document(id="a", text="policy"), Document)
assert_type(Chunk(id="a", text="policy"), Chunk)
assert_type(RagnorConfig(deployment_mode=DeploymentMode.LOCAL), RagnorConfig)
assert_type(ConfigurationError("invalid"), ConfigurationError)
assert_type(Registry[str](), Registry[str])
validate_identifier("chunks", "table")
metric: DistanceMetric = "l2"


async def workflow(client: Ragnor, resource: AsyncCloseable) -> list[SearchResult]:
    assert_type(await client.count(), int)
    assert_type(await client.ingest([Document(id="a", text="policy")]), int)
    results = await client.search("policy")
    assert_type(results, list[SearchResult])
    await resource.aclose()
    return results


def wrap(builder: RagnorBuilder) -> SyncRagnor:
    return SyncRagnor.from_builder(builder)


def as_operations(client: Ragnor) -> AsyncRagnorOperations:
    return client


def as_error(error: ConfigurationError) -> RagnorError:
    return error
