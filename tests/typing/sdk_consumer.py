"""mypy consumer contract: import from the installed wheel, not a source alias."""

from typing import Any, assert_type

from ragnor import (
    Chunk,
    Document,
    Ragnor,
    RagnorBuilder,
    SearchResult,
)
from ragnor.core.config import DeploymentMode, RagnorConfig
from ragnor.core.exceptions import (
    ConfigurationError,
    RagnorError,
    classify_exception,
    is_retryable_exception,
)
from ragnor.client.sync_client import AsyncRagnorOperations, SyncRagnor


document = Document(id="policy", text="real text")
result = SearchResult(id="source", document="test")
chunk = Chunk(id="part", text="chunk")
err = ConfigurationError("invalid", code="ERR_CONFIG_INVALID")
cfg = RagnorConfig(deployment_mode=DeploymentMode.LOCAL, embedding_dim=8)
classified = classify_exception(err)
retryable = is_retryable_exception(err)

assert_type(document.id, str)
assert_type(document.embedding, list[float] | None)
assert_type(document.metadata, dict[str, Any])
assert_type(result.document, str)
assert_type(chunk.text, str)
assert_type(RagnorBuilder, type[RagnorBuilder])
assert_type(Ragnor, type[Ragnor])
assert_type(err, ConfigurationError)
assert_type(isinstance(err, RagnorError), bool)
assert_type(err.code, str)
assert_type(err.retryable, bool)
assert_type(SyncRagnor, type[SyncRagnor])
assert_type(cfg, RagnorConfig)
assert_type(cfg.deployment_mode, DeploymentMode)
assert_type(classified, RagnorError)
assert_type(retryable, bool)
assert_type(AsyncRagnorOperations, type[AsyncRagnorOperations])
