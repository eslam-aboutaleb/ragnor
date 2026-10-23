"""mypy consumer contract: import from the installed wheel, not a source alias."""

from typing import Any, assert_type

from ragnor import Chunk, Document, Ragnor, RagnorBuilder, SearchResult
from ragnor.exceptions import ConfigurationError, RagnorError
from ragnor.sync_client import SyncRagnor


document = Document(id="policy", text="real text")
result = SearchResult(id="source", document="test")
chunk = Chunk(id="part", text="chunk")
err = ConfigurationError("invalid", code="ERR_CONFIG_INVALID")
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
