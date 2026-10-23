"""mypy consumer contract: import from the installed wheel, not a source alias."""

from typing import Any, assert_type

from ragnor import Chunk, Document, Ragnor, RagnorBuilder, SearchResult


document = Document(id="policy", text="real text")
result = SearchResult(id="source", document="test")
chunk = Chunk(id="part", text="chunk")
assert_type(document.id, str)
assert_type(document.embedding, list[float] | None)
assert_type(document.metadata, dict[str, Any])
assert_type(result.document, str)
assert_type(chunk.text, str)
assert_type(RagnorBuilder, type[RagnorBuilder])
assert_type(Ragnor, type[Ragnor])
