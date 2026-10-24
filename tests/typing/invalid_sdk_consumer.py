"""Deliberately invalid downstream SDK types: mypy MUST reject both errors."""

from ragnor import Document, RagnorBuilder, SearchResult

not_a_document = Document(id=123, text=["not", "a string"])
not_a_result = SearchResult(id=None, document=0)


def invalid_startup(builder: RagnorBuilder) -> None:
    builder.with_startup_check(lambda: None)
    builder.with_startup_check(lambda: None, timeout_seconds="unbounded")
