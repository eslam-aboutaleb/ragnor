"""Deliberately invalid downstream SDK types: mypy MUST reject both errors."""

from ragnor import Document, SearchResult

not_a_document = Document(id=123, text=["not", "a string"])
not_a_result = SearchResult(id=None, document=0)
