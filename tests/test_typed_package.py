"""PEP 561 typing marker and typed high-level SDK regression contract."""

from __future__ import annotations

from importlib.resources import files

from ragnor import Chunk, Document, Ragnor, RagnorBuilder, SearchResult


def test_pep561_marker_in_installed_ragnor_package() -> None:
    assert files("ragnor").joinpath("py.typed").is_file()


def test_root_types_match_canonical_implementations() -> None:
    from ragnor.builder import Ragnor as CanonicalRagnor
    from ragnor.builder import RagnorBuilder as CanonicalBuilder

    assert Ragnor is CanonicalRagnor
    assert RagnorBuilder is CanonicalBuilder
    assert Document(id="doc", text="text").embedding is None
    assert Chunk(id="chunk", text="text").metadata == {}
    assert SearchResult(id="doc", document="text").document == "text"
