"""SDK-07 Unicode property invariants and true ingestion generation retirement."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from ragnor.builder import InMemoryVersionStore, RagnorBuilder
from ragnor.chunking import TokenAwareChunker
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.types import Document


class ExactCodepointTokenizer:
    is_exact = True

    def __init__(self, name: str = "unicode-v1") -> None:
        self.name = name

    def count_tokens(self, text: str) -> int:
        return len(text)


ALPHABET = ("a", "é", "م", "中", "👩", "🏻", "🇪", "\u200d", "\u0301", "\n", " ", "{", "}")


@settings(max_examples=80, deadline=None)
@given(
    text=st.text(alphabet=ALPHABET, min_size=0, max_size=130),
    budget=st.integers(min_value=1, max_value=25),
    overlap=st.integers(min_value=0, max_value=24),
)
def test_arbitrary_unicode_spans_cover_source_without_exceeding_budget(
    text: str, budget: int, overlap: int
) -> None:
    chunker = TokenAwareChunker(
        ExactCodepointTokenizer(),
        max_tokens=budget,
        overlap_tokens=overlap % budget,
    )
    chunks = chunker.chunk(text)
    if not text.strip():
        assert not chunks
        return
    assert chunks
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[-1].metadata["end_offset"] == len(text)
    assert all(
        c.text == text[c.metadata["start_offset"] : c.metadata["end_offset"]]
        and 0 < len(c.text) <= budget
        for c in chunks
    )
    assert all(
        nxt.metadata["start_offset"] > prev.metadata["start_offset"]
        and nxt.metadata["start_offset"] <= prev.metadata["end_offset"]
        for prev, nxt in zip(chunks, chunks[1:], strict=False)
    )
    coverage = set().union(
        *(range(c.metadata["start_offset"], c.metadata["end_offset"]) for c in chunks)
    )
    assert coverage == set(range(len(text)))


class Embedder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, float(len(text))] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


@pytest.mark.asyncio
async def test_changed_tokenizer_invalidates_live_index_and_retires_old_chunks() -> None:
    embedder = Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    versions = InMemoryVersionStore(store)
    data = [Document(id="policy", text="Refunds are available within thirty days.")]

    async def build(tokenizer: str):
        return await RagnorBuilder.local(
            embedder,
            embedding_dim=2,
            store=store,
            version_store=versions,
            chunker=TokenAwareChunker(
                ExactCodepointTokenizer(tokenizer), max_tokens=13, overlap_tokens=2
            ),
        ).build()

    first = await build("model-encoding-v1")
    await first.ingest(data)
    active_one = await versions.find_active(None)
    assert active_one is not None
    embedded_after_first = embedder.calls
    stored_one = await store.get(filters={"version_id": active_one.version_id})
    assert stored_one

    await first.ingest(data)
    assert embedder.calls == embedded_after_first
    assert (await versions.find_active(None)).version_id == active_one.version_id

    second = await build("model-encoding-v2")
    await second.ingest(data)
    active_two = await versions.find_active(None)
    assert active_two is not None
    assert active_two.version_id != active_one.version_id
    assert active_two.snapshot.chunker_config_hash != active_one.snapshot.chunker_config_hash
    assert embedder.calls > embedded_after_first
    assert await store.get(filters={"version_id": active_one.version_id}) == []
    assert await store.get(filters={"version_id": active_two.version_id})
    assert await second.count() == len(await store.get())
