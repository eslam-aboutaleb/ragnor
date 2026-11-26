"""Persistent snapshot migration qualification on a real on-disk SQLite file.

The version store below is a real durable store (stdlib ``sqlite3`` file) that
persists ``IngestionSnapshot`` through the public serialization helpers. Each
"restart" builds a fresh client and vector store against the same file, so the
only state carried across is what was persisted. Historic rows are written
directly as payloads older releases produced.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from ragnor.builder import RagnorBuilder
from ragnor.chunking import TokenAwareChunker
from ragnor.ingestion.pipeline import VersionRecord
from ragnor.ingestion.versioning import (
    SNAPSHOT_FORMAT_VERSION,
    IngestionSnapshot,
    SnapshotFormatError,
    build_snapshot,
    snapshot_from_dict,
    snapshot_matches,
    snapshot_to_dict,
)
from ragnor.stores.memory import InMemoryVectorStore
from ragnor.types import Document


class SqliteVersionStore:
    def __init__(self, path: Path, store: Any) -> None:
        self._store = store
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS versions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot TEXT NOT NULL, active INTEGER NOT NULL)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def insert_raw(self, payload: dict[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO versions (snapshot, active) VALUES (?, 1)", (json.dumps(payload),)
        )
        self._db.commit()

    async def find_active(self, session: Any) -> VersionRecord | None:
        row = self._db.execute(
            "SELECT id, snapshot FROM versions WHERE active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        try:
            snapshot = snapshot_from_dict(json.loads(row[1]))
        except SnapshotFormatError:
            return None  # unreadable history must never count as "current"
        return VersionRecord(version_id=f"v{row[0]}", snapshot=snapshot)

    async def close_active(self, session: Any) -> None:
        self._db.execute("UPDATE versions SET active = 0 WHERE active = 1")
        self._db.commit()

    async def create_version(self, session: Any, snapshot: IngestionSnapshot) -> VersionRecord:
        cur = self._db.execute(
            "INSERT INTO versions (snapshot, active) VALUES (?, 1)",
            (json.dumps(snapshot_to_dict(snapshot)),),
        )
        self._db.commit()
        return VersionRecord(version_id=f"v{cur.lastrowid}", snapshot=snapshot)

    async def delete_chunks(self, session: Any, version_id: str) -> None:
        await self._store.delete(filters={"version_id": version_id})

    def active_count(self) -> int:
        return self._db.execute("SELECT count(*) FROM versions WHERE active = 1").fetchone()[0]


class _Settings:
    embedding_model = "text-embedding-3-small"


class Tok:
    is_exact = True

    def __init__(self, name: str) -> None:
        self.name = name

    def count_tokens(self, text: str) -> int:
        return len(text)


class Embedder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, float(len(t))] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self(texts)


DATA = [Document(id="policy", text="Refunds are available within thirty days of purchase.")]


async def _start(path: Path, tokenizer: str, embedder: Embedder):
    """Simulate a process start: new vector store and client, same durable file."""
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    versions = SqliteVersionStore(path, store)
    client = await RagnorBuilder.local(
        embedder,
        embedding_dim=2,
        store=store,
        version_store=versions,
        chunker=TokenAwareChunker(Tok(tokenizer), max_tokens=20, overlap_tokens=3),
    ).build()
    return client, versions, store


async def test_unchanged_config_skips_after_restart_and_changed_tokenizer_reindexes(
    tmp_path: Path,
) -> None:
    db = tmp_path / "versions.sqlite3"
    embedder = Embedder()

    client, versions, _ = await _start(db, "enc-v1", embedder)
    await client.ingest(DATA)
    first = (await versions.find_active(None)).version_id
    calls = embedder.calls
    versions.close()

    client, versions, _ = await _start(db, "enc-v1", embedder)
    await client.ingest(DATA)
    assert embedder.calls == calls  # skipped: snapshot survived the restart
    assert (await versions.find_active(None)).version_id == first
    versions.close()

    client, versions, _ = await _start(db, "enc-v2", embedder)
    await client.ingest(DATA)
    assert embedder.calls > calls
    active = await versions.find_active(None)
    assert active.version_id != first
    assert versions.active_count() == 1
    versions.close()


def _current(tokenizer: str) -> IngestionSnapshot:
    chunker = TokenAwareChunker(Tok(tokenizer), max_tokens=20, overlap_tokens=3)
    return build_snapshot("h", _Settings(), chunker)  # type: ignore[arg-type]


def test_roundtrip_is_lossless() -> None:
    snap = _current("enc-v1")
    payload = json.loads(json.dumps(snapshot_to_dict(snap)))
    assert payload["snapshot_format"] == SNAPSHOT_FORMAT_VERSION
    assert snapshot_from_dict(payload) == snap


HISTORIC_V0 = {
    "source_hash": "h",
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "chunker_version": "token-aware-v1",
    "chunk_size": 20,
    "overlap": 3,
    "retrieval_schema_version": "v1",
}


def test_historic_payload_without_newer_fields_loads_but_never_matches_current() -> None:
    old = snapshot_from_dict(dict(HISTORIC_V0))
    assert old.chunker_config_hash == ""
    assert old.extractor_id == "none"
    current = _current("enc-v1")
    assert current.chunker_config_hash
    assert not snapshot_matches(current, old)


def test_unknown_future_keys_are_ignored_but_future_format_is_refused() -> None:
    assert snapshot_from_dict({**HISTORIC_V0, "added_later": 1}).source_hash == "h"
    with pytest.raises(SnapshotFormatError, match="newer"):
        snapshot_from_dict({**HISTORIC_V0, "snapshot_format": SNAPSHOT_FORMAT_VERSION + 1})


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {},
        {k: v for k, v in HISTORIC_V0.items() if k != "source_hash"},
        {**HISTORIC_V0, "embedding_dim": "abc"},
        {**HISTORIC_V0, "snapshot_format": -1},
        {**HISTORIC_V0, "extractor_options": [1]},
    ],
)
def test_malformed_payloads_are_rejected(payload) -> None:
    with pytest.raises(SnapshotFormatError):
        snapshot_from_dict(payload)


async def test_historic_row_in_durable_file_is_reindexed_not_skipped(tmp_path: Path) -> None:
    db = tmp_path / "versions.sqlite3"
    embedder = Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    seeded = SqliteVersionStore(db, store)
    seeded.insert_raw(HISTORIC_V0)  # row written by an older release
    seeded.close()

    client, versions, _ = await _start(db, "enc-v1", embedder)
    await client.ingest(DATA)
    assert embedder.calls == 1
    active = await versions.find_active(None)
    assert active.snapshot.chunker_config_hash
    assert versions.active_count() == 1
    versions.close()


async def test_unreadable_row_is_reindexed_not_trusted(tmp_path: Path) -> None:
    db = tmp_path / "versions.sqlite3"
    embedder = Embedder()
    store = InMemoryVectorStore(embedding_dim=2, embedding_fn=embedder)
    seeded = SqliteVersionStore(db, store)
    seeded.insert_raw({"snapshot_format": 99, **HISTORIC_V0})
    seeded.close()

    client, versions, _ = await _start(db, "enc-v1", embedder)
    await client.ingest(DATA)
    assert embedder.calls == 1
    versions.close()
