"""Filesystem connector authority, paging, ignore, and race regressions."""

from __future__ import annotations

import asyncio
import errno
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragit.ingestion.connector_core import (
    SourceInaccessibleError,
    SourceNotFoundError,
    SourceStatus,
    TransientSourceError,
)
from ragit.ingestion.directory_connector import (
    DirectoryConnector,
    DirectoryDocumentSource,
    DirectorySource,
    DirectorySourceConnector,
    FileSystemConnector,
    FilesystemSourceConnector,
    LocalDirectoryConnector,
    PeriodicReconciler,
)


def test_old_aliases_share_implementation() -> None:
    assert all(
        alias is DirectoryConnector
        for alias in (
            DirectoryDocumentSource,
            DirectorySource,
            DirectorySourceConnector,
            FileSystemConnector,
            FilesystemSourceConnector,
            LocalDirectoryConnector,
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        {"page_size": 0},
        {"page_size": -1},
        {"page_size": True},
        {"max_file_size": 0},
        {"max_bytes": None},
        {"max_bytes": -1},
        {"max_file_size": float("inf")},
        {"ignore_files": ["../.ignore"]},
    ],
)
def test_invalid_bounds_rejected(tmp_path: Path, options: dict) -> None:
    with pytest.raises(ValueError):
        DirectoryConnector(tmp_path, **options)


async def test_sorted_relative_paths_and_full_load(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("last")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.txt").write_text("nested")
    (tmp_path / "a.txt").write_text("first")
    connector = DirectoryConnector(tmp_path, page_size=1)
    result = await connector.load()
    assert [doc.source_name for doc in result] == ["a.txt", "a/nested.txt", "z.txt"]
    assert result.complete and result.authoritative
    assert result.pages == 3
    assert all(doc.external_id == doc.source_name for doc in result)
    assert result[1].provenance["relative_path"] == "a/nested.txt"


async def test_new_scan_sees_creation_change_and_deletion(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("original")
    connector = DirectoryConnector(tmp_path)
    assert (await connector.load())[0].content == "original"
    path.write_text("changed")
    (tmp_path / "b.txt").write_text("new")
    result = await connector.reconcile()
    assert {doc.source_name: doc.content for doc in result} == {
        "a.txt": "changed",
        "b.txt": "new",
    }
    path.unlink()
    assert [doc.source_name for doc in await connector.load()] == ["b.txt"]


async def test_opaque_cursors_reject_replay_invalid_and_stale(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("content")
    connector = DirectoryConnector(tmp_path, page_size=1)
    first = await connector.discover()
    assert not first.complete and first.has_more
    second = await connector.discover(cursor=first.next_cursor)
    with pytest.raises(ValueError, match="cursor"):
        await connector.discover(cursor=first.next_cursor)
    await connector.discover()
    with pytest.raises(ValueError, match="cursor"):
        await connector.discover(cursor=second.next_cursor)
    for cursor in (0, "0", "garbage", {"offset": 0}):
        with pytest.raises(ValueError, match="cursor"):
            await connector.discover(cursor=cursor)


async def test_partial_load_can_continue_but_is_not_authoritative(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"{index}.txt").write_text(str(index))
    connector = DirectoryConnector(tmp_path)
    partial = await connector.load_all(max_pages=1, page_size=2)
    assert len(partial) == 2
    assert not partial.complete and partial.has_more
    assert partial.next_cursor
    final = await connector.load_page(cursor=partial.next_cursor, limit=2)
    assert [doc.source_name for doc in final] == ["2.txt", "3.txt"]
    assert not final.complete and not final.authoritative
    assert not final.has_more
    assert (await connector.load_all(page_size=2)).authoritative


async def test_discovery_bounds_skips_and_avoids_tree_prewalk(tmp_path: Path, monkeypatch) -> None:
    for index in range(20):
        directory = tmp_path / f"{index:02}"
        directory.mkdir()
        (directory / "item.txt").write_text("text")
    calls = []
    original = os.scandir

    def tracked(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(os, "scandir", tracked)
    connector = DirectoryConnector(tmp_path, include="*.md", page_size=1)
    page = await connector.discover()
    assert len(page) + len(page.skipped) <= 1
    assert len(calls) <= 3
    assert page.has_more


async def test_nested_ignore_rules_and_last_negation_win(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.txt\n!keep.txt\n/root-only.md\n")
    (tmp_path / ".ragignore").write_text("*.log\n!keep.log\n")
    (tmp_path / "keep.txt").write_text("keep")
    (tmp_path / "drop.txt").write_text("drop")
    (tmp_path / "root-only.md").write_text("drop")
    (tmp_path / "keep.log").write_text("keep")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".ignore").write_text("!allowed.txt\nlocal.md\n!local.md\n")
    (sub / "allowed.txt").write_text("allowed")
    (sub / "drop.txt").write_text("drop")
    (sub / "root-only.md").write_text("keep")
    (sub / "local.md").write_text("keep")
    result = await DirectoryConnector(tmp_path, exclude=[".*", "**/.*"]).load()
    assert {doc.source_name for doc in result} == {
        "keep.txt",
        "keep.log",
        "sub/allowed.txt",
        "sub/root-only.md",
        "sub/local.md",
    }
    assert {skip.path for skip in result.skipped if skip.reason == "ignored"} == {
        "drop.txt",
        "root-only.md",
        "sub/drop.txt",
    }
    assert result.authoritative


async def test_ignore_directory_pruning_and_negated_parent(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("blocked/\n!blocked/\nblocked/*\n!blocked/keep.md\n")
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked" / "keep.md").write_text("keep")
    (tmp_path / "blocked" / "drop.md").write_text("drop")
    result = await DirectoryConnector(tmp_path, extensions="md").load()
    assert [doc.source_name for doc in result] == ["blocked/keep.md"]
    assert any(
        skip.path == "blocked/drop.md" and skip.reason == "ignored" for skip in result.skipped
    )
    assert result.authoritative


async def test_include_extension_media_and_exclude_options(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for name in ("keep.md", "drop.md", "no.txt", "sub/deep.md"):
        (tmp_path / name).write_text("text")
    result = await DirectoryConnector(
        tmp_path,
        include_patterns="**/*.md",
        excludes="drop.md",
        allowed_extensions="md",
        mime_types=["text/*"],
        max_bytes=100,
    ).load()
    assert [doc.source_name for doc in result] == ["keep.md", "sub/deep.md"]
    assert result.authoritative


async def test_binary_and_oversized_are_policy_skips(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "big.txt").write_text("x" * 51)
    (tmp_path / "ok.txt").write_text("ok")
    result = await DirectoryConnector(tmp_path, max_file_size=50, page_size=1).load()
    assert [doc.source_name for doc in result] == ["ok.txt"]
    assert {skip.path: skip.reason for skip in result.skipped} == {
        "a.bin": "binary",
        "big.txt": "oversized",
    }
    assert not result.errors and result.authoritative


async def test_html_and_pdf_dispatched_before_binary_detection(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    (tmp_path / "blank.pdf").write_bytes(buffer.getvalue())
    (tmp_path / "page.html").write_text("<nav>ignore</nav><h1>Heading</h1><p>body</p>")
    result = await DirectoryConnector(tmp_path).load()
    assert len(result) == 1 and "Heading" in result[0].content
    assert "ignore" not in result[0].content
    assert result.authoritative
    assert any(skip.path == "blank.pdf" and skip.reason != "binary" for skip in result.skipped)


async def test_outside_root_symlinks_never_read(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "file-link").symlink_to(outside / "secret.txt")
    (root / "dir-link").symlink_to(outside, target_is_directory=True)
    for follow in (False, True):
        result = await DirectoryConnector(root, follow_symlinks=follow).load()
        assert not result
        assert {skip.reason for skip in result.skipped} == {
            "symlink_escape" if follow else "symlink",
        }
        assert result.authoritative


async def test_follow_contained_links_and_cycle_protection(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "item.txt").write_text("safe")
    (sub / "back").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "alias.txt").symlink_to(sub / "item.txt")
    result = await DirectoryConnector(tmp_path, follow_symlinks=True, page_size=1).load()
    assert {doc.source_name for doc in result} == {"alias.txt", "sub/item.txt"}
    assert any(skip.reason == "symlink_cycle" for skip in result.skipped)
    assert result.authoritative


async def test_follow_directory_aliases_are_sorted_by_posix_path(tmp_path: Path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "file.txt").write_text("safe")
    (tmp_path / "a").symlink_to(tmp_path / "z", target_is_directory=True)
    (tmp_path / "a.txt").write_text("first")
    result = await DirectoryConnector(tmp_path, follow_symlinks=True, page_size=1).load()
    assert [doc.source_name for doc in result] == ["a.txt", "a/file.txt", "z/file.txt"]


@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("follow", [False, True])
async def test_symlink_replacement_between_discovery_and_read_is_blocked(
    tmp_path: Path,
    monkeypatch,
    ancestor: bool,
    follow: bool,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "sub").mkdir()
    (root / "sub" / "item.txt").write_text("safe")
    (outside / "item.txt").write_text("secret")
    connector = DirectoryConnector(root, follow_symlinks=follow)
    original = connector._read

    def replace_then_read(relative):
        if ancestor:
            (root / "sub").rename(root / "moved")
            (root / "sub").symlink_to(outside, target_is_directory=True)
        else:
            (root / "sub" / "item.txt").unlink()
            (root / "sub" / "item.txt").symlink_to(outside / "item.txt")
        return original(relative)

    monkeypatch.setattr(connector, "_read", replace_then_read)
    result = await connector.load()
    assert not result and result.errors
    assert not result.complete and not result.authoritative
    assert result.errors[0]["error_type"] == "SourceInaccessibleError"


async def test_follow_symlinks_pins_original_root_boundary(tmp_path: Path, monkeypatch) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "item.txt").write_text("safe")
    (outside / "item.txt").write_text("secret")
    connector = DirectoryConnector(root, follow_symlinks=True)
    original = connector._read

    def replace_root(relative):
        root.rename(tmp_path / "old-root")
        root.symlink_to(outside, target_is_directory=True)
        return original(relative)

    monkeypatch.setattr(connector, "_read", replace_root)
    result = await connector.load()
    assert not result and result.errors and not result.authoritative
    rescanned = await connector.load()
    assert rescanned.errors and not rescanned.complete and not rescanned.authoritative


async def test_file_growth_is_rechecked_at_read_time(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "item.txt"
    path.write_text("small")
    connector = DirectoryConnector(tmp_path, max_bytes=10)
    original = connector._read

    def grow_then_read(relative):
        path.write_bytes(b"x" * 100)
        return original(relative)

    monkeypatch.setattr(connector, "_read", grow_then_read)
    result = await connector.load()
    assert not result and result.skipped[0].reason == "oversized"
    assert result.authoritative


async def test_read_is_bounded_when_file_grows_after_fstat(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "item.txt"
    path.write_text("small")
    original = os.fdopen
    requested = []

    class GrowingReader:
        def __init__(self, *args, **kwargs):
            self.stream = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            path.write_bytes(b"x" * 1000)
            requested.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(os, "fdopen", GrowingReader)
    result = await DirectoryConnector(tmp_path, max_bytes=10).load()
    assert requested == [11]
    assert not result and result.skipped[0].reason == "oversized"
    assert result.authoritative


async def test_missing_empty_and_inaccessible_are_distinct(tmp_path: Path) -> None:
    missing = await DirectoryConnector(tmp_path / "absent").load_page()
    assert missing.status == SourceStatus.NOT_FOUND
    assert missing.errors[0]["error_type"] == "SourceNotFoundError"
    assert not missing.complete
    empty = await DirectoryConnector(tmp_path).load()
    assert empty.status == SourceStatus.EMPTY and empty.authoritative
    not_directory = tmp_path / "file.txt"
    not_directory.write_text("file")
    failure = await DirectoryConnector(not_directory).load_page()
    assert failure.status == SourceStatus.INACCESSIBLE and not failure.complete
    assert failure.errors[0]["error_type"] == "SourceInaccessibleError"


@pytest.mark.parametrize(
    "error_number,error_type",
    [
        (errno.EACCES, SourceInaccessibleError),
        (errno.EIO, TransientSourceError),
        (errno.ENOENT, SourceNotFoundError),
    ],
)
async def test_scan_failure_is_never_authoritative(
    tmp_path: Path, monkeypatch, error_number, error_type
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "ok.txt").write_text("okay")
    original = os.scandir

    def fail_subdirectory(fd):
        if os.fstat(fd).st_ino == (tmp_path / "a").stat().st_ino:
            raise OSError(error_number, "injected listing failure")
        return original(fd)

    monkeypatch.setattr(os, "scandir", fail_subdirectory)
    result = await DirectoryConnector(tmp_path).load()
    assert result.errors and result.errors[0]["error_type"] == error_type.__name__
    assert not result.complete and not result.authoritative


async def test_extraction_error_invalidates_snapshot(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\ninvalid")
    result = await DirectoryConnector(tmp_path).load()
    assert result.errors and not result.authoritative and not result.complete


async def test_periodic_reconciler_runs_until_stopped() -> None:
    stop = asyncio.Event()

    class Connector:
        calls = 0

        async def reconcile(self):
            self.calls += 1
            if self.calls == 2:
                stop.set()
            return []

    connector = Connector()
    await asyncio.wait_for(PeriodicReconciler(connector, interval=0.001).run(stop), timeout=1)
    assert connector.calls == 2


async def test_watch_requests_sync_even_without_events(tmp_path: Path) -> None:
    watcher = DirectoryConnector(tmp_path).watch(interval=0.01)
    try:
        event = await asyncio.wait_for(anext(watcher), timeout=2)
        assert event.type == "sync_requested" and event.reason == "periodic"
        assert "path" not in event
    finally:
        await watcher.aclose()


async def test_watch_changes_are_hints_and_periodic_sync_is_independent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def changes(root):
        yield {(3, str(root / "deleted.txt"))}
        await asyncio.Event().wait()

    monkeypatch.setitem(sys.modules, "watchfiles", SimpleNamespace(awatch=changes))
    watcher = DirectoryConnector(tmp_path).watch(interval=0.01)
    try:
        change = await asyncio.wait_for(anext(watcher), timeout=1)
        periodic = await asyncio.wait_for(anext(watcher), timeout=1)
        assert change == {"type": "sync_requested", "reason": "change"}
        assert periodic == {"type": "sync_requested", "reason": "periodic"}
    finally:
        await watcher.aclose()


async def test_watcher_failure_still_requests_periodic_reconciliation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def changes(root):
        raise FileNotFoundError(root)
        yield  # pragma: no cover

    monkeypatch.setitem(sys.modules, "watchfiles", SimpleNamespace(awatch=changes))
    watcher = DirectoryConnector(tmp_path).watch(interval=0.01)
    try:
        event = await asyncio.wait_for(anext(watcher), timeout=1)
        assert event == {"type": "sync_requested", "reason": "periodic"}
    finally:
        await watcher.aclose()
