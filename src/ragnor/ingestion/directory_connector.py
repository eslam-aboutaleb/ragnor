"""Bounded, fresh filesystem discovery with race-resistant reads."""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import math
import mimetypes
import os
import secrets
import stat
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from .connector_core import (
    ChangeEvent,
    DiscoveryPage,
    LoadResult,
    SkipRecord,
    SourceInaccessibleError,
    SourceItem,
    SourceNotFoundError,
    SourceStatus,
    TransientSourceError,
    collect_pages,
    extract_documents,
    merge_extraction_quality,
    parse_checkpoint_since,
)
from .sources import RawDocument


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _patterns(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def _matches(path: str, patterns: list[str]) -> bool:
    def match_parts(parts: list[str], globs: list[str]) -> bool:
        if not globs:
            return not parts
        if globs[0] == "**":
            return match_parts(parts, globs[1:]) or (bool(parts) and match_parts(parts[1:], globs))
        return (
            bool(parts)
            and fnmatch.fnmatchcase(parts[0], globs[0])
            and match_parts(parts[1:], globs[1:])
        )

    for raw_pattern in patterns:
        pattern = raw_pattern.removeprefix("./")
        if pattern.endswith("/") and not path.endswith("/"):
            continue
        if "/" not in pattern and fnmatch.fnmatchcase(path.rstrip("/").rsplit("/", 1)[-1], pattern):
            return True
        if match_parts(path.strip("/").split("/"), pattern.strip("/").split("/")):
            return True
    return False


def _error(exc: OSError) -> tuple[SourceStatus, Exception]:
    if exc.errno == errno.ENOENT:
        return SourceStatus.NOT_FOUND, SourceNotFoundError(str(exc))
    if exc.errno in {errno.EACCES, errno.EPERM, errno.ELOOP, errno.ENOTDIR}:
        return SourceStatus.INACCESSIBLE, SourceInaccessibleError(str(exc))
    return SourceStatus.TRANSIENT, TransientSourceError(str(exc))


class DirectoryConnector:
    """A scan cursor belongs to one traversal and can be consumed only once.

    Starting a fresh scan invalidates any older cursor. Entries and ignore files
    are visited once, on demand; no permanent inventory is retained.
    """

    id = "directory"
    version = "2"

    def __init__(self, root: Any, **options: Any) -> None:
        self.root = Path(os.path.abspath(root))
        self.include = _patterns(
            options.get("include", options.get("includes", options.get("include_patterns")))
        )
        self.exclude = _patterns(
            options.get("exclude", options.get("excludes", options.get("exclude_patterns")))
        )
        self.ignore_file_names = _patterns(
            options.get(
                "ignore_files",
                options.get(
                    "ignore_file_names",
                    options.get("ignore_filenames", [".gitignore", ".ragignore", ".ignore"]),
                ),
            )
        )
        if any(Path(name).name != name or name in {".", ".."} for name in self.ignore_file_names):
            raise ValueError("ignore file names must be simple file names")
        self.extensions = {
            str(ext).lower().lstrip(".")
            for ext in _patterns(
                options.get(
                    "extensions",
                    options.get("allowed_extensions", options.get("extension_allowlist")),
                )
            )
        }
        self.media_types = {
            str(mt).lower()
            for mt in _patterns(
                options.get(
                    "media_types",
                    options.get(
                        "mime_types",
                        options.get("allowed_media_types", options.get("media_type_allowlist")),
                    ),
                )
            )
        }
        self.max_file_size = _positive(
            options.get(
                "max_file_size",
                options.get("max_bytes", options.get("max_file_size_bytes", 10 * 1024 * 1024)),
            ),
            "max_file_size",
        )
        self.page_size = _positive(
            options.get("page_size", options.get("page_limit", 100)), "page_size"
        )
        self.follow_symlinks = bool(options.get("follow_symlinks", False))
        self._boundary = self.root.resolve() if self.follow_symlinks else self.root
        self.recursive = bool(options.get("recursive", True))
        self.raise_on_not_found = bool(options.get("raise_on_not_found", False))
        self.raise_on_inaccessible = bool(options.get("raise_on_inaccessible", False))
        self.extractor = options.get("extractor")
        self._iterator: Iterator[tuple[str, Any]] | None = None
        self._pending: tuple[str, Any] | None = None
        self._cursor: str | None = None
        self._failed = False
        self._failure_status = SourceStatus.OK
        self._last_result: LoadResult | None = None
        self._skipped: list[SkipRecord] = []
        self._lock = asyncio.Lock()

    @property
    def options_fingerprint(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "include": self.include,
            "exclude": self.exclude,
            "extensions": sorted(self.extensions),
            "media_types": sorted(self.media_types),
            "max_file_size": self.max_file_size,
            "follow_symlinks": self.follow_symlinks,
            "recursive": self.recursive,
            "ignore_files": self.ignore_file_names,
        }

    @property
    def options(self) -> dict[str, Any]:
        return self.options_fingerprint

    @property
    def last_result(self) -> LoadResult | None:
        return self._last_result

    @property
    def skipped_files(self) -> list[SkipRecord]:
        return list(self._skipped)

    @property
    def status(self) -> SourceStatus:
        return self._last_result.status if self._last_result is not None else SourceStatus.OK

    def _open(self, path: Path, *, directory: bool = False) -> int:
        """Open from /, never following any ancestor or final-component link."""
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(path.anchor, flags | os.O_DIRECTORY)
        try:
            for index, part in enumerate(path.parts[1:]):
                is_directory = index < len(path.parts) - 2 or directory
                new_fd = os.open(part, flags | (os.O_DIRECTORY if is_directory else 0), dir_fd=fd)
                os.close(fd)
                fd = new_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _target(self, relative: str) -> Path:
        path = self.root / relative
        if self.follow_symlinks:
            path = path.resolve(strict=True)
            path.relative_to(self._boundary)
        return path

    def _read(self, relative: str) -> bytes:
        fd = self._open(self._target(relative))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EACCES, "Not a regular file", relative)
            if info.st_size > self.max_file_size:
                raise OverflowError("File exceeds maximum size")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(self.max_file_size + 1)
            if len(data) > self.max_file_size or os.fstat(fd).st_size > self.max_file_size:
                raise OverflowError("File exceeds maximum size")
            return data
        finally:
            os.close(fd)

    def _walk(self) -> Iterator[tuple[str, Any]]:  # noqa: PLR0915
        def walk(  # noqa: PLR0915
            relative: str,
            rules: list[tuple[str, Any]],
            ancestors: frozenset[tuple[int, int]],
        ) -> Iterator[tuple[str, Any]]:
            try:
                target = self._target(relative)
                fd = self._open(target, directory=True)
            except (ValueError, RuntimeError) as exc:
                if not relative:
                    yield "error", (".", OSError(errno.EACCES, "Root moved outside scan boundary"))
                else:
                    yield (
                        "skip",
                        SkipRecord(path=relative, reason="symlink_escape", detail=str(exc)),
                    )
                return
            except OSError as exc:
                yield "error", (relative or ".", exc)
                return
            try:
                info = os.fstat(fd)
                identity = (info.st_dev, info.st_ino)
                if identity in ancestors:
                    yield "skip", SkipRecord(path=relative, reason="symlink_cycle")
                    return
                ancestors = ancestors | {identity}
                entries = []
                with os.scandir(fd) as iterator:
                    for entry in iterator:
                        entries.append((entry.name, entry.stat(follow_symlinks=False)))
                rules = list(rules)
                names = {name for name, _ in entries}
                for name in self.ignore_file_names:
                    if name not in names:
                        continue
                    ignore_path = f"{relative}/{name}".lstrip("/")
                    try:
                        data = self._read(ignore_path)
                        # The dependency is required only when an ignore file is present.
                        from pathspec import GitIgnoreSpec  # noqa: PLC0415

                        rules.append(
                            (
                                relative,
                                GitIgnoreSpec.from_lines(
                                    data.decode("utf-8", errors="replace").splitlines()
                                ),
                            )
                        )
                    except (OSError, OverflowError, ImportError) as exc:
                        if not isinstance(exc, OSError):
                            exc = OSError(errno.EIO, str(exc))
                        yield "error", (ignore_path, exc)

                def sort_name(entry: tuple[str, os.stat_result]) -> str:
                    name, info = entry
                    if self.follow_symlinks and stat.S_ISLNK(info.st_mode):
                        try:
                            link_fd = self._open(self._target(f"{relative}/{name}".lstrip("/")))
                            try:
                                info = os.fstat(link_fd)
                            finally:
                                os.close(link_fd)
                        except (OSError, ValueError, RuntimeError):
                            # Inaccessible or broken symlink; fall back to
                            # non-dereferenced stat info for ordering.
                            pass
                    return name + ("/" if stat.S_ISDIR(info.st_mode) else "")

                # A directory's slash participates in POSIX lexical ordering.
                entries.sort(key=sort_name)
                for name, discovered_info in entries:
                    entry_info = discovered_info
                    rel = f"{relative}/{name}".lstrip("/")
                    mode = entry_info.st_mode
                    if stat.S_ISLNK(mode):
                        if not self.follow_symlinks:
                            yield "skip", SkipRecord(path=rel, reason="symlink")
                            continue
                        try:
                            resolved = self._target(rel)
                            link_fd = self._open(resolved)
                            try:
                                entry_info = os.fstat(link_fd)
                                mode = entry_info.st_mode
                            finally:
                                os.close(link_fd)
                        except (ValueError, RuntimeError) as exc:
                            yield (
                                "skip",
                                SkipRecord(path=rel, reason="symlink_escape", detail=str(exc)),
                            )
                            continue
                        except OSError as exc:
                            yield "error", (rel, exc)
                            continue
                    is_dir = stat.S_ISDIR(mode)
                    ignored = False
                    for base, spec in rules:
                        scoped = rel[len(base) + 1 :] if base else rel
                        match = spec.check_file(scoped + ("/" if is_dir else ""))
                        if match.include is not None:
                            ignored = match.include
                    reason = None
                    if ignored:
                        reason = "ignored"
                    elif _matches(rel, self.exclude) or _matches(rel + "/", self.exclude):
                        reason = "excluded"
                    elif is_dir and not self.recursive:
                        reason = "non_recursive"
                    elif not is_dir:
                        suffix = Path(name).suffix.lower().lstrip(".")
                        media_type = mimetypes.guess_type(name)[0] or (
                            "text/markdown" if suffix in {"md", "markdown"} else ""
                        )
                        if self.include and not _matches(rel, self.include):
                            reason = "excluded"
                        elif self.extensions and suffix not in self.extensions:
                            reason = "unsupported_extension"
                        elif self.media_types and not any(
                            fnmatch.fnmatchcase(media_type, mt) for mt in self.media_types
                        ):
                            reason = "unsupported_media_type"
                        elif not stat.S_ISREG(mode):
                            reason = "not_regular"
                        elif entry_info.st_size > self.max_file_size:
                            reason = "oversized"
                    if reason:
                        yield "skip", SkipRecord(path=rel, reason=reason)
                    elif is_dir:
                        yield from walk(rel, rules, ancestors)
                    else:
                        yield (
                            "item",
                            SourceItem(
                                path=str(self.root / rel),
                                relative_path=rel,
                                name=name,
                                size=entry_info.st_size,
                                modified_time=entry_info.st_mtime,
                                media_type=media_type,
                                extension=Path(name).suffix,
                            ),
                        )
            except OSError as exc:
                yield "error", (relative or ".", exc)
            finally:
                os.close(fd)

        yield from walk("", [], frozenset())

    def _discover(self, cursor: Any, limit: int) -> DiscoveryPage:
        if cursor is None:
            if self._iterator is not None:
                self._iterator.close()
            self._iterator = self._walk()
            self._pending = None
            self._failed = False
            self._failure_status = SourceStatus.OK
            self._skipped = []
        elif not isinstance(cursor, str) or self._cursor is None or cursor != self._cursor:
            raise ValueError("Invalid or stale directory scan cursor")
        self._cursor = None
        items, skipped, errors = [], [], []
        for _ in range(limit):
            record = self._pending or next(self._iterator, None)
            self._pending = None
            if record is None:
                break
            kind, value = record
            if kind == "item":
                items.append(value)
            elif kind == "skip":
                skipped.append(value)
            else:
                path, exc = value
                status, error = _error(exc)
                self._failed = True
                self._failure_status = status
                if self.raise_on_not_found and status == SourceStatus.NOT_FOUND:
                    raise error from exc
                if self.raise_on_inaccessible and status == SourceStatus.INACCESSIBLE:
                    raise error from exc
                errors.append(
                    {"path": path, "reason": status.value, "error_type": type(error).__name__}
                )
                skipped.append(SkipRecord(path=path, reason=status.value, detail=str(exc)))
        self._pending = next(self._iterator, None)
        has_more = self._pending is not None
        if has_more:
            self._cursor = secrets.token_urlsafe(24)
        else:
            self._iterator.close()
            self._iterator = None
        self._skipped = skipped
        status = self._failure_status if self._failed else SourceStatus.OK
        if not items and not skipped and not has_more and not self._failed:
            status = SourceStatus.EMPTY
        return DiscoveryPage(
            items,
            cursor=cursor,
            next_cursor=self._cursor,
            has_more=has_more,
            complete=not has_more and not self._failed,
            status=status,
            skipped=skipped,
            errors=errors,
        )

    async def discover(self, *, cursor: Any = None, limit: int | None = None) -> DiscoveryPage:
        size = self.page_size if limit is None else _positive(limit, "limit")
        async with self._lock:
            return await asyncio.to_thread(self._discover, cursor, size)

    def _load_page(self, cursor: Any, size: int) -> LoadResult:
        page = self._discover(cursor, size)
        documents, skipped, errors = [], list(page.skipped), list(page.errors)
        extraction_metadata: dict[str, Any] = {}
        for skip in list(skipped):
            if skip.reason == "oversized" and Path(skip.path).suffix.lower() == ".pdf":
                skipped.remove(skip)
                errors.append({"reason": "pdf_size_limit"})
                self._failed = True
                self._failure_status = SourceStatus.INCOMPLETE
        for item in page:
            rel = item["relative_path"]
            try:
                data = self._read(rel)
                extracted = extract_documents(
                    data,
                    source_name=rel,
                    external_id=rel,
                    metadata={
                        "path": rel,
                        "size": len(data),
                        "media_type": item["media_type"],
                        "modified_time": item.get("modified_time", 0.0),
                    },
                    provenance={"root": str(self.root), "path": item["path"], "relative_path": rel},
                    extractor=self.extractor,
                )
                documents.extend(extracted)
                merge_extraction_quality(extraction_metadata, extracted.metadata)
                skipped.extend(extracted.skipped)
                errors.extend(extracted.errors)
                if not extracted.complete or extracted.errors:
                    self._failed = True
                    self._failure_status = extracted.status
            except OverflowError as exc:
                if item["media_type"] == "application/pdf":
                    errors.append({"reason": "pdf_size_limit"})
                    self._failed = True
                    self._failure_status = SourceStatus.INCOMPLETE
                else:
                    skipped.append(SkipRecord(path=rel, reason="oversized", detail=str(exc)))
            except (OSError, ValueError, RuntimeError) as exc:
                if not isinstance(exc, OSError):
                    exc = OSError(errno.EACCES, str(exc))
                status, error = _error(exc)
                self._failed = True
                self._failure_status = status
                errors.append(
                    {"path": rel, "reason": status.value, "error_type": type(error).__name__}
                )
                skipped.append(SkipRecord(path=rel, reason=status.value, detail=str(exc)))
        result = LoadResult(
            documents,
            skipped=skipped,
            errors=errors,
            cursor=cursor,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
            complete=not page.has_more and not self._failed,
            status=self._failure_status if self._failed else page.status,
            total_items=len(page) + len(page.skipped),
            pages=1,
            metadata=extraction_metadata,
        )
        self._last_result = result
        self._skipped = skipped
        return result

    async def load_page(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        size = self.page_size if limit is None else _positive(limit, "limit")
        async with self._lock:
            return await asyncio.to_thread(self._load_page, cursor, size)

    async def load_all(
        self, *, max_pages: int | None = None, page_size: int | None = None
    ) -> LoadResult:
        if max_pages is not None:
            _positive(max_pages, "max_pages")
        if page_size is not None:
            _positive(page_size, "page_size")
        result = await collect_pages(
            self, max_pages=max_pages, page_size=page_size or self.page_size
        )
        self._last_result = result
        self._skipped = list(result.skipped)
        return result

    async def load(
        self, *, cursor: Any = None, limit: int | None = None, **kwargs: Any
    ) -> LoadResult:
        if cursor is None and limit is None:
            return await self.load_all(**kwargs)
        return await self.load_page(cursor=cursor, limit=limit)

    async def reconcile(self) -> LoadResult:
        return await self.load_all()

    async def load_incremental(self, *, checkpoint: Any = None) -> LoadResult:
        """Load documents modified after checkpoint.

        Absence from an incremental delta never implies deletion; incremental
        results set incremental=True, preventing full-snapshot replacement.
        """
        since = parse_checkpoint_since(checkpoint)

        full_result = await self.load_all()
        if not full_result.complete or full_result.errors:
            full_result.incremental = True
            return full_result

        latest_mtime = since or 0.0
        incremental_docs: list[RawDocument] = []
        skipped = list(full_result.skipped)

        for doc in full_result:
            mtime = float(doc.metadata.get("modified_time", 0.0))
            latest_mtime = max(latest_mtime, mtime)
            if since is not None and mtime <= since:
                skipped.append(SkipRecord(path=doc.source_name, reason="unmodified"))
            else:
                incremental_docs.append(doc)

        new_checkpoint = (
            str(latest_mtime)
            if latest_mtime > 0
            else (str(checkpoint) if checkpoint is not None else None)
        )
        status = SourceStatus.OK if incremental_docs else SourceStatus.EMPTY
        return LoadResult(
            incremental_docs,
            status=status,
            skipped=skipped,
            errors=full_result.errors,
            complete=full_result.complete,
            incremental=True,
            checkpoint=new_checkpoint,
            metadata={**full_result.metadata, "incremental": True},
        )

    async def watch(self, *, interval: float = 2.0) -> AsyncIterator[ChangeEvent]:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        try:
            from watchfiles import awatch  # noqa: PLC0415
        except ImportError:
            changes = None
        else:
            changes = awatch(self.root)
        pending = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + interval
        try:
            while True:
                if changes is not None and pending is None:
                    pending = asyncio.create_task(anext(changes))
                timeout = max(0.0, deadline - loop.time())
                if pending is None:
                    await asyncio.sleep(timeout)
                    ready = False
                else:
                    done, _ = await asyncio.wait({pending}, timeout=timeout)
                    ready = bool(done)
                if loop.time() >= deadline:
                    deadline = loop.time() + interval
                    yield ChangeEvent(type="sync_requested", reason="periodic")
                if ready:
                    try:
                        batch = pending.result()
                    except (OSError, RuntimeError, StopAsyncIteration):
                        await changes.aclose()
                        changes = None
                    else:
                        if batch:
                            yield ChangeEvent(type="sync_requested", reason="change")
                    pending = None
        finally:
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, OSError, RuntimeError, StopAsyncIteration):
                    await pending
            if changes is not None:
                await changes.aclose()

    async def changes(
        self, *, checkpoint: Any = None, interval: float = 2.0
    ) -> AsyncIterator[ChangeEvent]:
        """Yield change events satisfying the ChangeEventSource protocol.

        Watcher events are sync requests, not authoritative upserts/deletions.
        """
        async for event in self.watch(interval=interval):
            if checkpoint is not None and "checkpoint" not in event:
                event["checkpoint"] = checkpoint
            yield event


class FilesystemWatcher:
    def __init__(self, connector: DirectoryConnector, **kwargs: Any) -> None:
        self.connector = connector
        self.kwargs = kwargs

    async def watch(self) -> AsyncIterator[ChangeEvent]:
        async for event in self.connector.watch(**self.kwargs):
            yield event

    async def changes(self, *, checkpoint: Any = None) -> AsyncIterator[ChangeEvent]:
        async for event in self.connector.changes(checkpoint=checkpoint, **self.kwargs):
            yield event


class PeriodicReconciler:
    def __init__(self, connector: Any, interval: float = 60.0) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.connector = connector
        self.interval = interval

    async def run_once(self) -> LoadResult:
        if hasattr(self.connector, "reconcile"):
            return await self.connector.reconcile()
        return await self.connector.load_all()

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.run_once()
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval)

    async def sync(
        self,
        pipeline: Any,
        version_store: Any,
        **kwargs: Any,
    ) -> int:
        """Run reconciliation through the ingestion pipeline."""
        result = await self.run_once()
        return await pipeline.run(result, version_store, **kwargs)


DirectorySource = DirectoryConnector
DirectoryDocumentSource = DirectoryConnector
DirectorySourceConnector = DirectoryConnector
FileSystemConnector = DirectoryConnector
LocalDirectoryConnector = DirectoryConnector
FilesystemSourceConnector = DirectoryConnector
SourceConnectorBase = DirectoryConnector
