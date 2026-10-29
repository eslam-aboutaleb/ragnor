"""Shared connector records and authoritative snapshot collection.

``complete`` describes an entire snapshot, never merely a successful page.
Intentional policy skips are separate from errors that invalidate a snapshot.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from hashlib import sha256
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from ragnor.ingestion.records import RawDocument


class SourceStatus(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    INACCESSIBLE = "inaccessible"
    EMPTY = "empty"
    EMPTY_SUCCESS = "empty_success"
    TRANSIENT = "transient"
    TRANSIENT_FAILURE = "transient_failure"
    INCOMPLETE = "incomplete"
    PARTIAL = "partial"
    ERROR = "error"
    SUCCESS = "success"


class SourceError(Exception):
    """Base connector error."""


class SourceNotFoundError(SourceError):
    """Source does not exist."""


class SourceInaccessibleError(SourceError):
    """Source cannot be read."""


class TransientSourceError(SourceError):
    """Temporary source failure."""


class IncompleteScanError(SourceError):
    """Traversal did not yield an authoritative snapshot."""


def _error_record(error: Any) -> dict[str, Any]:
    """Do not retain exception messages, which may contain source credentials."""
    from ragnor.ingestion.extraction_types import PDFExtractionError  # noqa: PLC0415

    if isinstance(error, PDFExtractionError):
        code = error.code if error.code in PDFExtractionError.codes else "provider_failed"
        return {"reason": "pdf_extraction_failed", "code": code, "error_type": "PDFExtractionError"}
    if isinstance(error, dict):
        # Connector-owned structured errors remain useful without copying
        # arbitrary provider messages or payloads into public diagnostics.
        record: dict[str, Any] = {
            key: value
            for key, value in error.items()
            if key in {"reason", "error_type", "code"} and isinstance(value, str)
        }
        for key in ("status_code", "retry_after"):
            value = error.get(key)
            if key in error and (value is None or isinstance(value, (int, float))):
                record[key] = value
        if isinstance(error.get("page"), int) and error["page"] > 0:
            record["page"] = error["page"]
        source_id = error.get("source_id")
        if (
            isinstance(source_id, str)
            and len(source_id) == 64
            and all(char in "0123456789abcdef" for char in source_id)
        ):
            record["source_id"] = source_id
        return record
    return {"reason": "source_failure", "error_type": type(error).__name__}


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class SourceItem(_AttrDict):
    """Discovered source descriptor with mapping and attribute access."""


class SkipRecord(_AttrDict):
    def __init__(
        self, path: str = "", reason: str = "", detail: str | None = None, **extra: Any
    ) -> None:
        super().__init__(path=path, reason=reason, detail=detail, **extra)


class ChangeEvent(_AttrDict):
    """Change descriptor; deletion events must explicitly identify their target."""


class DiscoveryPage(list):
    def __init__(  # noqa: PLR0913
        self,
        items: Sequence[Any] | None = None,
        *,
        next_cursor: Any = None,
        has_more: bool = False,
        complete: bool = True,
        cursor: Any = None,
        total: int | None = None,
        status: SourceStatus = SourceStatus.OK,
        skipped: Sequence[Any] | None = None,
        errors: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        incremental: bool = False,
        checkpoint: Any = None,
    ) -> None:
        super().__init__(items or [])
        self.next_cursor = next_cursor
        self.has_more = has_more
        self.complete = complete and not has_more and cursor is None
        self.cursor = cursor
        self.total = total
        self.status = status
        self.skipped = list(skipped or [])
        self.errors = list(errors or [])
        self.metadata = dict(metadata or {})
        self.incremental = incremental
        self.checkpoint = checkpoint

    @property
    def items(self) -> list[Any]:
        return list(self)


class LoadResult(list):
    def __init__(  # noqa: PLR0913
        self,
        documents: Sequence[RawDocument] | None = None,
        *,
        status: SourceStatus = SourceStatus.OK,
        skipped: Sequence[Any] | None = None,
        errors: Sequence[Any] | None = None,
        cursor: Any = None,
        next_cursor: Any = None,
        has_more: bool = False,
        complete: bool = True,
        page: int = 0,
        pages: int = 0,
        total_items: int = 0,
        metadata: dict[str, Any] | None = None,
        incremental: bool = False,
        checkpoint: Any = None,
    ) -> None:
        super().__init__(documents or [])
        self.status = status
        self.skipped = list(skipped or [])
        self.errors = list(errors or [])
        self.cursor = cursor
        self.next_cursor = next_cursor
        self.has_more = has_more
        self.complete = complete and not has_more and cursor is None
        self.page = page
        self.pages = pages
        self.total_items = total_items
        self.metadata = dict(metadata or {})
        self.incremental = incremental
        self.checkpoint = checkpoint

    @property
    def documents(self) -> list[RawDocument]:
        return list(self)

    @property
    def authoritative(self) -> bool:
        return (
            self.complete
            and self.metadata.get("extraction_quality", {}).get("complete", True)
            and self.status in {"ok", "empty", "success", "empty_success"}
            and not self.errors
            and not self.incremental
        )


def merge_extraction_quality(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Aggregate bounded page counts independently of traversal completeness."""
    quality = incoming.get("extraction_quality")
    if quality is None:
        return
    previous = target.get("extraction_quality", {"complete": True, "pages": 0, "counts": {}})
    counts = dict(previous["counts"])
    for status, count in quality["counts"].items():
        counts[status] = counts.get(status, 0) + count
    target["extraction_quality"] = {
        "complete": previous["complete"] and quality["complete"],
        "pages": previous["pages"] + quality["pages"],
        "counts": counts,
    }


@runtime_checkable
class SourceConnector(Protocol):
    async def discover(self, *, cursor: Any = None, limit: int | None = None) -> DiscoveryPage: ...
    async def load_page(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult: ...
    async def load(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult: ...


@runtime_checkable
class IncrementalSourceConnector(Protocol):
    """Checkpoint is opaque and persisted only after successful application.

    Incremental results must set incremental=True, even for an initial checkpoint
    of None; absence from a delta never implies deletion from a full snapshot.
    """

    async def load_incremental(self, *, checkpoint: Any = None) -> LoadResult: ...


@runtime_checkable
class ChangeEventSource(Protocol):
    """Events carry resumable checkpoints, committed only after application.

    A None checkpoint starts the provider's event history, not a full snapshot.
    Replays may occur; consumers must apply events idempotently.
    """

    def changes(self, *, checkpoint: Any = None) -> AsyncIterator[ChangeEvent]: ...


async def collect_pages(
    connector: SourceConnector, max_pages: int | None = None, page_size: int | None = None
) -> LoadResult:
    """Collect from the beginning; never bless a partial or failed traversal."""
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    if page_size is not None and page_size < 1:
        raise ValueError("page_size must be positive")
    result = LoadResult(complete=False)
    cursor = None
    seen: list[Any] = []
    while True:
        try:
            page = await connector.load_page(cursor=cursor, limit=page_size)
        except Exception as exc:
            result.errors.append(_error_record(exc))
            result.next_cursor = cursor
            result.has_more = True
            result.status = SourceStatus.INCOMPLETE
            return result
        result.extend(page)
        result.skipped.extend(page.skipped)
        result.errors.extend(_error_record(error) for error in page.errors)
        result.pages += 1
        result.total_items += page.total_items
        result.incremental = result.incremental or page.incremental
        result.checkpoint = page.checkpoint
        result.next_cursor = page.next_cursor
        result.has_more = page.has_more
        merge_extraction_quality(result.metadata, page.metadata)
        result.metadata.update(
            {key: value for key, value in page.metadata.items() if key != "extraction_quality"}
        )
        if page.errors or page.status not in {"ok", "empty", "success", "empty_success"}:
            result.status = (
                SourceStatus.INCOMPLETE
                if page.status
                in {
                    "ok",
                    "empty",
                    "success",
                    "empty_success",
                }
                else page.status
            )
            return result
        if not page.has_more:
            # A terminal page reached after a cursor cannot itself be complete;
            # the collector can certify the traversal that began at None.
            result.complete = cursor is not None or page.complete
            result.status = SourceStatus.OK if result else SourceStatus.EMPTY
            if not result.complete:
                result.status = SourceStatus.INCOMPLETE
            return result
        token = page.next_cursor
        if token is None or token == cursor or token in seen:
            result.errors.append(
                {"reason": "invalid_continuation", "error_type": "IncompleteScanError"}
            )
            result.status = SourceStatus.INCOMPLETE
            return result
        if max_pages is not None and result.pages >= max_pages:
            result.status = SourceStatus.INCOMPLETE
            return result
        seen.append(token)
        cursor = token


def _outcome_pages(extractor: Any, data: bytes, source_name: str, result: LoadResult):
    from dataclasses import asdict  # noqa: PLC0415
    from ragnor.ingestion.extractors import PageExtractionStatus  # noqa: PLC0415

    outcomes = extractor.extract_outcomes(data)
    if outcomes and outcomes[0].diagnostics.get("runtime"):
        result.metadata["extraction_runtime"] = outcomes[0].diagnostics["runtime"]
    complete = bool(outcomes) and all(outcome.complete for outcome in outcomes)
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    quality = {"complete": complete, "pages": len(outcomes), "counts": counts}
    result.metadata["extraction_quality"] = quality
    if not complete:
        result.complete = False
        result.status = SourceStatus.INCOMPLETE
        result.errors = [
            {"reason": outcome.reason, "code": str(outcome.status), "page": outcome.page}
            for outcome in outcomes
            if not outcome.complete
        ][:100]
        if not result.errors:
            result.errors.append({"reason": "no_pages"})
        if getattr(extractor, "options", {}).get("policy", "strict") != "allow_partial":
            return []
    for outcome in outcomes:
        if outcome.status in {
            PageExtractionStatus.VERIFIED_BLANK,
            PageExtractionStatus.POLICY_EXCLUDED,
        }:
            result.skipped.append(
                SkipRecord(source_name, str(outcome.status), location={"page": outcome.page})
            )
    return [
        (
            outcome.page,
            outcome.text,
            {
                "page_extraction": {
                    "provider": outcome.provider,
                    "version": outcome.version,
                    "status": str(outcome.status),
                    "reason": outcome.reason,
                    "words": [asdict(word) for word in outcome.words],
                    "regions": [asdict(region) for region in outcome.regions],
                    "tables": [asdict(table) for table in outcome.tables],
                    "diagnostics": outcome.diagnostics,
                }
            }
            if outcome.words or outcome.tables or outcome.regions or outcome.diagnostics
            else {},
        )
        for outcome in outcomes
        if outcome.text.strip()
        and outcome.status
        not in {PageExtractionStatus.VERIFIED_BLANK, PageExtractionStatus.POLICY_EXCLUDED}
    ]


def extract_documents(  # noqa: PLR0913, PLR0917, PLR0915
    data: bytes,
    source_name: str,
    external_id: str | None,
    metadata: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
    extractor: Any = None,
) -> LoadResult:
    """Extract text or page documents, preserving the actual extraction config."""
    from ragnor.ingestion.extractors import (  # noqa: PLC0415
        HTMLExtractor,
        PDFExtractor,
        TextExtractor,
    )
    from ragnor.embeddings._safe_config import safe_config  # noqa: PLC0415

    metadata = dict(metadata or {})
    provenance = dict(provenance or {})
    media_type = str(metadata.get("media_type", "")).split(";", 1)[0].lower()
    suffix = PurePosixPath(source_name.split("?", 1)[0]).suffix.lower()
    if extractor is None:
        if media_type == "application/pdf" or suffix == ".pdf":
            extractor = PDFExtractor()
        elif media_type in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
            extractor = HTMLExtractor()
        else:
            extractor = TextExtractor()
    identity = str(getattr(extractor, "id", type(extractor).__name__))
    metadata["extraction"] = {
        "id": identity,
        "version": str(getattr(extractor, "version", "0")),
        "options": safe_config(getattr(extractor, "options", {})),
    }
    result = LoadResult()
    try:
        if identity == "text":
            # Decode using the actual text encoding, not a UTF-8-only byte
            # heuristic: UTF-16 and other explicitly configured encodings work.
            try:
                decoded = data.decode(getattr(extractor, "encoding", "utf-8"))
            except UnicodeDecodeError:
                decoded = "\x00"
            controls = sum(ord(char) < 32 and char not in "\n\r\t\f\b" for char in decoded)
            if "\x00" in decoded or controls / max(1, len(decoded)) > 0.1:
                result.skipped.append(SkipRecord(source_name, "binary"))
                return result
        if hasattr(extractor, "extract_outcomes"):
            pages = _outcome_pages(extractor, data, source_name, result)
            # Optional providers attest package/model identities while extracting.
            metadata["extraction"]["options"] = safe_config(getattr(extractor, "options", {}))
            metadata.update(result.metadata)
            if "extraction_runtime" in result.metadata:
                metadata["extraction"]["options"]["runtime"] = result.metadata["extraction_runtime"]
        else:
            # Custom legacy extractors retain their caller-defined completeness contract.
            pages = (
                extractor.extract_pages(data)
                if hasattr(extractor, "extract_pages")
                else [(None, extractor.extract(data, name=source_name))]
            )
        for extracted_page in pages:
            page, text = extracted_page[:2]
            page_metadata = extracted_page[2] if len(extracted_page) > 2 else {}
            location = {"page": page} if page is not None else None
            if not text.strip():
                result.skipped.append(SkipRecord(source_name, "no_text", location=location))
                continue
            result.append(
                RawDocument(
                    source_name,
                    text,
                    external_id,
                    {**metadata, **page_metadata},
                    dict(provenance),
                    identity,
                    location,
                )
            )
    except Exception as exc:
        result.clear()
        result.errors.append(_error_record(exc))
        result.complete = False
        result.status = SourceStatus.ERROR
        if hasattr(extractor, "extract_outcomes"):
            result.metadata["extraction_quality"] = {
                "complete": False,
                "pages": 0,
                "counts": {},
            }
    if result.errors and hasattr(extractor, "extract_outcomes"):
        source_id = sha256((external_id or source_name).encode()).hexdigest()
        for error in result.errors:
            error["source_id"] = source_id
    return result


def parse_checkpoint_since(checkpoint: Any) -> float | None:
    """Parse a checkpoint into an epoch timestamp float, or None."""
    if checkpoint is None:
        return None
    if isinstance(checkpoint, (int, float)):
        return float(checkpoint)
    if isinstance(checkpoint, str):
        try:
            return float(checkpoint)
        except ValueError:
            from datetime import datetime  # noqa: PLC0415

            try:
                return datetime.fromisoformat(checkpoint).timestamp()
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid checkpoint: {checkpoint}") from exc
    if isinstance(checkpoint, dict):
        return float(checkpoint.get("mtime", checkpoint.get("since", 0)))
    raise ValueError(f"Invalid checkpoint: {checkpoint}")
