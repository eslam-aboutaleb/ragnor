"""S3-compatible storage with bounded reads and fail-closed pagination.

Listing tokens are opaque continuation tokens, not incremental checkpoints.
Clients/credentials are injected and never included in document identities or
fingerprints. A successful list does not make failed object reads authoritative.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from ragnor.ingestion.connector_core import (
    DiscoveryPage,
    LoadResult,
    SkipRecord,
    SourceItem,
    SourceStatus,
    collect_pages,
    extract_documents,
    merge_extraction_quality,
    parse_checkpoint_since,
)


@runtime_checkable
class ObjectStorageClient(Protocol):
    """Injected synchronous S3-compatible client; SDK retry policy is host-owned."""

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...


def _error(exc: Exception, path: str) -> tuple[SourceStatus, dict[str, Any]]:
    """Expose only safe structured error fields, never SDK exception strings."""
    response = getattr(exc, "response", {})
    response = response if isinstance(response, dict) else {}
    code = str(response.get("Error", {}).get("Code", ""))
    status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if code in {"NoSuchKey", "NoSuchBucket", "NotFound", "404"} or status_code == 404:
        status = SourceStatus.NOT_FOUND
    elif code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"} or status_code in {
        401,
        403,
    }:
        status = SourceStatus.INACCESSIBLE
    else:
        status = SourceStatus.TRANSIENT
    return status, {"path": path, "reason": status.value}


class ObjectStorageConnector:
    """Bounded S3-compatible snapshot connector.

    Configure ``max_pages``/``max_items`` to cap a run. Hitting a cap before
    exhaustion, a malformed token, a changed object, or any read failure makes
    the result non-authoritative. No retries happen here: SDK retry/backoff and
    rate limiting belong to the injected client. Reconciliation restarts at the
    first page; a listing is not a point-in-time snapshot under concurrent writes.
    """

    def __init__(self, bucket: str, **options: Any) -> None:
        self.bucket = bucket
        self.prefix = str(options.get("prefix", options.get("root", "")))
        self.client = options.get("client")
        self._credentials = options.get("credentials")
        self._client_factory = options.get("client_factory")
        self._endpoint_url = options.get("endpoint_url")
        self._region = options.get("region")
        self.max_file_size = int(options.get("max_file_size", options.get("max_bytes", 10485760)))
        self.page_size = int(options.get("page_size", 100))
        self.max_pages = options.get("max_pages")
        self.max_items = options.get("max_items")
        extensions: Sequence[str] = options.get("extensions", options.get("allowed_extensions", []))
        self.extensions = {str(extension).lower().lstrip(".") for extension in extensions}
        self.extractor = options.get("extractor")
        self._last_result: LoadResult | None = None
        if self.max_file_size <= 0 or not 0 < self.page_size <= 1000:
            raise ValueError("max_file_size must be positive and page_size must be 1..1000")
        for name in ("max_pages", "max_items"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    @property
    def options(self) -> dict[str, Any]:
        """Non-credential configuration suitable for fingerprints."""
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "max_file_size": self.max_file_size,
            "extensions": sorted(self.extensions),
        }

    @property
    def last_result(self) -> LoadResult | None:
        return self._last_result

    def _client(self) -> ObjectStorageClient:
        if self.client is None:
            if self._client_factory is not None:
                self.client = self._client_factory(self._credentials)
            else:
                try:
                    import boto3  # noqa: PLC0415
                except ImportError:
                    raise ImportError(
                        "Install ragnor[object-storage] or inject an ObjectStorageClient"
                    ) from None
                self.client = boto3.client(
                    "s3", endpoint_url=self._endpoint_url, region_name=self._region
                )
        return self.client

    async def discover(self, *, cursor: Any = None, limit: int | None = None) -> DiscoveryPage:
        size = self.page_size if limit is None else limit
        if not isinstance(size, int) or not 0 < size <= 1000:
            raise ValueError("limit must be 1..1000")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ValueError("cursor must be a nonempty opaque listing token")
        kwargs = {"Bucket": self.bucket, "Prefix": self.prefix, "MaxKeys": size}
        if cursor is not None:
            kwargs["ContinuationToken"] = cursor
        try:
            response = await asyncio.to_thread(self._client().list_objects_v2, **kwargs)
        except ImportError:
            raise
        except Exception as exc:
            status, error = _error(exc, self.bucket)
            return DiscoveryPage([], status=status, errors=[error], complete=False)
        if not isinstance(response, dict) or not isinstance(response.get("Contents", []), list):
            return DiscoveryPage(
                [],
                status=SourceStatus.TRANSIENT,
                errors=[{"reason": "malformed_listing"}],
                complete=False,
            )
        has_more = bool(response.get("IsTruncated", False))
        token = response.get("NextContinuationToken")
        errors = []
        if has_more and (not isinstance(token, str) or not token or token == cursor):
            errors.append({"reason": "invalid_continuation"})
        items = []
        for entry in response.get("Contents", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("Key"), str):
                errors.append({"reason": "malformed_item"})
                continue
            key = entry["Key"]
            if not key.startswith(self.prefix):
                errors.append({"reason": "key_outside_prefix"})
                continue
            if key.endswith("/"):
                continue
            items.append(
                SourceItem(
                    key=key,
                    relative_path=key[len(self.prefix) :].lstrip("/"),
                    size=entry.get("Size"),
                    etag=entry.get("ETag"),
                    last_modified=entry.get("LastModified"),
                )
            )
        return DiscoveryPage(
            items,
            cursor=cursor,
            next_cursor=token if has_more else None,
            has_more=has_more,
            complete=not has_more and not errors,
            errors=errors,
            status=SourceStatus.TRANSIENT if errors else SourceStatus.OK,
        )

    def _read(self, item: SourceItem) -> tuple[bytes | None, dict[str, Any]]:
        kwargs = {"Bucket": self.bucket, "Key": item.key}
        if item.get("etag"):
            kwargs["IfMatch"] = item.etag
        response = self._client().get_object(**kwargs)
        body = response["Body"]
        try:
            size = response.get("ContentLength", 0)
            if size > self.max_file_size:
                return None, response
            pieces = []
            total = 0
            while total <= self.max_file_size:
                piece = body.read(min(65536, self.max_file_size + 1 - total))
                if not piece:
                    break
                total += len(piece)
                if total > self.max_file_size:
                    return None, response
                pieces.append(piece)
            data = b"".join(pieces)
            if size and len(data) != size:
                raise OSError("incomplete object read")
            return data, response
        finally:
            body.close()

    async def load_page(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        page = await self.discover(cursor=cursor, limit=limit)
        result = LoadResult(
            [],
            status=page.status,
            errors=page.errors,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
            complete=page.complete,
            total_items=len(page),
        )
        for item in page:
            relative = item.relative_path
            suffix = relative.rsplit(".", 1)[-1].lower() if "." in relative else ""
            if self.extensions and suffix not in self.extensions:
                result.skipped.append(SkipRecord(relative, "unsupported_extension"))
                continue
            if item.get("size") is not None and item.size > self.max_file_size:
                self._oversized(result, relative, suffix == "pdf")
                continue
            try:
                data, response = await asyncio.to_thread(self._read, item)
            except Exception as exc:
                status, error = _error(exc, relative)
                result.errors.append(error)
                result.status = status
                result.complete = False
                continue
            if data is None:
                self._oversized(
                    result,
                    relative,
                    suffix == "pdf"
                    or response.get("ContentType", "").split(";", 1)[0] == "application/pdf",
                )
                continue
            extracted = extract_documents(
                data,
                source_name=relative,
                external_id=item.key,
                metadata={
                    "media_type": response.get("ContentType", ""),
                    "size": len(data),
                    "last_modified": response.get("LastModified") or item.get("last_modified"),
                },
                provenance={"bucket": self.bucket, "key": item.key, "etag": item.get("etag")},
                extractor=self.extractor,
            )
            result.extend(extracted)
            merge_extraction_quality(result.metadata, extracted.metadata)
            result.skipped.extend(extracted.skipped)
            result.errors.extend(extracted.errors)
            if extracted.errors:
                result.complete = False
                result.status = extracted.status
        if not result and not result.errors and not page.has_more:
            result.status = SourceStatus.EMPTY
        self._last_result = result
        return result

    @staticmethod
    def _oversized(result: LoadResult, relative: str, pdf: bool) -> None:
        if pdf:
            result.errors.append({"reason": "pdf_size_limit"})
            result.complete = False
            result.status = SourceStatus.INCOMPLETE
        else:
            result.skipped.append(SkipRecord(relative, "oversized"))

    async def load_all(
        self, *, max_pages: int | None = None, page_size: int | None = None
    ) -> LoadResult:
        count = 0

        async def bounded_page(*, cursor=None, limit=None):
            nonlocal count
            remaining = None if self.max_items is None else self.max_items - count
            size = self.page_size if limit is None else limit
            if remaining is not None:
                size = min(size, remaining)
            page = await self.load_page(cursor=cursor, limit=size)
            count += len(page)
            if remaining is not None and len(page) > remaining:
                del page[remaining:]
                page.errors.append({"reason": "max_items"})
                page.complete = False
                page.status = SourceStatus.INCOMPLETE
            elif self.max_items is not None and count >= self.max_items and page.has_more:
                page.errors.append({"reason": "max_items"})
                page.complete = False
                page.status = SourceStatus.INCOMPLETE
            return page

        result = await collect_pages(
            SimpleNamespace(load_page=bounded_page),
            max_pages=self.max_pages if max_pages is None else max_pages,
            page_size=self.page_size if page_size is None else page_size,
        )
        self._last_result = result
        return result

    async def load(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        if cursor is not None or limit is not None:
            return await self.load_page(cursor=cursor, limit=limit)
        return await self.load_all()

    async def reconcile(self) -> LoadResult:
        return await self.load_all()

    async def load_incremental(self, *, checkpoint: Any = None) -> LoadResult:
        """Load objects modified after checkpoint."""
        since = parse_checkpoint_since(checkpoint)

        full = await self.load_all()
        if not full.complete or full.errors:
            full.incremental = True
            return full

        latest = since or 0.0
        docs = []
        skipped = list(full.skipped)
        for doc in full:
            mtime = doc.metadata.get("last_modified")
            mtime_val = 0.0
            if isinstance(mtime, (int, float)):
                mtime_val = float(mtime)
            elif hasattr(mtime, "timestamp"):
                mtime_val = mtime.timestamp()
            elif isinstance(mtime, str):
                from datetime import datetime  # noqa: PLC0415

                try:
                    mtime_val = datetime.fromisoformat(mtime).timestamp()
                except Exception:
                    mtime_val = 0.0
            latest = max(latest, mtime_val)
            if since is not None and mtime_val <= since:
                skipped.append(SkipRecord(doc.source_name, "unmodified"))
            else:
                docs.append(doc)

        new_cp = (
            str(latest) if latest > 0 else (str(checkpoint) if checkpoint is not None else None)
        )
        status = SourceStatus.OK if docs else SourceStatus.EMPTY
        return LoadResult(
            docs,
            status=status,
            skipped=skipped,
            errors=full.errors,
            complete=full.complete,
            incremental=True,
            checkpoint=new_cp,
            metadata={**full.metadata, "incremental": True},
        )

    scan = load_all
    fetch_all = load_all


ObjectStorageSource = ObjectStorageConnector
S3Connector = ObjectStorageConnector
ObjectStoreConnector = ObjectStorageConnector
