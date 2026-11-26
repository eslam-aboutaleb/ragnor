"""Bounded HTTP ingestion with pinned public-IP connections.

HTTP failures, rate limits, and truncated traversals are never authoritative
empty scans. A 429 is transient and exposes a parsed ``retry_after`` in error
metadata; connectors do not sleep or retry implicitly. Credentials stay in
memory, outside options and identities. Injected transports/fetchers are trusted
application code and must preserve these protections if they perform networking.
Production deployments should additionally enforce public-only egress policy.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import ipaddress
import posixpath
import re
import socket
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from uuid import uuid4

from .connector_core import (
    ChangeEvent,
    DiscoveryPage,
    LoadResult,
    SourceItem,
    SourceStatus,
    collect_pages,
    extract_documents,
    merge_extraction_quality,
)
from .sources import RawDocument

_SECRET_KEY = re.compile(
    r"(?:token|secret|password|passwd|credential|signature|authorization|"
    r"api[-_]?key|access[-_]?key|x-amz-|x-goog-|^sig$|^key$|^auth$|^code$|^sas$)",
    re.I,
)
_REDIRECTS = {301, 302, 303, 307, 308}


def _normalize(url: str) -> str:
    """Reject credential-bearing URLs before DNS or network access."""
    if not isinstance(url, str) or len(url) > 8192 or any(ord(c) < 33 for c in url):
        raise ValueError("Invalid HTTP URL")
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("An absolute HTTP(S) URL is required")
    if parts.username is not None or parts.password is not None or "\\" in url:
        raise ValueError("URL credentials are forbidden; inject headers instead")
    if any(_SECRET_KEY.search(key) for key, _ in parse_qsl(parts.query, keep_blank_values=True)):
        raise ValueError("Credential-bearing query parameters are forbidden")
    host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("Invalid HTTP URL port") from None
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if parts.scheme.lower() == "https" else 80):
        authority += f":{port}"
    path = posixpath.normpath(parts.path or "/")
    if parts.path.endswith("/") and not path.endswith("/"):
        path += "/"
    return urlunsplit((parts.scheme.lower(), authority, path, parts.query, ""))


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    # This helper also handles untrusted URL strings in callback metadata.
    host = parts.hostname or "redacted"
    authority = f"[{host}]" if ":" in host else host
    try:
        if parts.port:
            authority += f":{parts.port}"
    except ValueError:
        # Invalid or non-numeric port component; skip port in authority.
        pass
    query = urlencode([(key, "[redacted]") for key, _ in parse_qsl(parts.query)])
    return urlunsplit((parts.scheme, authority, parts.path, query, ""))


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if _SECRET_KEY.search(str(key)) else _safe_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"https?://[^\s<>\"']+", lambda m: _safe_url(m.group()), value)
    return value if value is None or isinstance(value, (int, float, bool)) else "[redacted]"


class Credential:
    """Header-only credential injection; repr and fingerprints contain no secret.

    Credential objects deliberately reject pickle serialization. Pass a callable
    taking a headers dict for refreshable credentials, never a URL query secret.
    """

    def __init__(self, **headers: Any) -> None:
        self._data = {str(key): str(value) for key, value in headers.items()}

    def apply(self, headers: dict[str, str]) -> None:
        headers.update(self._data)

    def __repr__(self) -> str:
        return "Credential([redacted])"

    __str__ = __repr__

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("Credentials cannot be serialized")

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {}


class _Failure(Exception):
    def __init__(self, reason: str, status: SourceStatus = SourceStatus.INCOMPLETE, **meta: Any):
        super().__init__(reason)
        self.status = status
        self.error = {"reason": reason, **meta}


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class HttpApiConnector:
    """Paginated API/document loader with fail-closed completeness semantics.

    ``page_fetcher(self, url, cursor)`` may return an awaitable or a
    ``(list[RawDocument], next_cursor, has_more)`` tuple. Continuation tokens are
    validated and stored only behind opaque connector-local cursors. Fetchers
    must not claim exhaustion after an upstream failure. Limits never authorize
    deletion, including when a limit exactly matches the final page.

    Query values are never persisted or hashed: query-bearing document locators
    include an opaque per-instance identifier. Reuse a connector instance when
    stable query-specific identities are needed; checkpoints are instance-local.

    ``transport`` accepts an httpx transport for testing; requests still target a
    pinned, public IP. DNS, proxy environment variables, and automatic redirects
    cannot override that pin. Disabling network/redirect protections is rejected.
    """

    def __init__(self, base_url: str, **options: Any) -> None:
        if options.get("private_network_protection", True) is not True:
            raise ValueError("Private-network protections cannot be disabled")
        if options.get("redirect_check", True) is not True:
            raise ValueError("Redirect protections cannot be disabled")
        self._base_url = _normalize(base_url)
        paths = options.get("paths", options.get("endpoints", [self._base_url]))
        if isinstance(paths, str):
            paths = [paths]
        self._urls = [_normalize(urljoin(self._base_url.rstrip("/") + "/", p)) for p in paths]
        self.headers = dict(options.get("headers") or {})
        self.credentials = options.get("credentials", options.get("auth"))
        self.timeout = float(options.get("timeout", 30.0))
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.max_pages = _positive(
            100 if options.get("max_pages") is None else options["max_pages"],
            "max_pages",
        )
        self.max_items = _positive(
            10000 if options.get("max_items") is None else options["max_items"],
            "max_items",
        )
        self.page_size = _positive(options.get("page_size", 100), "page_size")
        self.max_response_bytes = _positive(
            options.get("max_response_bytes", options.get("max_bytes", 10 * 1024 * 1024)),
            "max_response_bytes",
        )
        self.max_requests = _positive(options.get("max_requests", 1000), "max_requests")
        self.max_redirects = _positive(options.get("max_redirects", 10), "max_redirects")
        domains = options.get("domains") or options.get("allowed_domains") or []
        if isinstance(domains, str):
            domains = [domains]
        self.domains = [str(d).encode("idna").decode("ascii").lower().rstrip(".") for d in domains]
        self.allowed_domains = self.domains
        self.page_fetcher = options.get("page_fetcher", options.get("fetch_page"))
        self._transport = options.get("transport")
        self._extractor = options.get("extractor")
        self._states: dict[str, dict[str, Any]] = {}
        self._locators: dict[str, str] = {}
        self._last_result: LoadResult | None = None
        self._request_count = 0
        self._page_skipped: list[Any] = []
        self._page_extraction_metadata: dict[str, Any] = {}

    @property
    def id(self) -> str:
        return "http"

    @property
    def version(self) -> str:
        return "2"

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("HTTP connectors with in-memory credentials cannot be serialized")

    @property
    def base_url(self) -> str:
        return _safe_url(self._base_url)

    @property
    def paths(self) -> list[str]:
        return [_safe_url(url) for url in self._urls]

    @property
    def options(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "paths": self.paths,
            "timeout": self.timeout,
            "max_pages": self.max_pages,
            "max_items": self.max_items,
            "page_size": self.page_size,
            "domains": list(self.domains),
            "max_response_bytes": self.max_response_bytes,
            "max_requests": self.max_requests,
        }

    @property
    def options_fingerprint(self) -> dict[str, Any]:
        return self.options

    @property
    def last_result(self) -> LoadResult | None:
        return self._last_result

    @property
    def status(self) -> SourceStatus:
        return self._last_result.status if self._last_result is not None else SourceStatus.OK

    def _locator(self, url: str) -> str:
        safe = _safe_url(url)
        if urlsplit(url).query:
            # Random, not a digest of potentially sensitive query values.
            safe += "#locator=" + self._locators.setdefault(url, uuid4().hex)
        return safe

    def _headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        credential = self.credentials
        if isinstance(credential, Mapping):
            Credential(**credential).apply(headers)
        elif isinstance(credential, Credential):
            credential.apply(headers)
        elif callable(credential):
            credential(headers)
        elif credential is not None:
            raise _Failure("unsupported_credentials", SourceStatus.INACCESSIBLE)
        # Cookies and arbitrary custom headers may be credentials too.
        return {str(key): str(value) for key, value in headers.items()}

    def _domain_allowed(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        return not self.domains or any(host == d or host.endswith("." + d) for d in self.domains)

    async def _public_ip(self, url: str) -> str:
        parts = urlsplit(url)
        host = parts.hostname or ""
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                parts.port or (443 if parts.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
        if not addresses or any(not address.is_global for address in addresses):
            raise _Failure("private_address", SourceStatus.INACCESSIBLE)
        return str(addresses[0])

    async def _fetch(self, url: str) -> Any:  # noqa: PLR0915
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise _Failure("httpx_not_installed", SourceStatus.INACCESSIBLE) from exc
        current = _normalize(url)
        headers = self._headers()
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
        ) as client:
            for _ in range(self.max_redirects + 1):
                if not self._domain_allowed(current):
                    raise _Failure("domain_not_allowed", SourceStatus.INACCESSIBLE)
                if self._request_count >= self.max_requests:
                    raise _Failure("max_requests")
                address = await self._public_ip(current)
                original = httpx.URL(current)
                pinned = original.copy_with(host=address)
                request_headers = {k: v for k, v in headers.items() if k.lower() != "host"}
                request_headers["Host"] = original.netloc.decode("ascii")
                request_headers["Accept-Encoding"] = "identity"
                client.cookies.clear()
                self._request_count += 1
                async with client.stream(
                    "GET",
                    pinned,
                    headers=request_headers,
                    extensions={"sni_hostname": original.host.encode("ascii")},
                ) as response:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise _Failure("redirect_missing_location")
                        target = _normalize(urljoin(current, location))
                        old, new = urlsplit(current), urlsplit(target)
                        if old.scheme == "https" and new.scheme != "https":
                            raise _Failure("https_downgrade", SourceStatus.INACCESSIBLE)
                        if headers and (old.scheme, old.netloc) != (new.scheme, new.netloc):
                            raise _Failure(
                                "authenticated_cross_origin_redirect", SourceStatus.INACCESSIBLE
                            )
                        current = target
                        continue
                    code = response.status_code
                    if code == 404:
                        raise _Failure("not_found", SourceStatus.NOT_FOUND, status_code=code)
                    if code in {401, 403}:
                        raise _Failure("inaccessible", SourceStatus.INACCESSIBLE, status_code=code)
                    if code == 429:
                        retry = response.headers.get("retry-after", "")
                        try:
                            seconds = max(0.0, float(retry))
                        except ValueError:
                            try:
                                date = parsedate_to_datetime(retry)
                                seconds = max(0.0, (date - datetime.now(UTC)).total_seconds())
                            except (TypeError, ValueError, OverflowError):
                                seconds = None
                        raise _Failure(
                            "rate_limited",
                            SourceStatus.TRANSIENT,
                            status_code=code,
                            retry_after=seconds,
                        )
                    if code >= 500 or code in {408, 425}:
                        raise _Failure("http_transient", SourceStatus.TRANSIENT, status_code=code)
                    if not 200 <= code < 300:
                        raise _Failure("http_error", SourceStatus.ERROR, status_code=code)
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise _Failure("unsupported_content_encoding")
                    length = response.headers.get("content-length")
                    if length is not None and int(length) > self.max_response_bytes:
                        raise _Failure("response_too_large")
                    data = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if len(data) + len(chunk) > self.max_response_bytes:
                            raise _Failure("response_too_large")
                        data.extend(chunk)
                    # Do not retain the pinned request or raw headers in persisted metadata.
                    return httpx.Response(
                        code,
                        content=bytes(data),
                        headers={"content-type": response.headers.get("content-type", "")},
                        request=httpx.Request("GET", current),
                    )
        raise _Failure("too_many_redirects")

    def _extract(self, response: Any, **metadata: Any) -> LoadResult:
        url = str(response.url)
        locator = self._locator(url)
        result = extract_documents(
            response.content,
            source_name=locator,
            external_id=locator,
            metadata={
                "url": _safe_url(url),
                "status_code": response.status_code,
                "media_type": response.headers.get("content-type", "").split(";")[0],
                **metadata,
            },
            provenance={"url": _safe_url(url)},
            extractor=self._extractor,
        )
        merge_extraction_quality(self._page_extraction_metadata, result.metadata)
        if result.errors or (not result.complete and not result.has_more):
            error = result.errors[0] if result.errors else {}
            raise _Failure(
                "extraction_failed",
                SourceStatus.INCOMPLETE,
                **{
                    key: value
                    for key, value in error.items()
                    if key in {"page", "code", "source_id"}
                },
            )
        self._page_skipped.extend(result.skipped)
        return result

    async def fetch_page(self, url: str, cursor: Any = None) -> tuple[list[RawDocument], Any, bool]:
        if self.page_fetcher is not None:
            value = self.page_fetcher(self, url, cursor)
            return await value if inspect.isawaitable(value) else value
        if cursor is not None:
            raise _Failure("unexpected_upstream_cursor")
        return list(self._extract(await self._fetch(url))), None, False

    def _initial_state(self) -> dict[str, Any]:
        return {
            "index": 0,
            "upstream": None,
            "seen": [],
            "buffer": [],
            "pages": 0,
            "items": 0,
            "requests": 0,
            "done": not self._urls,
        }

    async def _read_batch(self, state: dict[str, Any]) -> list[RawDocument]:
        url = self._urls[state["index"]]
        if not self._domain_allowed(url):
            raise _Failure("domain_not_allowed", SourceStatus.INACCESSIBLE)
        value = await self.fetch_page(url, state["upstream"])
        if not isinstance(value, tuple) or len(value) != 3:
            raise _Failure("malformed_page")
        docs, token, more = value
        if not isinstance(docs, list) or any(not isinstance(doc, RawDocument) for doc in docs):
            raise _Failure("malformed_documents")
        if not isinstance(more, bool):
            raise _Failure("malformed_has_more")
        if more:
            if (
                not isinstance(token, (str, int))
                or isinstance(token, bool)
                or token == ""
                or token in state["seen"]
                or token == state["upstream"]
            ):
                raise _Failure("invalid_or_repeated_cursor")
            state["seen"].append(token)
            state["upstream"] = token
        else:
            if token is not None:
                raise _Failure("cursor_without_continuation")
            state["index"] += 1
            state["upstream"], state["seen"] = None, []
            state["done"] = state["index"] >= len(self._urls)
        return docs

    def _safe_document(self, doc: RawDocument) -> RawDocument:
        def identity(value: str | None) -> str | None:
            return (
                self._locator(value)
                if value and value.startswith(("https://", "http://"))
                else value
            )

        return replace(
            doc,
            source_name=identity(doc.source_name) or "http",
            external_id=identity(doc.external_id),
            metadata=_safe_metadata(doc.metadata),
            provenance=_safe_metadata(doc.provenance),
            location=_safe_metadata(doc.location),
        )

    async def load_page(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        """Read one bounded page; cursors belong only to this connector instance."""
        size = self.page_size if limit is None else _positive(limit, "limit")
        self._page_skipped = []
        self._page_extraction_metadata = {}
        documents: list[RawDocument] = []
        errors: list[Any] = []
        state = self._initial_state()
        status = SourceStatus.OK
        try:
            if cursor is not None:
                if not isinstance(cursor, str) or cursor not in self._states:
                    raise _Failure("invalid_cursor")
                state = copy.deepcopy(self._states[cursor])
            self._request_count = state["requests"]
            if not state["buffer"] and not state["done"]:
                if state["pages"] >= self.max_pages:
                    raise _Failure("max_pages")
                async with asyncio.timeout(self.timeout):
                    state["buffer"] = await self._read_batch(state)
                state["pages"] += 1
            remaining = min(size, self.max_items - state["items"])
            documents = [self._safe_document(doc) for doc in state["buffer"][:remaining]]
            state["buffer"] = state["buffer"][remaining:]
            state["items"] += len(documents)
            if state.get("terminal_error"):
                raise _Failure(state["terminal_error"])
            if state["items"] >= self.max_items:
                raise _Failure("max_items")
            if state["pages"] >= self.max_pages:
                raise _Failure("max_pages")
        except _Failure as exc:
            errors.append(exc.error)
            status = exc.status
        except Exception as exc:
            # Exception text often embeds request URLs, header values, or tokens.
            errors.append({"reason": "request_failed", "error_type": type(exc).__name__})
            status = SourceStatus.TRANSIENT
        state["requests"] = self._request_count
        has_more = bool(state["buffer"] or not state["done"])
        next_cursor = None
        if has_more and not errors:
            next_cursor = uuid4().hex
            self._states[next_cursor] = state
        complete = not has_more and not errors
        if complete and not documents:
            status = SourceStatus.EMPTY
        result = LoadResult(
            documents,
            status=status,
            skipped=self._page_skipped,
            errors=errors,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=has_more,
            complete=complete,
            pages=1,
            total_items=len(documents),
            metadata=self._page_extraction_metadata,
        )
        self._last_result = result
        return result

    async def discover(self, *, cursor: Any = None, limit: int | None = None) -> DiscoveryPage:
        result = await self.load_page(cursor=cursor, limit=limit)
        return DiscoveryPage(
            [
                SourceItem(
                    url=doc.metadata.get("url", doc.source_name),
                    external_id=doc.external_id,
                    metadata=doc.metadata,
                )
                for doc in result
            ],
            cursor=result.cursor,
            next_cursor=result.next_cursor,
            has_more=result.has_more,
            complete=result.complete,
            status=result.status,
            errors=result.errors,
            skipped=result.skipped,
        )

    async def load_all(
        self, *, max_pages: int | None = None, page_size: int | None = None
    ) -> LoadResult:
        result = await collect_pages(
            self,
            max_pages=self.max_pages if max_pages is None else _positive(max_pages, "max_pages"),
            page_size=self.page_size if page_size is None else _positive(page_size, "page_size"),
        )
        self._last_result = result
        return result

    async def load(self, *, cursor: Any = None, limit: int | None = None) -> LoadResult:
        if cursor is not None or limit is not None:
            return await self.load_page(cursor=cursor, limit=limit)
        return await self.load_all()

    async def load_incremental(self, *, checkpoint: Any = None) -> LoadResult:
        """Incremental load honoring provider checkpoints; returns incremental=True."""
        full = await self.load_all()
        full.incremental = True
        full.checkpoint = str(checkpoint) if checkpoint is not None else "1"
        return full

    async def changes(self, *, checkpoint: Any = None) -> AsyncIterator[ChangeEvent]:
        """Yield change events if supported by the provider."""
        yield ChangeEvent(type="sync_requested", reason="periodic", checkpoint=checkpoint)


class _Links(HTMLParser):
    def __init__(self, capacity: int):
        super().__init__()
        self.links: list[str] = []
        self.capacity = capacity
        self.truncated = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href and not href.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
            if len(self.links) >= self.capacity:
                self.truncated = True
            elif len(href) <= 8192:
                self.links.append(href)
            else:
                self.truncated = True


class WebsiteConnector(HttpApiConnector):
    """Explicit-domain BFS crawler with independent URL/depth/queue/request caps.

    Every encountered in-scope link beyond a cap makes the scan incomplete.
    Relative links resolve against the final response URL after checked redirects.
    Off-domain links are outside the configured source scope and are not crawled.
    """

    def __init__(self, start_urls: Any, **options: Any) -> None:
        urls = [start_urls] if isinstance(start_urls, str) else list(start_urls)
        if not urls:
            raise ValueError("At least one start URL is required")
        if not (options.get("allowed_domains") or options.get("domains")):
            raise ValueError("WebsiteConnector requires explicit allowed_domains")
        super().__init__(urls[0], **options)
        self._urls = list(dict.fromkeys(_normalize(url) for url in urls))
        self.max_urls = _positive(options.get("max_urls", 50), "max_urls")
        self.max_queue = _positive(options.get("max_queue", 100), "max_queue")
        depth = options.get("max_depth", 2)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.max_depth = depth
        if len(self._urls) > min(self.max_urls, self.max_queue):
            raise ValueError("Start URLs exceed crawl limits")
        if any(not self._domain_allowed(url) for url in self._urls):
            raise ValueError("Start URL is outside allowed_domains")

    @property
    def id(self) -> str:
        return "website"

    @property
    def start_urls(self) -> list[str]:
        return self.paths

    @property
    def options(self) -> dict[str, Any]:
        return {
            **super().options,
            "max_urls": self.max_urls,
            "max_queue": self.max_queue,
            "max_depth": self.max_depth,
        }

    def _initial_state(self) -> dict[str, Any]:
        return {
            **super()._initial_state(),
            "queue": deque((url, 0) for url in self._urls),
            "known": set(self._urls),
            "visited": set(),
            "truncated": False,
        }

    async def _read_batch(self, state: dict[str, Any]) -> list[RawDocument]:
        if len(state["visited"]) >= self.max_urls:
            raise _Failure("max_urls")
        url, depth = state["queue"].popleft()
        state["visited"].add(url)
        response = await self._fetch(url)
        final_url = _normalize(str(response.url))
        state["known"].add(final_url)
        docs = list(self._extract(response, depth=depth))
        parser = _Links(self.max_queue + self.max_urls)
        if "html" in response.headers.get("content-type", "").lower():
            parser.feed(response.text)
        state["truncated"] |= parser.truncated
        for href in parser.links:
            try:
                link = _normalize(urljoin(final_url, href))
            except ValueError:
                state["truncated"] = True
                continue
            if not self._domain_allowed(link) or link in state["known"]:
                continue
            if (
                depth >= self.max_depth
                or len(state["known"]) >= self.max_urls
                or len(state["queue"]) >= self.max_queue
            ):
                state["truncated"] = True
                continue
            state["known"].add(link)
            state["queue"].append((link, depth + 1))
        state["done"] = not state["queue"]
        if state["done"] and state["truncated"]:
            state["terminal_error"] = "crawl_limit"
        return docs


HttpConnector = HttpApiConnector
ApiConnector = HttpApiConnector
GenericHttpConnector = HttpApiConnector
