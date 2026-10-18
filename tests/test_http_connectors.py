"""HTTP connector tests use only mock transports and local fetcher doubles."""

import asyncio
import json
import pickle
import socket
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from ragnor.ingestion.connector_core import SourceConnector, SourceStatus
from ragnor.ingestion.http_connector import (
    ApiConnector,
    Credential,
    GenericHttpConnector,
    HttpApiConnector,
    HttpConnector,
    WebsiteConnector,
)
from ragnor.ingestion.sources import RawDocument

httpx = pytest.importorskip("httpx")


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)


def connector(handler, **options):
    return HttpApiConnector(
        "https://example.com/root", transport=httpx.MockTransport(handler), **options
    )


def text_response(request):
    return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})


def test_aliases_protocol_and_credentials():
    assert HttpConnector is ApiConnector is GenericHttpConnector is HttpApiConnector
    assert isinstance(connector(text_response), SourceConnector)
    credential = Credential(Authorization="Bearer secret")
    headers = {}
    credential.apply(headers)
    assert headers["Authorization"] == "Bearer secret"
    assert "secret" not in repr(credential)
    assert credential.fingerprint == {}
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(credential)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.com",
        "https://user@example.com",
        "file:///tmp/test",
        "https://example.com/?api_key=secret",
        "https://example.com/?token=secret",
        "https://example.com/?X-Amz-Signature=secret",
        "https://example.com/?sig=secret",
        "https://example.com/?%61uthorization=secret",
        "https://example.com/\\evil",
    ],
)
def test_unsafe_urls_rejected(url):
    with pytest.raises(ValueError):
        HttpApiConnector(url)


@pytest.mark.parametrize("option", ["private_network_protection", "redirect_check"])
def test_protections_cannot_be_disabled(option):
    with pytest.raises(ValueError):
        HttpApiConnector("https://example.com", **{option: False})


@pytest.mark.parametrize(
    "option", ["max_pages", "max_items", "max_requests", "page_size", "max_response_bytes"]
)
def test_positive_limits_required(option):
    with pytest.raises(ValueError):
        HttpApiConnector("https://example.com", **{option: 0})


async def test_pinned_connection_retains_host_and_tls_sni(monkeypatch):
    resolutions = []
    requests = []

    def rebind(host, port, **kwargs):
        resolutions.append(host)
        address = "93.184.216.34" if len(resolutions) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", rebind)

    def handle(request):
        requests.append(request)
        return text_response(request)

    result = await connector(handle).load()
    assert result.complete and result.authoritative
    assert len(resolutions) == len(requests) == 1
    assert requests[0].url.host == "93.184.216.34"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].extensions["sni_hostname"] == b"example.com"
    assert result[0].metadata["url"] == "https://example.com/root"
    assert "93.184" not in repr(result[0])


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.1.1.1", "169.254.169.254", "::1", "100.64.0.1", "0.0.0.0"],  # noqa: S104
)
async def test_private_literal_never_reaches_transport(address):
    requests = []
    host = f"[{address}]" if ":" in address else address
    c = HttpApiConnector(f"http://{host}", transport=httpx.MockTransport(requests.append))
    result = await c.load_page()
    assert result.status == SourceStatus.INACCESSIBLE
    assert not result.complete and not requests
    assert result.errors[0]["reason"] == "private_address"


async def test_mixed_dns_answers_are_rejected(monkeypatch):
    def resolve(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
            for ip in ["93.184.216.34", "10.0.0.1"]
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    result = await connector(lambda r: pytest.fail("No network permitted")).load_page()
    assert result.errors[0]["reason"] == "private_address"


async def test_authenticated_cross_origin_redirect_does_not_leak():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://other.example/path"})

    c = connector(handle, credentials=Credential(Authorization="Bearer secret"))
    result = await c.load_page()
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert result.errors[0]["reason"] == "authenticated_cross_origin_redirect"
    assert not result.complete and not result.skipped
    assert "secret" not in repr(result.errors) + repr(c.options)


@pytest.mark.parametrize(
    "location,reason",
    [
        ("http://example.com/path", "https_downgrade"),
        ("https://127.0.0.1/", "private_address"),
        ("https://example.com/?token=secret", "request_failed"),
    ],
)
async def test_every_redirect_is_validated(location, reason):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    result = await connector(handle).load_page()
    assert len(requests) == 1
    assert not result.complete and result.errors[0]["reason"] == reason
    assert "secret" not in repr(result.errors)


async def test_redirect_dns_is_revalidated(monkeypatch):
    calls = []

    def resolve(host, port, **kwargs):
        calls.append(host)
        address = "93.184.216.34" if len(calls) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "/next"})

    result = await connector(handle).load_page()
    assert len(requests) == 1 and len(calls) == 2
    assert result.errors[0]["reason"] == "private_address"


@pytest.mark.parametrize(
    "code,status,reason",
    [
        (404, SourceStatus.NOT_FOUND, "not_found"),
        (401, SourceStatus.INACCESSIBLE, "inaccessible"),
        (403, SourceStatus.INACCESSIBLE, "inaccessible"),
        (429, SourceStatus.TRANSIENT, "rate_limited"),
        (503, SourceStatus.TRANSIENT, "http_transient"),
    ],
)
async def test_http_statuses_are_structured_failures(code, status, reason):
    result = await connector(
        lambda r: httpx.Response(code, headers={"retry-after": "12"})
    ).load_page()
    assert result.status == status
    assert result.errors[0]["reason"] == reason
    assert not result.complete and not result.authoritative and not result.skipped
    if code == 429:
        assert result.errors[0]["retry_after"] == 12


async def test_retry_after_http_date():
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    result = await connector(lambda r: httpx.Response(429, headers={"retry-after": date})).load()
    assert 58 <= result.errors[0]["retry_after"] <= 60
    assert not result.complete


async def test_empty_success_is_distinct():
    result = await connector(lambda r: httpx.Response(204)).load_page()
    assert not result and result.status == SourceStatus.EMPTY
    assert result.complete and result.authoritative and not result.errors


async def test_timeout_does_not_leak_exception_url():
    def handle(request):
        raise httpx.ReadTimeout("secret https://example.com/?q=sensitive", request=request)

    result = await connector(handle).load()
    assert not result.complete and result.errors
    assert "sensitive" not in repr(result.errors)
    assert "ReadTimeout" in repr(result.errors)


@pytest.mark.parametrize("advertise", [False, True])
async def test_response_bytes_bounded(advertise):
    headers = {"content-length": "100"} if advertise else {}
    result = await connector(
        lambda r: httpx.Response(200, content=b"x" * 100, headers=headers), max_response_bytes=10
    ).load_page()
    assert result.errors[0]["reason"] == "response_too_large"
    assert not result.complete


async def test_stream_consumption_stops_at_byte_cap():
    reads = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(50):
                reads.append(1)
                yield b"x" * 65536

    result = await connector(
        lambda r: httpx.Response(200, stream=Stream()), max_response_bytes=70000
    ).load_page()
    assert result.errors[0]["reason"] == "response_too_large"
    assert len(reads) == 2


async def test_protocol_discovery_load_and_limit_cursor():
    calls = []

    async def fetch(self, url, cursor):
        calls.append((url, cursor))
        if cursor is None:
            return [RawDocument("a", "A"), RawDocument("b", "B")], "next", True
        return [RawDocument("c", "C")], None, False

    c = HttpApiConnector("https://example.com", page_fetcher=fetch)
    first = await c.discover(limit=1)
    assert len(first) == 1 and first.has_more and not first.complete
    assert first[0].external_id is None
    second = await c.load_page(cursor=first.next_cursor, limit=1)
    assert [doc.content for doc in second] == ["B"]
    assert second.has_more and len(calls) == 1
    third = await c.load_page(cursor=second.next_cursor, limit=1)
    assert [doc.content for doc in third] == ["C"]
    assert not third.has_more and not third.complete
    all_docs = await c.load()
    assert [doc.content for doc in all_docs] == ["A", "B", "C"]
    assert all_docs.complete and all_docs.authoritative


@pytest.mark.parametrize(
    "token,more", [(None, True), ("", True), ([], True), ("next", "yes"), ("unused", False)]
)
async def test_bad_upstream_cursors_fail_closed(token, more):
    async def fetch(self, url, cursor):
        return [], token, more

    result = await HttpApiConnector("https://example.com", page_fetcher=fetch).load()
    assert not result.complete and result.errors
    assert not result.skipped


async def test_repeated_cursor_is_not_exhaustion():
    async def fetch(self, url, cursor):
        return [RawDocument("a", "A")], "repeat", True

    result = await HttpApiConnector("https://example.com", page_fetcher=fetch).load()
    assert not result.complete
    assert result.errors[0]["reason"] == "invalid_or_repeated_cursor"


@pytest.mark.parametrize("options", [{"max_pages": 1}, {"max_items": 1}])
async def test_exact_limits_are_not_authoritative(options):
    async def fetch(self, url, cursor):
        return [RawDocument("a", "A")], None, False

    result = await HttpApiConnector("https://example.com", page_fetcher=fetch, **options).load()
    assert len(result) == 1
    assert not result.complete and result.errors


async def test_all_endpoints_and_inaccessible_endpoint():
    calls = []

    async def fetch(self, url, cursor):
        calls.append(url)
        if url.endswith("bad"):
            raise RuntimeError("secret")
        return [RawDocument(url, "A")], None, False

    c = HttpApiConnector("https://example.com", endpoints=["good", "bad"], page_fetcher=fetch)
    result = await c.load()
    assert len(calls) == 2 and len(result) == 1
    assert not result.complete and result.errors and not result.skipped
    assert "secret" not in repr(result.errors)


async def test_query_metadata_options_and_identities_redacted_without_hashing():
    c = HttpApiConnector(
        "https://example.com/?q=sensitive",
        paths=["/?q=sensitive", "/?q=other"],
        headers={"X-Custom-Secret": "secret"},
        transport=httpx.MockTransport(text_response),
    )
    result = await c.load()
    assert result.complete and len(result) == 2
    assert result[0].external_id != result[1].external_id
    persisted = repr(c.options) + repr(result)
    assert "sensitive" not in persisted and "other" not in persisted and "secret" not in persisted
    assert "redacted" in persisted


async def test_fetcher_metadata_and_identity_sanitized():
    async def fetch(self, url, cursor):
        return (
            [
                RawDocument(
                    "https://user:secret@example.com/?q=sensitive",
                    "body",
                    metadata={"url": "https://example.com/?q=sensitive", "Authorization": "secret"},
                    provenance={"links": ["https://example.com/?q=sensitive"]},
                )
            ],
            None,
            False,
        )

    result = await HttpApiConnector("https://example.com", page_fetcher=fetch).load()
    assert result.complete
    assert "sensitive" not in repr(result) and "user:secret" not in repr(result)
    assert result[0].metadata["Authorization"] == "[redacted]"


async def test_invalid_local_cursor_never_fetches():
    c = connector(lambda r: pytest.fail("No requests expected"))
    result = await c.load_page(cursor="unknown")
    assert result.errors[0]["reason"] == "invalid_cursor"
    assert not result.complete


def test_website_requires_explicit_scope():
    with pytest.raises(ValueError, match="allowed_domains"):
        WebsiteConnector("https://example.com")


async def test_website_relative_links_use_final_response_url_and_deduplicate():
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/dir/index.html"})
        if request.url.path == "/dir/index.html":
            return httpx.Response(
                200,
                text='<article><h1>Title</h1><a href="child#x">A</a>'
                '<a href="child#y">B</a><a href="https://outside.com">X</a></article>',
                headers={"content-type": "text/html"},
            )
        return text_response(request)

    c = WebsiteConnector(
        "https://example.com/start",
        allowed_domains=["example.com"],
        transport=httpx.MockTransport(handle),
    )
    result = await c.load()
    assert result.complete and result.authoritative
    assert paths == ["/start", "/dir/index.html", "/dir/child"]
    assert result[0].metadata["url"] == "https://example.com/dir/index.html"
    assert len(result) == 2


@pytest.mark.parametrize("limits", [{"max_urls": 1}, {"max_depth": 0}, {"max_queue": 1}])
async def test_website_caps_are_incomplete_and_keep_documents(limits):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        return httpx.Response(
            200,
            text='<article>Text<a href="/a">A</a><a href="/b">B</a></article>',
            headers={"content-type": "text/html"},
        )

    c = WebsiteConnector(
        "https://example.com",
        allowed_domains=["example.com"],
        transport=httpx.MockTransport(handle),
        **limits,
    )
    result = await c.load()
    assert result and not result.complete and not result.authoritative
    assert result.errors
    if "max_urls" in limits or "max_depth" in limits:
        assert len(paths) == 1


async def test_request_cap_includes_redirects():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "/next"})

    result = await connector(handle, max_requests=1).load()
    assert len(calls) == 1
    assert not result.complete and result.errors[0]["reason"] == "max_requests"


async def test_async_fetcher_timeout_is_bounded():
    async def fetch(self, url, cursor):
        await asyncio.Event().wait()

    result = await HttpApiConnector("https://example.com", timeout=0.01, page_fetcher=fetch).load()
    assert not result.complete
    assert result.errors[0]["error_type"] == "TimeoutError"


async def test_cursors_do_not_serialize_upstream_tokens():
    async def fetch(self, url, cursor):
        return [RawDocument("a", "A")], "sensitive-upstream-token", True

    c = HttpApiConnector("https://example.com", page_fetcher=fetch)
    result = await c.load_page()
    assert "sensitive" not in json.dumps({"cursor": result.next_cursor, "options": c.options})


async def test_default_transport_connects_to_pin_and_verifies_original_tls_name(monkeypatch):
    import httpcore
    from httpcore._backends.anyio import AnyIOBackend

    connected = []
    tls_names = []

    class Stream:
        async def start_tls(self, **kwargs):
            tls_names.append(kwargs["server_hostname"])
            raise httpcore.ConnectError("Stop before any live TLS or HTTP")

    async def connect_tcp(self, host, port, **kwargs):
        connected.append((host, port))
        return Stream()

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect_tcp)
    c = HttpApiConnector("https://example.com/path")
    result = await c.load_page()
    assert connected == [("93.184.216.34", 443)]
    assert tls_names == [b"example.com"]
    assert not result.complete and result.errors


def test_connector_credentials_cannot_be_pickled():
    c = HttpApiConnector("https://example.com", headers={"Authorization": "secret"})
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(c)
    assert "secret" not in repr(c)


async def test_compressed_response_cannot_bypass_byte_limit():
    result = await connector(
        lambda r: httpx.Response(
            200,
            headers={"content-encoding": "custom"},
            content=b"opaque",
        )
    ).load_page()
    assert result.errors[0]["reason"] == "unsupported_content_encoding"
    assert not result.complete


async def test_query_free_url_normalization_removes_fragments_and_dot_segments():
    c = HttpApiConnector(
        "https://EXAMPLE.COM:443/a/../b#fragment", transport=httpx.MockTransport(text_response)
    )
    result = await c.load()
    assert result.complete
    assert result[0].metadata["url"] == "https://example.com/b"


async def test_http_incremental_and_changes():
    c = HttpApiConnector("https://example.com/root", transport=httpx.MockTransport(text_response))
    inc = await c.load_incremental(checkpoint="cp-1")
    assert inc.incremental
    assert inc.checkpoint == "cp-1"

    events = [e async for e in c.changes(checkpoint="cp-1")]
    assert len(events) == 1
    assert events[0].type == "sync_requested"
