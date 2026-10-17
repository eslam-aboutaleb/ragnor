"""Object-storage acceptance tests without a cloud SDK or network."""

from __future__ import annotations

import io

import pytest

from ragit.ingestion.connector_core import SourceStatus
from ragit.ingestion.object_connector import ObjectStorageConnector


class Client:
    def __init__(self, pages=None, objects=None):
        self.pages = pages or {None: {"Contents": [], "IsTruncated": False}}
        self.objects = objects or {}
        self.bodies = []
        self.calls = []

    def list_objects_v2(self, **kwargs):
        self.calls.append(kwargs)
        response = self.pages[kwargs.get("ContinuationToken")]
        if isinstance(response, Exception):
            raise response
        return response

    def get_object(self, **kwargs):
        value = self.objects[kwargs["Key"]]
        if isinstance(value, Exception):
            raise value
        body = io.BytesIO(value)
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(value), "ContentType": "text/plain"}


class SDKError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


async def test_successful_paginated_listing_closes_bodies():
    client = Client(
        pages={
            None: {
                "Contents": [{"Key": "docs/a.txt", "Size": 1}],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            },
            "next": {"Contents": [{"Key": "docs/b.txt", "Size": 1}], "IsTruncated": False},
        },
        objects={"docs/a.txt": b"a", "docs/b.txt": b"b"},
    )
    connector = ObjectStorageConnector("bucket", prefix="docs/", client=client)
    result = await connector.load()
    assert result.authoritative
    assert [doc.source_name for doc in result] == ["a.txt", "b.txt"]
    assert all(body.closed for body in client.bodies)
    assert result[0].provenance["bucket"] == "bucket"


@pytest.mark.parametrize(
    "response",
    [
        {"Contents": [], "IsTruncated": True},
        {"Contents": [], "IsTruncated": True, "NextContinuationToken": "repeat"},
    ],
)
async def test_missing_and_repeated_tokens_are_not_authoritative(response):
    client = Client(pages={None: response, "repeat": response})
    result = await ObjectStorageConnector("bucket", client=client).load()
    assert not result.authoritative
    assert not result.complete
    assert result.errors


async def test_interruption_preserves_continuation():
    client = Client(
        pages={
            None: {
                "Contents": [{"Key": "a.txt", "Size": 1}],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            }
        },
        objects={"a.txt": b"a"},
    )
    result = await ObjectStorageConnector("bucket", client=client).load_all(max_pages=1)
    assert len(result) == 1
    assert not result.authoritative
    assert result.next_cursor == "next"


@pytest.mark.parametrize(
    "code,status",
    [
        ("NoSuchBucket", SourceStatus.NOT_FOUND),
        ("AccessDenied", SourceStatus.INACCESSIBLE),
        ("SlowDown", SourceStatus.TRANSIENT),
    ],
)
async def test_distinct_listing_failures(code, status):
    result = await ObjectStorageConnector(
        "bucket", client=Client(pages={None: SDKError(code)})
    ).load()
    assert result.status == status
    assert not result.authoritative


async def test_empty_scan_is_authoritative():
    result = await ObjectStorageConnector("bucket", client=Client()).load()
    assert not result
    assert result.authoritative
    assert result.status == SourceStatus.EMPTY


async def test_object_failure_is_not_authoritative_or_leaked():
    client = Client(
        pages={None: {"Contents": [{"Key": "a.txt", "Size": 3}]}},
        objects={"a.txt": OSError("credential-secret")},
    )
    result = await ObjectStorageConnector("bucket", client=client).load()
    assert not result.authoritative
    assert result.errors
    assert "credential-secret" not in repr(result.errors)


async def test_growth_after_listing_is_bounded_and_reported():
    client = Client(
        pages={None: {"Contents": [{"Key": "a.txt", "Size": 1}]}}, objects={"a.txt": b"x" * 100}
    )
    result = await ObjectStorageConnector("bucket", client=client, max_bytes=10).load()
    assert not result
    assert result.skipped[0].reason == "oversized"
    assert all(body.closed for body in client.bodies)


def test_credentials_not_in_options():
    connector = ObjectStorageConnector("bucket", credentials={"secret": "never-serialize"})
    assert "never-serialize" not in repr(connector.options)


async def test_object_storage_load_incremental():
    client = Client(
        pages={
            None: {
                "Contents": [
                    {"Key": "old.txt", "Size": 3, "LastModified": 100.0},
                    {"Key": "new.txt", "Size": 3, "LastModified": 200.0},
                ],
                "IsTruncated": False,
            }
        },
        objects={"old.txt": b"old", "new.txt": b"new"},
    )
    connector = ObjectStorageConnector("bucket", client=client)
    res1 = await connector.load_incremental(checkpoint=None)
    assert res1.incremental
    assert len(res1) == 2
    assert res1.checkpoint == "200.0"

    res2 = await connector.load_incremental(checkpoint=150.0)
    assert res2.incremental
    assert len(res2) == 1
    assert res2[0].source_name == "new.txt"
    assert any(s.reason == "unmodified" for s in res2.skipped)
