"""Real LiteLLM/OpenAI HTTP clients against a controlled local protocol fixture.

This tests cancellation, socket cleanup and client ownership. The fixture's
vectors are synthetic; it does not attest a live model or provider credentials.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("litellm")
from httpx import AsyncClient  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from ragnor import RagnorBuilder  # noqa: E402
from ragnor.embeddings.litellm import LitellmEmbeddingFunction  # noqa: E402
from ragnor.exceptions import BackendUnavailableError, OperationTimeoutError  # noqa: E402

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def embedding_endpoint(*, reset=False):
    entered, release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    handlers = set()
    inputs = []

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        eof = ready = None
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            request = json.loads(await reader.readexactly(length))
            inputs.append(request["input"])
            entered.set()
            if reset:
                writer.transport.abort()
                return
            eof = asyncio.create_task(reader.read())
            ready = asyncio.create_task(release.wait())
            done, _ = await asyncio.wait({eof, ready}, return_when=asyncio.FIRST_COMPLETED)
            if eof in done:
                disconnected.set()
                return
            response = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                        for i, _ in enumerate(request["input"])
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                + f"Content-Length: {len(response)}\r\n\r\n".encode()
                + response
            )
            await writer.drain()
        finally:
            for pending in (eof, ready):
                if pending is not None:
                    pending.cancel()
            await asyncio.gather(
                *(p for p in (eof, ready) if p is not None), return_exceptions=True
            )
            writer.close()
            await writer.wait_closed()
            handlers.remove(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", entered, release, disconnected, inputs
    finally:
        # Release an old blocking implementation too, so a failing regression
        # does not strand its executor thread while the event loop shuts down.
        release.set()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*tuple(handlers)), 5)


async def test_cancelled_embedding_startup_closes_the_inflight_provider_socket():
    async with embedding_endpoint() as (url, entered, _, disconnected, _):
        embedder = LitellmEmbeddingFunction(
            api_key="test-key", model_name="text-embedding-3-small", api_base=url
        )

        async def ready():
            await embedder.embed_query("readiness")

        task = asyncio.create_task(
            RagnorBuilder.local(embedder, embedding_dim=2).with_startup_check(ready).build()
        )
        try:
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
            await asyncio.wait_for(disconnected.wait(), 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("owned", [False, True])
async def test_startup_deadline_releases_request_and_respects_real_client_ownership(owned):
    async with embedding_endpoint() as (url, entered, release, disconnected, inputs):
        async with AsyncClient() as http:
            api = AsyncOpenAI(api_key="test-key", base_url=url, http_client=http, max_retries=0)
            embedder = LitellmEmbeddingFunction(
                api_key="test-key",
                model_name="text-embedding-3-small",
                api_base=url,
                client=api,
                cache_max_entries=4,
            )

            async def ready():
                await embedder.embed_query("readiness")

            builder = RagnorBuilder.local(embedder, embedding_dim=2).with_startup_check(
                ready, timeout_seconds=0.5
            )
            if owned:
                builder.with_owned_resource(http)
            with pytest.raises(OperationTimeoutError):
                await builder.build()
            assert entered.is_set()
            await asyncio.wait_for(disconnected.wait(), 2)
            assert http.is_closed is owned
            if not owned:
                release.set()
                # A cancelled cache miss must be fetched again, then cached.
                assert await embedder.embed_query("readiness") == [1.0, 0.0]
                assert await embedder.embed_query("readiness") == [1.0, 0.0]
                assert len(inputs) == 2


async def test_real_async_client_preserves_ordered_batches_and_cache_hits():
    async with embedding_endpoint() as (url, _, release, _, inputs):
        release.set()
        async with AsyncOpenAI(api_key="test-key", base_url=url, max_retries=0) as api:
            embedder = LitellmEmbeddingFunction(
                api_key="test-key",
                model_name="text-embedding-3-small",
                api_base=url,
                client=api,
                max_batch_size=1,
                cache_max_entries=4,
            )
            assert await embedder.embed_documents(["first", "second"]) == [[1.0, 0.0], [1.0, 0.0]]
            assert await embedder.embed_documents(["second", "first"]) == [[1.0, 0.0], [1.0, 0.0]]
            assert len(inputs) == 2
            assert inputs == [["first"], ["second"]]


async def test_reset_provider_connection_rejects_startup_without_closing_borrowed_client():
    async with embedding_endpoint(reset=True) as (url, entered, _, _, _):
        async with AsyncOpenAI(api_key="test-key", base_url=url, max_retries=0) as api:
            embedder = LitellmEmbeddingFunction(
                api_key="test-key",
                model_name="text-embedding-3-small",
                api_base=url,
                client=api,
            )

            async def ready():
                await embedder.embed_query("readiness")

            with pytest.raises(BackendUnavailableError):
                await (
                    RagnorBuilder.local(embedder, embedding_dim=2).with_startup_check(ready).build()
                )
            assert entered.is_set()
            assert not api.is_closed()
