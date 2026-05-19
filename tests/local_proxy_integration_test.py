from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, TypedDict

import aiohttp
import pytest
from aiohttp import web
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from meshagent.llm_proxy.local_proxy import (
    LocalLLMProxyServer,
    MESHAGENT_PROJECT_ID_HEADER,
)
from meshagent.llm_proxy.proxy import (
    ProxyWebSocketOutcome,
    proxy_websocket_request,
)


class _RecordedRequest(TypedDict):
    provider: str
    headers: dict[str, str]
    payload: dict[str, Any]


async def _start_test_server(
    app: web.Application,
) -> tuple[web.AppRunner, web.TCPSite, str]:
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    assert site._server is not None
    sockets = site._server.sockets
    assert sockets is not None and len(sockets) > 0
    port = sockets[0].getsockname()[1]
    return runner, site, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_local_proxy_start_closes_owned_session_when_site_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _upstream_token_provider() -> str:
        return "meshagent-upstream-token"

    async def _fail_start(self):
        del self
        raise OSError("site start failed")

    monkeypatch.setattr(web.TCPSite, "start", _fail_start)

    proxy_server = LocalLLMProxyServer(
        api_base_url="https://example.test",
        project_id="project-123",
        upstream_bearer_token_provider=_upstream_token_provider,
        host="127.0.0.1",
        port=0,
        bearer_token="local-proxy-bearer",
    )

    with pytest.raises(OSError, match="site start failed"):
        await proxy_server.start()

    assert proxy_server._http_session.closed is True


@pytest.mark.asyncio
async def test_local_proxy_live_forwards_openai_and_anthropic_requests() -> None:
    recorded_requests: list[_RecordedRequest] = []

    upstream_app = web.Application()

    async def _handle_openai_chat_completion(request: web.Request) -> web.Response:
        payload = await request.json()
        recorded_requests.append(
            {
                "provider": "openai",
                "headers": dict(request.headers),
                "payload": payload,
            }
        )
        return web.json_response(
            {
                "id": "chatcmpl-local-proxy",
                "object": "chat.completion",
                "created": 1_744_000_000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                },
            }
        )

    async def _handle_anthropic_messages(request: web.Request) -> web.Response:
        payload = await request.json()
        recorded_requests.append(
            {
                "provider": "anthropic",
                "headers": dict(request.headers),
                "payload": payload,
            }
        )
        return web.json_response(
            {
                "id": "msg_local_proxy",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 13,
                    "output_tokens": 7,
                },
            }
        )

    upstream_app.add_routes(
        [
            web.post("/openai/v1/chat/completions", _handle_openai_chat_completion),
            web.post("/anthropic/v1/messages", _handle_anthropic_messages),
        ]
    )

    upstream_runner, _upstream_site, upstream_base_url = await _start_test_server(
        upstream_app
    )

    async def _upstream_token_provider() -> str:
        return "meshagent-upstream-token"

    proxy_server = LocalLLMProxyServer(
        api_base_url=upstream_base_url,
        project_id="project-123",
        upstream_bearer_token_provider=_upstream_token_provider,
        host="127.0.0.1",
        port=0,
        bearer_token="local-proxy-bearer",
    )
    await proxy_server.start()

    proxy_env = proxy_server.env()
    openai_client = AsyncOpenAI(
        base_url=proxy_env["OPENAI_BASE_URL"],
        api_key=proxy_env["OPENAI_API_KEY"],
    )
    anthropic_client = AsyncAnthropic(
        base_url=proxy_env["ANTHROPIC_BASE_URL"],
        api_key=proxy_env["ANTHROPIC_API_KEY"],
    )

    try:
        openai_response = await openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16,
        )
        anthropic_response = await anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        )

        snapshot = await proxy_server.usage_collector.snapshot()
    finally:
        await openai_client.close()
        await anthropic_client.close()
        await proxy_server.close()
        await upstream_runner.cleanup()

    assert openai_response.choices[0].message.content == "ok"
    assert anthropic_response.content[0].text == "ok"

    assert len(recorded_requests) == 2

    openai_request = next(
        request for request in recorded_requests if request["provider"] == "openai"
    )
    assert openai_request["headers"][MESHAGENT_PROJECT_ID_HEADER] == "project-123"
    assert (
        openai_request["headers"]["Authorization"] == "Bearer meshagent-upstream-token"
    )
    assert openai_request["payload"]["model"] == "gpt-4.1-mini"

    anthropic_request = next(
        request for request in recorded_requests if request["provider"] == "anthropic"
    )
    assert anthropic_request["headers"][MESHAGENT_PROJECT_ID_HEADER] == "project-123"
    assert (
        anthropic_request["headers"]["Authorization"]
        == "Bearer meshagent-upstream-token"
    )
    assert anthropic_request["payload"]["model"] == "claude-sonnet-4-5"

    assert snapshot.total_requests == 2
    assert snapshot.subtotal > 0
    assert snapshot.surcharge > 0
    assert snapshot.total > snapshot.subtotal

    summaries_by_provider = {
        (summary.provider, summary.model): summary for summary in snapshot.summaries
    }
    assert ("openai", "gpt-4.1-mini") in summaries_by_provider
    assert ("anthropic", "claude-sonnet-4-5") in summaries_by_provider

    request_activity_by_provider = {
        event.provider: event for event in snapshot.recent_requests
    }
    assert request_activity_by_provider["openai"].status == 200
    assert request_activity_by_provider["openai"].path == "/openai/v1/chat/completions"
    assert request_activity_by_provider["openai"].total is not None
    assert request_activity_by_provider["anthropic"].status == 200
    assert request_activity_by_provider["anthropic"].path == "/anthropic/v1/messages"
    assert request_activity_by_provider["anthropic"].total is not None


@pytest.mark.asyncio
async def test_websocket_proxy_idle_connection_survives_low_heartbeats() -> None:
    heartbeat = 0.5
    outcomes: list[ProxyWebSocketOutcome] = []
    upstream_ready = asyncio.Event()
    proxy_app = web.Application()
    upstream_app = web.Application()

    async def _handle_upstream_websocket(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse(heartbeat=heartbeat)
        await websocket.prepare(request)
        upstream_ready.set()
        async for msg in websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await websocket.send_bytes(msg.data)
        return websocket

    upstream_app.router.add_get("/messages", _handle_upstream_websocket)
    upstream_runner, _upstream_site, upstream_base_url = await _start_test_server(
        upstream_app
    )
    proxy_session = aiohttp.ClientSession()

    async def _record_outcome(outcome: ProxyWebSocketOutcome) -> None:
        outcomes.append(outcome)

    async def _handle_proxy_websocket(request: web.Request) -> web.StreamResponse:
        return await proxy_websocket_request(
            request=request,
            http_session=proxy_session,
            upstream_http_url=f"{upstream_base_url}/messages",
            heartbeat=heartbeat,
            upstream_headers={},
            on_complete=_record_outcome,
        )

    proxy_app.router.add_get("/messages", _handle_proxy_websocket)
    proxy_runner, _proxy_site, proxy_base_url = await _start_test_server(proxy_app)

    try:
        async with aiohttp.ClientSession() as client_session:
            async with client_session.ws_connect(
                f"{proxy_base_url.replace('http', 'ws')}/messages",
            ) as websocket:
                received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

                async def _receive_messages() -> None:
                    async for message in websocket:
                        await received.put(message)

                receive_task = asyncio.create_task(_receive_messages())
                await asyncio.wait_for(upstream_ready.wait(), timeout=1.0)
                await asyncio.sleep(heartbeat * 4)
                assert not websocket.closed

                await websocket.send_str("still here")
                message = await asyncio.wait_for(received.get(), timeout=1.0)
                assert message.type == aiohttp.WSMsgType.TEXT
                assert message.data == "still here"
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
    finally:
        await proxy_session.close()
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()

    assert outcomes == [] or outcomes[-1].error is None


@pytest.mark.asyncio
async def test_browser_facing_websocket_responds_to_server_heartbeat() -> None:
    heartbeat = 0.2
    app = web.Application()
    server_ready = asyncio.Event()

    async def _handle_websocket(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse(heartbeat=heartbeat)
        await websocket.prepare(request)
        server_ready.set()
        async for msg in websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_str(msg.data)
        return websocket

    app.router.add_get("/messages", _handle_websocket)
    runner, _site, base_url = await _start_test_server(app)

    try:
        async with aiohttp.ClientSession() as client_session:
            async with client_session.ws_connect(
                f"{base_url.replace('http', 'ws')}/messages"
            ) as websocket:
                received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

                async def _receive_messages() -> None:
                    async for message in websocket:
                        await received.put(message)

                receive_task = asyncio.create_task(_receive_messages())
                await asyncio.wait_for(server_ready.wait(), timeout=1.0)
                await asyncio.sleep(heartbeat * 4)
                assert not websocket.closed

                await websocket.send_str("browser-hop")
                message = await asyncio.wait_for(received.get(), timeout=1.0)
                assert message.type == aiohttp.WSMsgType.TEXT
                assert message.data == "browser-hop"
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_router_to_room_pod_websocket_responds_to_client_heartbeat() -> None:
    heartbeat = 0.2
    room_pod_app = web.Application()
    room_pod_ready = asyncio.Event()

    async def _handle_room_pod_websocket(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        room_pod_ready.set()
        async for msg in websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_str(msg.data)
        return websocket

    room_pod_app.router.add_get("/rooms/test-room", _handle_room_pod_websocket)
    runner, _site, base_url = await _start_test_server(room_pod_app)

    try:
        async with aiohttp.ClientSession() as router_session:
            async with router_session.ws_connect(
                f"{base_url.replace('http', 'ws')}/rooms/test-room",
                heartbeat=heartbeat,
            ) as websocket:
                received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

                async def _receive_messages() -> None:
                    async for message in websocket:
                        await received.put(message)

                receive_task = asyncio.create_task(_receive_messages())
                await asyncio.wait_for(room_pod_ready.wait(), timeout=1.0)
                await asyncio.sleep(heartbeat * 4)
                assert not websocket.closed

                await websocket.send_str("room-pod-hop")
                message = await asyncio.wait_for(received.get(), timeout=1.0)
                assert message.type == aiohttp.WSMsgType.TEXT
                assert message.data == "room-pod-hop"
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_room_pod_to_container_proxy_responds_to_proxy_heartbeats() -> None:
    heartbeat = 0.2
    outcomes: list[ProxyWebSocketOutcome] = []
    container_ready = asyncio.Event()
    container_app = web.Application()
    room_pod_proxy_app = web.Application()

    async def _handle_container_websocket(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        container_ready.set()
        async for msg in websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await websocket.send_bytes(msg.data)
        return websocket

    container_app.router.add_get("/messages", _handle_container_websocket)
    container_runner, _container_site, container_base_url = await _start_test_server(
        container_app
    )
    room_pod_session = aiohttp.ClientSession()

    async def _record_outcome(outcome: ProxyWebSocketOutcome) -> None:
        outcomes.append(outcome)

    async def _handle_room_pod_proxy(request: web.Request) -> web.StreamResponse:
        return await proxy_websocket_request(
            request=request,
            http_session=room_pod_session,
            upstream_http_url=f"{container_base_url}/messages",
            heartbeat=heartbeat,
            upstream_headers={},
            on_complete=_record_outcome,
        )

    room_pod_proxy_app.router.add_get(
        "/tunnel/container-1/3001/messages", _handle_room_pod_proxy
    )
    proxy_runner, _proxy_site, proxy_base_url = await _start_test_server(
        room_pod_proxy_app
    )

    try:
        async with aiohttp.ClientSession() as browser_session:
            async with browser_session.ws_connect(
                f"{proxy_base_url.replace('http', 'ws')}"
                "/tunnel/container-1/3001/messages"
            ) as websocket:
                received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

                async def _receive_messages() -> None:
                    async for message in websocket:
                        await received.put(message)

                receive_task = asyncio.create_task(_receive_messages())
                await asyncio.wait_for(container_ready.wait(), timeout=1.0)
                await asyncio.sleep(heartbeat * 4)
                assert not websocket.closed

                await websocket.send_str("container-hop")
                message = await asyncio.wait_for(received.get(), timeout=1.0)
                assert message.type == aiohttp.WSMsgType.TEXT
                assert message.data == "container-hop"
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
    finally:
        await room_pod_session.close()
        await proxy_runner.cleanup()
        await container_runner.cleanup()

    assert outcomes == [] or outcomes[-1].error is None


@pytest.mark.asyncio
async def test_chained_websocket_proxies_idle_connection_survives_low_heartbeats() -> (
    None
):
    heartbeat = 0.5
    outcomes: list[tuple[str, ProxyWebSocketOutcome]] = []
    upstream_ready = asyncio.Event()
    upstream_app = web.Application()
    inner_proxy_app = web.Application()
    outer_proxy_app = web.Application()

    async def _handle_upstream_websocket(request: web.Request) -> web.StreamResponse:
        websocket = web.WebSocketResponse(heartbeat=heartbeat)
        await websocket.prepare(request)
        upstream_ready.set()
        async for msg in websocket:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await websocket.send_bytes(msg.data)
        return websocket

    upstream_app.router.add_get("/messages", _handle_upstream_websocket)
    upstream_runner, _upstream_site, upstream_base_url = await _start_test_server(
        upstream_app
    )

    inner_session = aiohttp.ClientSession()
    outer_session = aiohttp.ClientSession()

    async def _record_inner_outcome(outcome: ProxyWebSocketOutcome) -> None:
        outcomes.append(("inner", outcome))

    async def _record_outer_outcome(outcome: ProxyWebSocketOutcome) -> None:
        outcomes.append(("outer", outcome))

    async def _handle_inner_proxy(request: web.Request) -> web.StreamResponse:
        return await proxy_websocket_request(
            request=request,
            http_session=inner_session,
            upstream_http_url=f"{upstream_base_url}/messages",
            heartbeat=heartbeat,
            upstream_headers={},
            on_complete=_record_inner_outcome,
        )

    inner_proxy_app.router.add_get(
        "/projects/project/rooms/room/ports/3001/messages", _handle_inner_proxy
    )
    inner_runner, _inner_site, inner_base_url = await _start_test_server(
        inner_proxy_app
    )

    async def _handle_outer_proxy(request: web.Request) -> web.StreamResponse:
        return await proxy_websocket_request(
            request=request,
            http_session=outer_session,
            upstream_http_url=(
                f"{inner_base_url}/projects/project/rooms/room/ports/3001/messages"
            ),
            heartbeat=heartbeat,
            upstream_headers={},
            on_complete=_record_outer_outcome,
        )

    outer_proxy_app.router.add_get("/messages", _handle_outer_proxy)
    outer_runner, _outer_site, outer_base_url = await _start_test_server(
        outer_proxy_app
    )

    try:
        async with aiohttp.ClientSession() as client_session:
            async with client_session.ws_connect(
                f"{outer_base_url.replace('http', 'ws')}/messages",
            ) as websocket:
                received: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

                async def _receive_messages() -> None:
                    async for message in websocket:
                        await received.put(message)

                receive_task = asyncio.create_task(_receive_messages())
                await asyncio.wait_for(upstream_ready.wait(), timeout=1.0)
                await asyncio.sleep(heartbeat * 4)
                assert not websocket.closed

                await websocket.send_str("still here")
                message = await asyncio.wait_for(received.get(), timeout=1.0)
                assert message.type == aiohttp.WSMsgType.TEXT
                assert message.data == "still here"
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
    finally:
        await outer_session.close()
        await inner_session.close()
        await outer_runner.cleanup()
        await inner_runner.cleanup()
        await upstream_runner.cleanup()

    assert all(outcome.error is None for _, outcome in outcomes)


@pytest.mark.asyncio
async def test_local_proxy_records_http_error_activity() -> None:
    upstream_app = web.Application()

    async def _handle_openai_chat_completion(request: web.Request) -> web.Response:
        del request
        return web.Response(status=403, text="Invalid token.")

    upstream_app.add_routes(
        [
            web.post("/openai/v1/chat/completions", _handle_openai_chat_completion),
        ]
    )

    upstream_runner, _upstream_site, upstream_base_url = await _start_test_server(
        upstream_app
    )

    async def _upstream_token_provider() -> str:
        return "meshagent-upstream-token"

    proxy_server = LocalLLMProxyServer(
        api_base_url=upstream_base_url,
        project_id="project-123",
        upstream_bearer_token_provider=_upstream_token_provider,
        host="127.0.0.1",
        port=0,
        bearer_token="local-proxy-bearer",
    )
    await proxy_server.start()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{proxy_server.base_url}/openai/v1/chat/completions",
                headers={
                    "Authorization": "Bearer local-proxy-bearer",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                    "max_tokens": 16,
                },
            ) as response:
                response_status = response.status
                response_text = await response.text()

        snapshot = await proxy_server.usage_collector.snapshot()
    finally:
        await proxy_server.close()
        await upstream_runner.cleanup()

    assert response_status == 403
    assert response_text == "Invalid token."
    assert snapshot.total_requests == 0
    assert len(snapshot.recent_requests) == 1
    assert snapshot.recent_requests[0].provider == "openai"
    assert snapshot.recent_requests[0].transport == "http"
    assert snapshot.recent_requests[0].method == "POST"
    assert snapshot.recent_requests[0].path == "/openai/v1/chat/completions"
    assert snapshot.recent_requests[0].status == 403
    assert snapshot.recent_requests[0].result == "Invalid token."
    assert snapshot.recent_requests[0].total is None


@pytest.mark.asyncio
async def test_local_proxy_records_openai_websocket_activity_and_usage() -> None:
    recorded_headers: dict[str, str] | None = None

    upstream_app = web.Application()

    async def _handle_openai_realtime(request: web.Request) -> web.StreamResponse:
        nonlocal recorded_headers
        recorded_headers = dict(request.headers)

        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        await websocket.send_str(
            json.dumps(
                {
                    "event_id": "evt-1",
                    "type": "response.done",
                    "response": {
                        "id": "resp-1",
                        "model": "gpt-4.1-mini",
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 3,
                        },
                    },
                }
            )
        )
        await websocket.close(code=1000)
        return websocket

    upstream_app.add_routes(
        [
            web.get("/openai/v1/realtime", _handle_openai_realtime),
        ]
    )

    upstream_runner, _upstream_site, upstream_base_url = await _start_test_server(
        upstream_app
    )

    async def _upstream_token_provider() -> str:
        return "meshagent-upstream-token"

    proxy_server = LocalLLMProxyServer(
        api_base_url=upstream_base_url,
        project_id="project-123",
        upstream_bearer_token_provider=_upstream_token_provider,
        host="127.0.0.1",
        port=0,
        bearer_token="local-proxy-bearer",
    )
    await proxy_server.start()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                (f"{proxy_server.base_url}/openai/v1/realtime?model=gpt-4.1-mini"),
                headers={"Authorization": "Bearer local-proxy-bearer"},
            ) as websocket:
                message = await websocket.receive()

        snapshot = await proxy_server.usage_collector.snapshot()
    finally:
        await proxy_server.close()
        await upstream_runner.cleanup()

    assert message.type == aiohttp.WSMsgType.TEXT
    payload = json.loads(message.data)
    assert payload["type"] == "response.done"
    assert recorded_headers is not None
    assert recorded_headers[MESHAGENT_PROJECT_ID_HEADER] == "project-123"
    assert recorded_headers["Authorization"] == "Bearer meshagent-upstream-token"

    assert snapshot.total_requests == 1
    assert len(snapshot.recent_requests) == 1
    assert snapshot.recent_requests[0].provider == "openai"
    assert snapshot.recent_requests[0].transport == "websocket"
    assert snapshot.recent_requests[0].path == "/openai/v1/realtime"
    assert snapshot.recent_requests[0].status == 101
    assert snapshot.recent_requests[0].result.startswith("closed")
    assert snapshot.recent_requests[0].total is not None
