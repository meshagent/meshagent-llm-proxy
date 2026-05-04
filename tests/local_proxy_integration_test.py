from __future__ import annotations

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
