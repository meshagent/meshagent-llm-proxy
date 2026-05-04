from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import Any, Literal
from uuid import uuid4

import aiohttp
from aiohttp import web
from meshagent.api.http import new_client_session

from meshagent.llm_proxy.providers import (
    is_anthropic_path_allowed,
    is_openai_path_allowed,
    is_openai_websocket_path_allowed,
)
from meshagent.llm_proxy.proxy import (
    ProxyWebSocketOutcome,
    filter_proxied_response_headers,
    proxy_websocket_request,
)
from meshagent.llm_proxy.sse import SSEEvent
from meshagent.llm_proxy.usage import (
    UsageCollector,
    UsageEvent,
    extract_anthropic_completion_usage,
    extract_openai_audio_speech_usage,
    extract_openai_completion_usage,
    extract_openai_image_usage,
    extract_openai_realtime_usage,
    extract_openai_transcription_model_from_session,
    merge_cumulative_usage,
)


logger = logging.getLogger("meshagent.llm_proxy.local_proxy")

MESHAGENT_PROJECT_ID_HEADER = "Meshagent-Project-Id"
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 8766

UpstreamBearerTokenProvider = Callable[[], Awaitable[str | None]]


def is_websocket_request(request: web.Request) -> bool:
    upgrade = request.headers.get("Upgrade", "")
    connection = request.headers.get("Connection", "")
    return upgrade.lower() == "websocket" and "upgrade" in connection.lower()


def _resolve_client_host(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def build_local_proxy_env(
    *,
    base_url: str,
    bearer_token: str | None,
    insecure: bool,
) -> dict[str, str]:
    del insecure
    api_key = bearer_token if bearer_token is not None else "meshagent-insecure"
    normalized_base_url = base_url.rstrip("/")
    return {
        "OPENAI_BASE_URL": f"{normalized_base_url}/openai/v1",
        "OPENAI_API_KEY": api_key,
        "ANTHROPIC_BASE_URL": f"{normalized_base_url}/anthropic",
        "ANTHROPIC_API_KEY": api_key,
    }


def _normalize_activity_text(text: str | None, *, fallback: str) -> str:
    if not isinstance(text, str):
        return fallback

    normalized = " ".join(text.split())
    if normalized == "":
        return fallback
    if len(normalized) <= 180:
        return normalized
    return normalized[:177] + "..."


def _extract_error_text_from_json(payload: object) -> str | None:
    if isinstance(payload, dict):
        nested_error = payload.get("error")
        if nested_error is not None:
            nested_text = _extract_error_text_from_json(nested_error)
            if nested_text is not None:
                return nested_text

        for key in ("message", "detail", "title", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()

    if isinstance(payload, str) and payload.strip() != "":
        return payload.strip()

    return None


def _decode_error_body(body: bytes) -> str | None:
    if len(body) == 0:
        return None
    decoded = body.decode("utf-8", errors="replace").strip()
    return decoded or None


def _status_fallback(status: int, *, failed: bool) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "request failed" if failed else "request completed"


def _summarize_http_result(
    *,
    status: int,
    json_payload: object,
    body: bytes,
    stream: bool,
) -> str:
    if status >= 400:
        error_text = _extract_error_text_from_json(json_payload)
        if error_text is None:
            error_text = _decode_error_body(body)
        return _normalize_activity_text(
            error_text,
            fallback=_status_fallback(status, failed=True),
        )

    if stream:
        return "stream completed"

    return "ok"


def _summarize_websocket_result(outcome: ProxyWebSocketOutcome) -> str:
    if outcome.error is not None:
        return _normalize_activity_text(outcome.error, fallback="websocket failed")

    if (
        outcome.client_close_code is not None
        and outcome.upstream_close_code is not None
    ):
        if outcome.client_close_code == outcome.upstream_close_code:
            return f"closed code={outcome.client_close_code}"
        return (
            "closed "
            f"client={outcome.client_close_code} "
            f"upstream={outcome.upstream_close_code}"
        )

    if outcome.upstream_close_code is not None:
        return f"closed upstream={outcome.upstream_close_code}"
    if outcome.client_close_code is not None:
        return f"closed client={outcome.client_close_code}"
    return "closed"


class LocalLLMProxyServer:
    def __init__(
        self,
        *,
        api_base_url: str,
        project_id: str,
        upstream_bearer_token_provider: UpstreamBearerTokenProvider,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        bearer_token: str | None = None,
        insecure: bool = False,
        session: aiohttp.ClientSession | None = None,
        websocket_heartbeat: float | None = None,
    ) -> None:
        if not insecure and (bearer_token is None or bearer_token.strip() == ""):
            raise ValueError("bearer_token is required unless insecure is enabled")

        self._api_base_url = api_base_url.rstrip("/")
        self._project_id = project_id
        self._upstream_bearer_token_provider = upstream_bearer_token_provider
        self._host = host
        self._port = port
        self._bearer_token = (
            bearer_token.strip() if isinstance(bearer_token, str) else None
        )
        self._insecure = insecure
        self._websocket_heartbeat = websocket_heartbeat
        self._usage_collector = UsageCollector()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._owns_session = session is None
        self._http_session = session or new_client_session()
        self._bound_host = host
        self._bound_port = port

        self._app = web.Application(client_max_size=1024 * 1024 * 1024)
        self._app.add_routes(
            [
                web.get("/healthz", self.healthz),
                web.get("/openai/v1/{openai_path:.*}", self.openai_router),
                web.post("/openai/v1/{openai_path:.*}", self.openai_router),
                web.put("/openai/v1/{openai_path:.*}", self.openai_router),
                web.delete("/openai/v1/{openai_path:.*}", self.openai_router),
                web.get("/v1/{openai_path:.*}", self.openai_router),
                web.post("/v1/{openai_path:.*}", self.openai_router),
                web.put("/v1/{openai_path:.*}", self.openai_router),
                web.delete("/v1/{openai_path:.*}", self.openai_router),
                web.get("/anthropic/v1/{anthropic_path:.*}", self.anthropic_router),
                web.post("/anthropic/v1/{anthropic_path:.*}", self.anthropic_router),
                web.put("/anthropic/v1/{anthropic_path:.*}", self.anthropic_router),
                web.delete("/anthropic/v1/{anthropic_path:.*}", self.anthropic_router),
            ]
        )

    @property
    def usage_collector(self) -> UsageCollector:
        return self._usage_collector

    @property
    def host(self) -> str:
        return self._bound_host

    @property
    def port(self) -> int:
        return self._bound_port

    @property
    def base_url(self) -> str:
        return f"http://{_resolve_client_host(self._bound_host)}:{self._bound_port}"

    def env(self) -> dict[str, str]:
        return build_local_proxy_env(
            base_url=self.base_url,
            bearer_token=self._bearer_token,
            insecure=self._insecure,
        )

    async def start(self) -> None:
        if self._runner is not None:
            return

        if self._owns_session and self._http_session.closed:
            self._http_session = new_client_session()

        self._runner = web.AppRunner(self._app, access_log=None)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
        except Exception:
            if self._runner is not None:
                await self._runner.cleanup()
            self._runner = None
            self._site = None
            if self._owns_session and not self._http_session.closed:
                await self._http_session.close()
            raise

        if self._site._server is not None and self._site._server.sockets:
            socket = self._site._server.sockets[0]
            sock_name = socket.getsockname()
            if isinstance(sock_name, tuple):
                self._bound_host = str(sock_name[0])
                self._bound_port = int(sock_name[1])

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

        if self._owns_session and not self._http_session.closed:
            await self._http_session.close()

    async def healthz(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({"ok": True})

    def _extract_request_token(self, request: web.Request) -> str | None:
        auth_header = request.headers.get("Authorization")
        if isinstance(auth_header, str) and auth_header.startswith("Bearer "):
            return auth_header.removeprefix("Bearer ").strip()

        api_key = request.headers.get("x-api-key")
        if isinstance(api_key, str):
            normalized = api_key.strip()
            if normalized != "":
                return normalized

        return None

    def _ensure_local_auth(self, request: web.Request) -> None:
        if self._insecure:
            return

        request_token = self._extract_request_token(request)
        if request_token is None or self._bearer_token is None:
            raise web.HTTPUnauthorized(text="Missing local proxy bearer token.")

        if not secrets.compare_digest(request_token, self._bearer_token):
            raise web.HTTPUnauthorized(text="Invalid local proxy bearer token.")

    async def _prepare_upstream_headers(
        self,
        request: web.Request,
    ) -> dict[str, str]:
        upstream_bearer_token = await self._upstream_bearer_token_provider()
        if (
            not isinstance(upstream_bearer_token, str)
            or upstream_bearer_token.strip() == ""
        ):
            raise web.HTTPUnauthorized(
                text=(
                    "Unable to acquire a MeshAgent bearer token. "
                    "Set MESHAGENT_TOKEN or run meshagent auth login."
                )
            )

        headers = dict(request.headers)
        headers.pop("Host", None)
        headers.pop("Authorization", None)
        headers.pop("x-api-key", None)
        headers[MESHAGENT_PROJECT_ID_HEADER] = self._project_id
        headers["Authorization"] = f"Bearer {upstream_bearer_token.strip()}"
        return headers

    async def _record_usage(
        self,
        *,
        request_id: str,
        usage: Any | None,
    ) -> UsageEvent | None:
        if usage is None:
            return None
        return await self._usage_collector.record_model_usage(
            usage,
            request_id=request_id,
        )

    async def _record_request_activity(
        self,
        *,
        provider: str,
        transport: Literal["http", "websocket"],
        method: str,
        path: str,
        result: str,
        request_id: str,
        status: int | None = None,
    ) -> None:
        await self._usage_collector.record_request_activity(
            provider=provider,
            transport=transport,
            method=method,
            path=path,
            result=result,
            request_id=request_id,
            status=status,
        )

    async def openai_router(self, request: web.Request) -> web.StreamResponse:
        request_id = str(uuid4())
        request_path = request.path

        try:
            self._ensure_local_auth(request)

            v1_index = request.url.path.index("/v1")
            api_path = request.url.path[v1_index:]
            if not is_openai_path_allowed(api_path):
                raise web.HTTPBadRequest(
                    text=f"Unsupported OpenAI API path '{api_path}'."
                )

            query_string = request.url.query_string
            upstream_http_url = f"{self._api_base_url}/openai{api_path}"
            if query_string:
                upstream_http_url = f"{upstream_http_url}?{query_string}"

            upstream_headers = await self._prepare_upstream_headers(request)

            if is_websocket_request(request):
                if not is_openai_websocket_path_allowed(api_path):
                    raise web.HTTPBadRequest(
                        text=f"Unsupported OpenAI websocket path '{api_path}'."
                    )

                upstream_headers.pop("Origin", None)
                upstream_headers.pop("Content-Type", None)
                upstream_headers.pop("Content-Length", None)
                upstream_headers.pop("Transfer-Encoding", None)

                async def _on_websocket_complete(
                    outcome: ProxyWebSocketOutcome,
                ) -> None:
                    await self._record_request_activity(
                        provider="openai",
                        transport="websocket",
                        method=request.method,
                        path=request_path,
                        status=outcome.status,
                        result=_summarize_websocket_result(outcome),
                        request_id=request_id,
                    )

                if api_path == "/v1/realtime":
                    realtime_model = request.url.query.get("model")
                    transcription_model_state: dict[str, str | None] = {"model": None}
                    tracked_response_ids: set[str] = set()
                    tracked_event_ids: set[str] = set()

                    async def _on_client_event(msg: aiohttp.WSMessage) -> None:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            return
                        try:
                            payload = json.loads(msg.data)
                        except Exception:
                            return
                        if not isinstance(payload, dict):
                            return
                        transcription_model = (
                            extract_openai_transcription_model_from_session(
                                payload.get("session")
                            )
                        )
                        if transcription_model is not None:
                            transcription_model_state["model"] = transcription_model

                    async def _on_upstream_event(msg: aiohttp.WSMessage) -> None:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            return
                        try:
                            payload = json.loads(msg.data)
                        except Exception:
                            return
                        if not isinstance(payload, dict):
                            return

                        transcription_model = (
                            extract_openai_transcription_model_from_session(
                                payload.get("session")
                            )
                        )
                        if transcription_model is not None:
                            transcription_model_state["model"] = transcription_model

                        event_id = payload.get("event_id")
                        if isinstance(event_id, str) and event_id in tracked_event_ids:
                            return
                        if isinstance(event_id, str):
                            tracked_event_ids.add(event_id)

                        if payload.get("type") == "response.done":
                            response_payload = payload.get("response")
                            if isinstance(response_payload, dict):
                                response_id = response_payload.get("id")
                                if (
                                    isinstance(response_id, str)
                                    and response_id in tracked_response_ids
                                ):
                                    return
                                if isinstance(response_id, str):
                                    tracked_response_ids.add(response_id)

                        usage = extract_openai_realtime_usage(
                            default_model=realtime_model
                            if isinstance(realtime_model, str)
                            else None,
                            transcription_model=transcription_model_state.get("model"),
                            event=payload,
                        )
                        await self._record_usage(request_id=request_id, usage=usage)

                    return await proxy_websocket_request(
                        request=request,
                        http_session=self._http_session,
                        upstream_http_url=upstream_http_url,
                        heartbeat=self._websocket_heartbeat,
                        upstream_headers=upstream_headers,
                        on_client_event=_on_client_event,
                        on_upstream_event=_on_upstream_event,
                        on_complete=_on_websocket_complete,
                    )

                responses_model_state: dict[str, str | None] = {"model": None}
                responses_service_tier_state: dict[str, str | None] = {
                    "service_tier": None
                }
                tracked_response_ids: set[str] = set()
                tracked_event_ids: set[str] = set()

                async def _on_client_event(msg: aiohttp.WSMessage) -> None:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        return
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        return
                    if (
                        not isinstance(payload, dict)
                        or payload.get("type") != "response.create"
                    ):
                        return

                    responses_model_state["model"] = None
                    responses_service_tier_state["service_tier"] = None

                    response_config = payload.get("response")
                    if not isinstance(response_config, dict):
                        response_config = None

                    model = payload.get("model")
                    if not isinstance(model, str) and response_config is not None:
                        model = response_config.get("model")
                    if isinstance(model, str) and model.strip() != "":
                        responses_model_state["model"] = model.strip()

                    service_tier = payload.get("service_tier")
                    if (
                        not isinstance(service_tier, str)
                        and response_config is not None
                    ):
                        service_tier = response_config.get("service_tier")
                    if isinstance(service_tier, str) and service_tier.strip() != "":
                        responses_service_tier_state["service_tier"] = (
                            service_tier.strip()
                        )

                async def _on_upstream_event(msg: aiohttp.WSMessage) -> None:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        return
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        return
                    if not isinstance(payload, dict):
                        return

                    event_id = payload.get("event_id")
                    if isinstance(event_id, str) and event_id in tracked_event_ids:
                        return
                    if isinstance(event_id, str):
                        tracked_event_ids.add(event_id)

                    if payload.get("type") not in {
                        "response.completed",
                        "response.done",
                    }:
                        return

                    response_payload = payload.get("response")
                    if not isinstance(response_payload, dict):
                        return

                    response_id = response_payload.get("id")
                    if (
                        isinstance(response_id, str)
                        and response_id in tracked_response_ids
                    ):
                        return
                    if isinstance(response_id, str):
                        tracked_response_ids.add(response_id)

                    response_model = response_payload.get("model")
                    if (
                        not isinstance(response_model, str)
                        or response_model.strip() == ""
                    ):
                        response_model = responses_model_state.get("model")
                    if (
                        not isinstance(response_model, str)
                        or response_model.strip() == ""
                    ):
                        return

                    request_payload: dict[str, str] = {
                        "model": responses_model_state.get("model")
                        or response_model.strip()
                    }
                    service_tier = responses_service_tier_state.get("service_tier")
                    if isinstance(service_tier, str) and service_tier.strip() != "":
                        request_payload["service_tier"] = service_tier.strip()

                    usage = extract_openai_completion_usage(
                        model=response_model.strip(),
                        request=request_payload,
                        response=response_payload,
                    )
                    await self._record_usage(request_id=request_id, usage=usage)

                return await proxy_websocket_request(
                    request=request,
                    http_session=self._http_session,
                    upstream_http_url=upstream_http_url,
                    heartbeat=self._websocket_heartbeat,
                    upstream_headers=upstream_headers,
                    on_client_event=_on_client_event,
                    on_upstream_event=_on_upstream_event,
                    on_complete=_on_websocket_complete,
                )

            request_data = await request.read()
            content_type = request.headers.get("Content-Type") or ""
            if content_type.startswith("application/json"):
                parsed_request = json.loads(request_data)
                if isinstance(parsed_request, dict):
                    json_request = parsed_request
                    model = json_request.get("model")
                    if not isinstance(model, str):
                        model = None
                else:
                    json_request = None
                    model = None
            else:
                json_request = None
                model = None

            async with self._http_session.request(
                request.method,
                upstream_http_url,
                headers=upstream_headers,
                data=request_data,
            ) as resp:
                proxied_headers = filter_proxied_response_headers(resp.headers)
                response = web.StreamResponse(
                    status=resp.status,
                    headers=proxied_headers,
                )
                await response.prepare(request)

                data = bytearray()
                lines: list[str] = []
                hanging_line = ""
                stream = resp.content_type == "text/event-stream"

                async for chunk in resp.content.iter_any():
                    if stream:
                        chunk_lines = chunk.decode("utf8").splitlines(keepends=True)
                        for line in chunk_lines:
                            if hanging_line:
                                line = hanging_line + line
                                hanging_line = ""

                            if line.endswith("\n") or line.endswith("\r"):
                                if line.strip() == "":
                                    if not lines:
                                        continue

                                    current_event = SSEEvent.parse("".join(lines))
                                    lines = []

                                    if (
                                        current_event.data
                                        and json_request is not None
                                        and model is not None
                                    ):
                                        try:
                                            payload = json.loads(current_event.data)
                                        except Exception:
                                            payload = None

                                        if isinstance(payload, dict):
                                            if (
                                                current_event.event
                                                == "response.completed"
                                            ):
                                                usage = extract_openai_completion_usage(
                                                    model=model,
                                                    request=json_request,
                                                    response=payload.get("response")
                                                    or payload,
                                                )
                                                await self._record_usage(
                                                    request_id=request_id,
                                                    usage=usage,
                                                )
                                            elif (
                                                api_path == "/v1/chat/completions"
                                                and "usage" in payload
                                            ):
                                                usage = extract_openai_completion_usage(
                                                    model=model,
                                                    request=json_request,
                                                    response=payload,
                                                )
                                                await self._record_usage(
                                                    request_id=request_id,
                                                    usage=usage,
                                                )
                                elif not line.startswith(":"):
                                    lines.append(line)
                            else:
                                hanging_line = line
                        await response.write(chunk)
                        continue

                    data.extend(chunk)

                if stream:
                    await self._record_request_activity(
                        provider="openai",
                        transport="http",
                        method=request.method,
                        path=request_path,
                        status=resp.status,
                        result=_summarize_http_result(
                            status=resp.status,
                            json_payload=None,
                            body=b"",
                            stream=True,
                        ),
                        request_id=request_id,
                    )
                    return response

                json_response: dict[str, Any] | None = None
                content_header = resp.headers.get("Content-Type", "")
                if resp.content_type == "application/json" or content_header.startswith(
                    "application/json"
                ):
                    try:
                        json_response = json.loads(data)
                    except Exception:
                        json_response = None

                body = bytes(data)
                if isinstance(json_response, dict):
                    usage = None
                    if (
                        model is not None
                        and json_request is not None
                        and "usage" in json_response
                    ):
                        usage = extract_openai_completion_usage(
                            model=model,
                            request=json_request,
                            response=json_response,
                        )
                    elif (
                        api_path.startswith("/v1/images/")
                        and model is not None
                        and json_request is not None
                    ):
                        usage = extract_openai_image_usage(
                            model=model,
                            request=json_request,
                            response=json_response,
                        )

                    await self._record_usage(request_id=request_id, usage=usage)
                    body = json.dumps(json_response).encode("utf-8")

                await response.write(body)

                if (
                    api_path == "/v1/audio/speech"
                    and model is not None
                    and json_request is not None
                ):
                    usage = extract_openai_audio_speech_usage(
                        model=model,
                        request=json_request,
                    )
                    await self._record_usage(request_id=request_id, usage=usage)

                await self._record_request_activity(
                    provider="openai",
                    transport="http",
                    method=request.method,
                    path=request_path,
                    status=resp.status,
                    result=_summarize_http_result(
                        status=resp.status,
                        json_payload=json_response,
                        body=body,
                        stream=False,
                    ),
                    request_id=request_id,
                )

                return response
        except web.HTTPException as ex:
            await self._record_request_activity(
                provider="openai",
                transport="websocket" if is_websocket_request(request) else "http",
                method=request.method,
                path=request_path,
                status=ex.status,
                result=_normalize_activity_text(
                    ex.text or ex.reason,
                    fallback=_status_fallback(ex.status, failed=True),
                ),
                request_id=request_id,
            )
            raise
        except Exception as ex:
            logger.exception("OpenAI proxy request failed", exc_info=ex)
            await self._record_request_activity(
                provider="openai",
                transport="websocket" if is_websocket_request(request) else "http",
                method=request.method,
                path=request_path,
                status=500,
                result=_normalize_activity_text(
                    str(ex),
                    fallback=ex.__class__.__name__,
                ),
                request_id=request_id,
            )
            raise

    async def anthropic_router(self, request: web.Request) -> web.StreamResponse:
        request_id = str(uuid4())
        request_path = request.path

        try:
            self._ensure_local_auth(request)

            v1_index = request.url.path.index("/v1")
            api_path = request.url.path[v1_index:]
            if not is_anthropic_path_allowed(api_path):
                raise web.HTTPBadRequest(
                    text=f"Unsupported Anthropic API path '{api_path}'."
                )

            upstream_http_url = f"{self._api_base_url}/anthropic{api_path}"
            if request.url.query_string:
                upstream_http_url = f"{upstream_http_url}?{request.url.query_string}"

            upstream_headers = await self._prepare_upstream_headers(request)
            request_data = await request.read()

            content_type = request.headers.get("Content-Type") or ""
            if content_type.startswith("application/json"):
                parsed_request = json.loads(request_data)
                if isinstance(parsed_request, dict):
                    json_request = parsed_request
                    model = json_request.get("model")
                    if not isinstance(model, str):
                        model = None
                else:
                    json_request = None
                    model = None
            else:
                json_request = None
                model = None

            async with self._http_session.request(
                request.method,
                upstream_http_url,
                headers=upstream_headers,
                data=request_data,
            ) as resp:
                proxied_headers = filter_proxied_response_headers(resp.headers)
                response = web.StreamResponse(
                    status=resp.status,
                    headers=proxied_headers,
                )
                await response.prepare(request)

                data = bytearray()
                lines: list[str] = []
                hanging_line = ""
                stream = resp.content_type == "text/event-stream"
                stream_usage: dict[str, float] = {}

                async for chunk in resp.content.iter_any():
                    if stream:
                        chunk_lines = chunk.decode("utf8").splitlines(keepends=True)
                        for line in chunk_lines:
                            if hanging_line:
                                line = hanging_line + line
                                hanging_line = ""

                            if line.endswith("\n") or line.endswith("\r"):
                                if line.strip() == "":
                                    if not lines:
                                        continue

                                    current_event = SSEEvent.parse("".join(lines))
                                    lines = []

                                    if (
                                        current_event.data
                                        and json_request is not None
                                        and model is not None
                                    ):
                                        try:
                                            payload = json.loads(current_event.data)
                                        except Exception:
                                            payload = None

                                        if isinstance(payload, dict):
                                            usage = payload.get("usage")
                                            if not isinstance(usage, dict):
                                                message = payload.get("message")
                                                if isinstance(message, dict):
                                                    usage = message.get("usage")

                                            if isinstance(usage, dict):
                                                merge_cumulative_usage(
                                                    stream_usage,
                                                    usage,
                                                )
                                elif not line.startswith(":"):
                                    lines.append(line)
                            else:
                                hanging_line = line
                        await response.write(chunk)
                        continue

                    data.extend(chunk)

                if stream:
                    if stream_usage and json_request is not None and model is not None:
                        usage = extract_anthropic_completion_usage(
                            model=model,
                            request=json_request,
                            response={"usage": stream_usage},
                        )
                        await self._record_usage(request_id=request_id, usage=usage)

                    await self._record_request_activity(
                        provider="anthropic",
                        transport="http",
                        method=request.method,
                        path=request_path,
                        status=resp.status,
                        result=_summarize_http_result(
                            status=resp.status,
                            json_payload=None,
                            body=b"",
                            stream=True,
                        ),
                        request_id=request_id,
                    )
                    return response

                json_response: dict[str, Any] | None = None
                content_header = resp.headers.get("Content-Type", "")
                if resp.content_type == "application/json" or content_header.startswith(
                    "application/json"
                ):
                    try:
                        json_response = json.loads(data)
                    except Exception:
                        json_response = None

                body = bytes(data)
                if (
                    isinstance(json_response, dict)
                    and json_request is not None
                    and model is not None
                ):
                    usage = extract_anthropic_completion_usage(
                        model=model,
                        request=json_request,
                        response=json_response,
                    )
                    await self._record_usage(request_id=request_id, usage=usage)
                    body = json.dumps(json_response).encode("utf-8")

                await response.write(body)
                await self._record_request_activity(
                    provider="anthropic",
                    transport="http",
                    method=request.method,
                    path=request_path,
                    status=resp.status,
                    result=_summarize_http_result(
                        status=resp.status,
                        json_payload=json_response,
                        body=body,
                        stream=False,
                    ),
                    request_id=request_id,
                )
                return response
        except web.HTTPException as ex:
            await self._record_request_activity(
                provider="anthropic",
                transport="http",
                method=request.method,
                path=request_path,
                status=ex.status,
                result=_normalize_activity_text(
                    ex.text or ex.reason,
                    fallback=_status_fallback(ex.status, failed=True),
                ),
                request_id=request_id,
            )
            raise
        except Exception as ex:
            logger.exception("Anthropic proxy request failed", exc_info=ex)
            await self._record_request_activity(
                provider="anthropic",
                transport="http",
                method=request.method,
                path=request_path,
                status=500,
                result=_normalize_activity_text(
                    str(ex),
                    fallback=ex.__class__.__name__,
                ),
                request_id=request_id,
            )
            raise
