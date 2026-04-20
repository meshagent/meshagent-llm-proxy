from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

import aiohttp
from aiohttp import web


logger = logging.getLogger("meshagent.llm_proxy.proxy")

HOP_BY_HOP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
        "content-encoding",
        "content-length",
    }
)
EXTRA_RESPONSE_HEADERS_TO_STRIP = frozenset(
    {
        "access-control-expose-headers",
        "access-control-allow-origin",
        "access-control-allow-credentials",
        "access-control-max-age",
    }
)
HOP_BY_HOP_WEBSOCKET_HEADERS = frozenset(
    {
        "connection",
        "upgrade",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
        "sec-websocket-accept",
        "proxy-connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
    }
)


class ProxyWebSocketClientError(Exception):
    def __init__(self, payload: dict):
        super().__init__("websocket client event rejected")
        self.payload = payload


@dataclass(frozen=True)
class ProxyWebSocketOutcome:
    status: int | None
    client_close_code: int | None
    upstream_close_code: int | None
    error: str | None


def filter_proxied_response_headers(
    headers: Mapping[str, str],
    *,
    remove_cors: bool = False,
) -> dict[str, str]:
    blocked_headers = HOP_BY_HOP_RESPONSE_HEADERS
    if remove_cors:
        blocked_headers = blocked_headers | EXTRA_RESPONSE_HEADERS_TO_STRIP

    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in blocked_headers
    }


def http_to_websocket_url(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://") :]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://") :]
    return http_url


async def proxy_websocket_request(
    *,
    request: web.Request,
    http_session: aiohttp.ClientSession,
    upstream_http_url: str,
    heartbeat: float | None = None,
    upstream_headers: dict[str, str],
    on_client_event: Callable[[aiohttp.WSMessage], Awaitable[str | bytes | None]]
    | None = None,
    on_upstream_event: Callable[[aiohttp.WSMessage], Awaitable[str | bytes | None]]
    | None = None,
    on_complete: Callable[[ProxyWebSocketOutcome], Awaitable[None]] | None = None,
) -> web.StreamResponse:
    async def _pump_ws(
        src: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
        dst: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
        on_event: Callable[[aiohttp.WSMessage], Awaitable[str | bytes | None]]
        | None = None,
    ) -> None:
        async for msg in src:
            event_payload: str | bytes | None = None
            if on_event is not None:
                try:
                    event_payload = await on_event(msg)
                except ProxyWebSocketClientError as ex:
                    logger.info(
                        "websocket client event rejected",
                        extra={"payload": ex.payload},
                    )
                    if isinstance(src, web.WebSocketResponse):
                        await src.send_str(json.dumps(ex.payload))
                    continue
                except Exception as ex:
                    logger.warning("websocket event listener failed", exc_info=ex)

            if msg.type == aiohttp.WSMsgType.TEXT:
                if isinstance(event_payload, str):
                    await dst.send_str(event_payload)
                else:
                    await dst.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                if isinstance(event_payload, bytes):
                    await dst.send_bytes(event_payload)
                else:
                    await dst.send_bytes(msg.data)
            elif msg.type == aiohttp.WSMsgType.PING:
                await dst.ping()
            elif msg.type == aiohttp.WSMsgType.PONG:
                pass
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                try:
                    await dst.close()
                except Exception:
                    pass
                return
            elif msg.type == aiohttp.WSMsgType.ERROR:
                return

    client_ws = web.WebSocketResponse(
        autoping=True,
        heartbeat=heartbeat,
        max_msg_size=0,
    )
    await client_ws.prepare(request)

    protocols = request.headers.get("Sec-WebSocket-Protocol")
    protocol_list = [p.strip() for p in protocols.split(",")] if protocols else None
    upstream_ws_url = http_to_websocket_url(upstream_http_url)
    outcome = ProxyWebSocketOutcome(
        status=None,
        client_close_code=None,
        upstream_close_code=None,
        error=None,
    )

    try:
        async with http_session.ws_connect(
            upstream_ws_url,
            headers={
                key: value
                for key, value in upstream_headers.items()
                if key.lower() not in HOP_BY_HOP_WEBSOCKET_HEADERS
            },
            protocols=protocol_list,
            autoping=True,
            heartbeat=heartbeat,
            max_msg_size=0,
        ) as upstream_ws:
            client_task = asyncio.create_task(
                _pump_ws(client_ws, upstream_ws, on_client_event)
            )
            upstream_task = asyncio.create_task(
                _pump_ws(upstream_ws, client_ws, on_upstream_event)
            )

            _, pending = await asyncio.wait(
                {client_task, upstream_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            outcome = ProxyWebSocketOutcome(
                status=101,
                client_close_code=client_ws.close_code,
                upstream_close_code=upstream_ws.close_code,
                error=None,
            )

    except aiohttp.WSServerHandshakeError as ex:
        error_message = str(ex).strip() or "websocket handshake failed"
        outcome = ProxyWebSocketOutcome(
            status=ex.status,
            client_close_code=client_ws.close_code,
            upstream_close_code=None,
            error=error_message,
        )
        logger.exception("WebSocket tunnel failed to %s", upstream_ws_url, exc_info=ex)
        try:
            await client_ws.close(code=1011, message=b"tunnel error")
        except Exception:
            pass

    except Exception as ex:
        error_message = str(ex).strip() or ex.__class__.__name__
        outcome = ProxyWebSocketOutcome(
            status=None,
            client_close_code=client_ws.close_code,
            upstream_close_code=None,
            error=error_message,
        )
        logger.exception("WebSocket tunnel failed to %s", upstream_ws_url, exc_info=ex)
        try:
            await client_ws.close(code=1011, message=b"tunnel error")
        except Exception:
            pass
    finally:
        if on_complete is not None:
            try:
                await on_complete(outcome)
            except Exception as ex:
                logger.warning("websocket completion listener failed", exc_info=ex)

    return client_ws
