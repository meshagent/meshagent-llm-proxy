from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from meshagent.llm_proxy.pricing import (
    UsagePricingLineItem,
    build_usage_pricing_line_items,
    is_pricing_available,
    preprocess,
    pricing,
)


@dataclass(frozen=True)
class ModelUsage:
    provider: str
    model: str
    tokens: dict[str, float]


@dataclass(frozen=True)
class UsageEvent:
    provider: str
    model: str
    request_id: str | None
    tokens: dict[str, float]
    line_items: tuple[UsagePricingLineItem, ...]
    subtotal: float
    surcharge: float
    total: float
    timestamp: datetime


@dataclass(frozen=True)
class RequestActivityEvent:
    provider: str
    transport: Literal["http", "websocket"]
    method: str
    path: str
    status: int | None
    result: str
    request_id: str | None
    total: float | None
    timestamp: datetime


@dataclass(frozen=True)
class UsageSummary:
    provider: str
    model: str
    request_count: int
    tokens: dict[str, float]
    subtotal: float
    surcharge: float
    total: float
    last_seen: datetime


@dataclass(frozen=True)
class UsageSnapshot:
    total_requests: int
    subtotal: float
    surcharge: float
    total: float
    summaries: tuple[UsageSummary, ...]
    recent_events: tuple[UsageEvent, ...]
    recent_requests: tuple[RequestActivityEvent, ...]


@dataclass(slots=True)
class _MutableUsageSummary:
    provider: str
    model: str
    request_count: int = 0
    tokens: dict[str, float] = field(default_factory=dict)
    subtotal: float = 0.0
    surcharge: float = 0.0
    total: float = 0.0
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def resolve_usage_model(
    *,
    provider: str,
    response_model: str | None,
    request_model: str | None,
    service_tier: str | None = None,
) -> str | None:
    normalized_response_model = (
        response_model.strip() if isinstance(response_model, str) else ""
    )
    normalized_request_model = (
        request_model.strip() if isinstance(request_model, str) else ""
    )

    if normalized_response_model != "" and is_pricing_available(
        provider=provider,
        model=normalized_response_model,
        service_tier=service_tier,
    ):
        return normalized_response_model

    if normalized_request_model != "":
        return normalized_request_model

    if normalized_response_model != "":
        return normalized_response_model

    return None


def extract_openai_completion_usage(
    *,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any],
) -> ModelUsage | None:
    request_model = request.get("model")
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model.strip() == "":
        response_model = model

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    service_tier = request.get("service_tier")
    if not isinstance(service_tier, str):
        service_tier = None

    resolved_model = resolve_usage_model(
        provider="openai",
        response_model=response_model,
        request_model=request_model if isinstance(request_model, str) else None,
        service_tier=service_tier,
    )
    if not isinstance(resolved_model, str):
        return None

    tokens = preprocess(
        provider="openai",
        model=resolved_model,
        usage=usage,
        service_tier=service_tier,
    )
    if tokens is None:
        return None

    return ModelUsage(provider="openai", model=resolved_model, tokens=tokens)


def extract_anthropic_completion_usage(
    *,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any],
) -> ModelUsage | None:
    request_model = request.get("model")
    response_model = response.get("model")
    if not isinstance(response_model, str) or response_model.strip() == "":
        response_model = model

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    resolved_model = resolve_usage_model(
        provider="anthropic",
        response_model=response_model,
        request_model=request_model if isinstance(request_model, str) else None,
    )
    if not isinstance(resolved_model, str):
        return None

    tokens = preprocess(
        provider="anthropic",
        model=resolved_model,
        usage=usage,
    )
    if tokens is None:
        return None

    return ModelUsage(provider="anthropic", model=resolved_model, tokens=tokens)


def extract_openai_transcription_model_from_session(session_obj: object) -> str | None:
    if not isinstance(session_obj, dict):
        return None

    transcription = session_obj.get("input_audio_transcription")
    if not isinstance(transcription, dict):
        audio_options = session_obj.get("audio")
        if isinstance(audio_options, dict):
            input_options = audio_options.get("input")
            if isinstance(input_options, dict):
                nested_transcription = input_options.get("transcription")
                if isinstance(nested_transcription, dict):
                    transcription = nested_transcription

    if not isinstance(transcription, dict):
        return None

    model = transcription.get("model")
    if not isinstance(model, str):
        return None
    normalized = model.strip()
    return normalized or None


def extract_openai_realtime_usage(
    *,
    default_model: str | None,
    transcription_model: str | None,
    event: dict[str, Any],
) -> ModelUsage | None:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return None

    if event_type == "response.done":
        response = event.get("response")
        if not isinstance(response, dict):
            return None

        response_model = response.get("model")
        if not isinstance(response_model, str) or response_model.strip() == "":
            response_model = default_model
        if not isinstance(response_model, str) or response_model.strip() == "":
            return None

        request_model = default_model
        if not isinstance(request_model, str) or request_model.strip() == "":
            request_model = response_model

        return extract_openai_completion_usage(
            model=response_model,
            request={"model": request_model},
            response=response,
        )

    if event_type != "conversation.item.input_audio_transcription.completed":
        return None

    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None

    response_model = event.get("model")
    if not isinstance(response_model, str) or response_model.strip() == "":
        response_model = transcription_model
    if not isinstance(response_model, str) or response_model.strip() == "":
        return None

    request_model = transcription_model
    if not isinstance(request_model, str) or request_model.strip() == "":
        request_model = response_model

    return extract_openai_completion_usage(
        model=response_model,
        request={"model": request_model},
        response={"model": response_model, "usage": usage},
    )


def merge_cumulative_usage(
    totals: dict[str, float], usage: Mapping[str, object]
) -> dict[str, float]:
    for key, value in usage.items():
        if not isinstance(value, int | float):
            continue
        totals[key] = max(totals.get(key, 0.0), float(value))
    return totals


def resolve_openai_image_pricing_key(
    *,
    model: str,
    size: str | None,
    quality: str | None,
) -> str | None:
    normalized_size = size.strip() if isinstance(size, str) and size.strip() else None
    normalized_quality = (
        quality.strip() if isinstance(quality, str) and quality.strip() else None
    )

    if model in {"gpt-image-1.5", "chatgpt-image-latest"}:
        resolved_size = normalized_size or "1024x1024"
        resolved_quality = (normalized_quality or "medium").lower()
        return f"images_{resolved_quality}_{resolved_size}"

    if model in {"gpt-image-1", "gpt-image-1-mini"}:
        resolved_size = normalized_size or "1024x1024"
        resolved_quality = (normalized_quality or "medium").lower()
        return f"images_{resolved_quality}_{resolved_size}"

    if model == "dall-e-3":
        resolved_size = normalized_size or "1024x1024"
        resolved_quality = (normalized_quality or "standard").lower()
        return f"images_{resolved_quality}_{resolved_size}"

    if model == "dall-e-2":
        resolved_size = normalized_size or "1024x1024"
        return f"images_{resolved_size}"

    return None


def extract_openai_image_usage(
    *,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any],
) -> ModelUsage | None:
    data_items = response.get("data")
    if not isinstance(data_items, list):
        return None

    image_count = len(data_items)
    if image_count <= 0:
        return None

    pricing_key = resolve_openai_image_pricing_key(
        model=model,
        size=request.get("size") if isinstance(request.get("size"), str) else None,
        quality=request.get("quality")
        if isinstance(request.get("quality"), str)
        else None,
    )
    if pricing_key is None:
        return None

    model_pricing = pricing.get("openai", {}).get(model)
    if not isinstance(model_pricing, dict) or pricing_key not in model_pricing:
        return None

    return ModelUsage(
        provider="openai",
        model=model,
        tokens={pricing_key: float(image_count)},
    )


def extract_openai_audio_speech_usage(
    *,
    model: str,
    request: dict[str, Any],
) -> ModelUsage | None:
    input_text = request.get("input")
    if not isinstance(input_text, str):
        return None

    model_pricing = pricing.get("openai", {}).get(model)
    if not isinstance(model_pricing, dict) or "input_characters" not in model_pricing:
        return None

    return ModelUsage(
        provider="openai",
        model=model,
        tokens={"input_characters": float(len(input_text))},
    )


class UsageCollector:
    def __init__(
        self,
        *,
        max_recent_events: int = 50,
        max_recent_requests: int = 100,
    ) -> None:
        self._lock = asyncio.Lock()
        self._events: deque[UsageEvent] = deque(maxlen=max_recent_events)
        self._request_activity: deque[RequestActivityEvent] = deque(
            maxlen=max_recent_requests
        )
        self._summaries: dict[tuple[str, str], _MutableUsageSummary] = {}
        self._request_totals: dict[str, float] = {}
        self._total_requests = 0
        self._subtotal = 0.0
        self._surcharge = 0.0
        self._total = 0.0

    async def record_usage(
        self,
        *,
        provider: str,
        model: str,
        tokens: dict[str, float],
        request_id: str | None = None,
    ) -> UsageEvent:
        line_items = tuple(
            build_usage_pricing_line_items(
                provider=provider,
                model=model,
                usage=tokens,
            )
        )
        subtotal = sum(item.amount for item in line_items if not item.is_surcharge)
        surcharge = sum(item.amount for item in line_items if item.is_surcharge)
        total = subtotal + surcharge
        timestamp = datetime.now(timezone.utc)

        event = UsageEvent(
            provider=provider,
            model=model,
            request_id=request_id,
            tokens=dict(tokens),
            line_items=line_items,
            subtotal=subtotal,
            surcharge=surcharge,
            total=total,
            timestamp=timestamp,
        )

        async with self._lock:
            self._events.append(event)
            self._total_requests += 1
            self._subtotal += subtotal
            self._surcharge += surcharge
            self._total += total
            if isinstance(request_id, str):
                self._request_totals[request_id] = (
                    self._request_totals.get(request_id, 0.0) + total
                )

            key = (provider, model)
            summary = self._summaries.get(key)
            if summary is None:
                summary = _MutableUsageSummary(provider=provider, model=model)
                self._summaries[key] = summary

            summary.request_count += 1
            summary.subtotal += subtotal
            summary.surcharge += surcharge
            summary.total += total
            summary.last_seen = timestamp
            for usage_type, quantity in tokens.items():
                summary.tokens[usage_type] = summary.tokens.get(
                    usage_type, 0.0
                ) + float(quantity)

        return event

    async def record_model_usage(
        self,
        usage: ModelUsage,
        *,
        request_id: str | None = None,
    ) -> UsageEvent:
        return await self.record_usage(
            provider=usage.provider,
            model=usage.model,
            tokens=usage.tokens,
            request_id=request_id,
        )

    async def record_request_activity(
        self,
        *,
        provider: str,
        transport: Literal["http", "websocket"],
        method: str,
        path: str,
        result: str,
        request_id: str | None = None,
        status: int | None = None,
        total: float | None = None,
    ) -> RequestActivityEvent:
        timestamp = datetime.now(timezone.utc)

        async with self._lock:
            resolved_total = total
            if isinstance(request_id, str):
                tracked_total = self._request_totals.pop(request_id, None)
                if resolved_total is None:
                    resolved_total = tracked_total

            event = RequestActivityEvent(
                provider=provider,
                transport=transport,
                method=method,
                path=path,
                status=status,
                result=result,
                request_id=request_id,
                total=resolved_total,
                timestamp=timestamp,
            )
            self._request_activity.append(event)

        return event

    async def snapshot(self) -> UsageSnapshot:
        async with self._lock:
            summaries = tuple(
                UsageSummary(
                    provider=summary.provider,
                    model=summary.model,
                    request_count=summary.request_count,
                    tokens=dict(summary.tokens),
                    subtotal=summary.subtotal,
                    surcharge=summary.surcharge,
                    total=summary.total,
                    last_seen=summary.last_seen,
                )
                for summary in sorted(
                    self._summaries.values(),
                    key=lambda summary: (
                        -summary.total,
                        summary.provider,
                        summary.model,
                    ),
                )
            )
            recent_events = tuple(reversed(self._events))
            recent_requests = tuple(reversed(self._request_activity))

            return UsageSnapshot(
                total_requests=self._total_requests,
                subtotal=self._subtotal,
                surcharge=self._surcharge,
                total=self._total,
                summaries=summaries,
                recent_events=recent_events,
                recent_requests=recent_requests,
            )
