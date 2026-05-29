import pytest

from meshagent.llm_proxy.local_proxy import build_local_proxy_env
from meshagent.llm_proxy.usage import (
    UsageCollector,
    extract_anthropic_completion_usage,
    extract_openai_completion_usage,
    extract_openai_realtime_usage,
    extract_openai_transcription_model_from_session,
)


def test_extract_openai_completion_usage_prefers_priced_response_model() -> None:
    usage = extract_openai_completion_usage(
        model="gpt-5.4-2026-03-05",
        request={"model": "gpt-5.4"},
        response={
            "model": "gpt-5.4-2026-03-05",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-5.4-2026-03-05"
    assert usage.tokens == {"input_tokens": 10.0, "output_tokens": 2.0}


@pytest.mark.parametrize(
    "response_usage",
    [
        {
            "prompt_tokens": 5084,
            "prompt_tokens_details": {"cached_tokens": 4864},
            "completion_tokens": 2,
            "total_tokens": 5086,
        },
        {
            "input_tokens": 5084,
            "input_tokens_details": {"cached_tokens": 4864},
            "output_tokens": 2,
            "total_tokens": 5086,
        },
    ],
)
def test_extract_openai_completion_usage_splits_cached_aggregate_input_tokens(
    response_usage: dict,
) -> None:
    usage = extract_openai_completion_usage(
        model="gpt-4.1",
        request={"model": "gpt-4.1"},
        response={"model": "gpt-4.1", "usage": response_usage},
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-4.1"
    assert usage.tokens == {
        "cached_tokens": 4864.0,
        "input_tokens": 220.0,
        "output_tokens": 2.0,
        "total_tokens": 5086.0,
    }


def test_extract_openai_completion_usage_falls_back_to_request_model() -> None:
    usage = extract_openai_completion_usage(
        model="gpt-5.4-2099-01-01",
        request={"model": "gpt-5.4", "service_tier": "flex"},
        response={
            "model": "gpt-5.4-2099-01-01",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-5.4"
    assert usage.tokens == {"input_tokens_flex": 10.0, "output_tokens_flex": 2.0}


@pytest.mark.parametrize(
    ("response_model", "request_model"),
    [
        ("claude-opus-4-6-20990101", "claude-opus-4-6"),
        ("claude-opus-4-7-20990416", "claude-opus-4-7"),
        ("claude-opus-4-8-20990528", "claude-opus-4-8"),
    ],
)
def test_extract_anthropic_completion_usage_falls_back_to_request_model(
    response_model: str, request_model: str
) -> None:
    usage = extract_anthropic_completion_usage(
        model=response_model,
        request={"model": request_model},
        response={
            "model": response_model,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    )

    assert usage is not None
    assert usage.provider == "anthropic"
    assert usage.model == request_model
    assert usage.tokens == {
        "input_tokens": 10.0,
        "output_tokens": 2.0,
    }


def test_extract_anthropic_completion_usage_keeps_prompt_cache_tokens_separate() -> (
    None
):
    usage = extract_anthropic_completion_usage(
        model="claude-sonnet-4-6",
        request={"model": "claude-sonnet-4-6"},
        response={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 900,
                "output_tokens": 5,
            },
        },
    )

    assert usage is not None
    assert usage.provider == "anthropic"
    assert usage.model == "claude-sonnet-4-6"
    assert usage.tokens == {
        "input_tokens": 100.0,
        "cache_creation_input_tokens": 1000.0,
        "cache_read_input_tokens": 900.0,
        "output_tokens": 5.0,
    }


def test_extract_openai_completion_usage_drops_zero_reasoning_tokens() -> None:
    usage = extract_openai_completion_usage(
        model="gpt-5.4-2026-03-05",
        request={"model": "gpt-5.4"},
        response={
            "model": "gpt-5.4-2026-03-05",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "reasoning_tokens": 0,
            },
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-5.4-2026-03-05"
    assert usage.tokens == {"input_tokens": 10.0, "output_tokens": 2.0}


def test_extract_openai_transcription_model_from_nested_realtime_session() -> None:
    assert (
        extract_openai_transcription_model_from_session(
            {"audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}}}
        )
        == "gpt-realtime-whisper"
    )


def test_extract_openai_realtime_transcription_usage_prices_audio_seconds() -> None:
    usage = extract_openai_realtime_usage(
        default_model="gpt-realtime-2",
        transcription_model="gpt-realtime-whisper",
        event={
            "type": "conversation.item.input_audio_transcription.completed",
            "usage": {"audio_seconds": 30},
        },
    )

    assert usage is not None
    assert usage.provider == "openai"
    assert usage.model == "gpt-realtime-whisper"
    assert usage.tokens == {"audio_minutes": 0.5}


@pytest.mark.asyncio
async def test_usage_collector_prices_realtime_whisper_audio_minutes() -> None:
    collector = UsageCollector()

    await collector.record_usage(
        provider="openai",
        model="gpt-realtime-whisper",
        tokens={"audio_minutes": 2.0},
        request_id="req-transcribe",
    )

    snapshot = await collector.snapshot()

    assert snapshot.subtotal == pytest.approx(0.034)
    assert snapshot.surcharge == pytest.approx(0.0017)


def test_extract_anthropic_completion_usage_returns_none_for_zero_only_usage() -> None:
    usage = extract_anthropic_completion_usage(
        model="claude-sonnet-4-6",
        request={"model": "claude-sonnet-4-6"},
        response={
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    )

    assert usage is None


@pytest.mark.asyncio
async def test_usage_collector_tracks_llm_proxy_surcharge() -> None:
    collector = UsageCollector()

    await collector.record_usage(
        provider="openai",
        model="gpt-4.1",
        tokens={"input_tokens": 1_000.0, "output_tokens": 500.0},
        request_id="req-1",
    )

    snapshot = await collector.snapshot()

    assert snapshot.total_requests == 1
    assert snapshot.subtotal == pytest.approx(0.006)
    assert snapshot.surcharge == pytest.approx(0.0003)
    assert snapshot.total == pytest.approx(0.0063)
    assert len(snapshot.recent_events) == 1
    assert snapshot.recent_events[0].request_id == "req-1"
    assert snapshot.recent_events[0].surcharge == pytest.approx(0.0003)


@pytest.mark.asyncio
async def test_usage_collector_tracks_request_activity_with_aggregated_total() -> None:
    collector = UsageCollector()

    await collector.record_usage(
        provider="openai",
        model="gpt-4.1",
        tokens={"input_tokens": 1_000.0, "output_tokens": 500.0},
        request_id="req-2",
    )
    await collector.record_request_activity(
        provider="openai",
        transport="http",
        method="POST",
        path="/openai/v1/chat/completions",
        status=200,
        result="ok",
        request_id="req-2",
    )

    snapshot = await collector.snapshot()

    assert len(snapshot.recent_requests) == 1
    assert snapshot.recent_requests[0].request_id == "req-2"
    assert snapshot.recent_requests[0].status == 200
    assert snapshot.recent_requests[0].result == "ok"
    assert snapshot.recent_requests[0].total == pytest.approx(0.0063)


def test_build_local_proxy_env_uses_expected_base_urls() -> None:
    env = build_local_proxy_env(
        base_url="http://127.0.0.1:8766",
        bearer_token="local-token",
        insecure=False,
    )

    assert env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8766/openai/v1",
        "OPENAI_API_KEY": "local-token",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8766/anthropic",
        "ANTHROPIC_API_KEY": "local-token",
    }
